"""
Train a transformer on the saved dataset and plot attention heatmaps for K
example test windows.

The transformer architecture matches `_transformer_factory` in src/modeling.py
but with attention weights returned alongside the prediction, so we can pull
the final-layer attention map and render it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_dataset import load_dataset
from src.modeling import TrainingConfig, _balanced_tail, _split_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer attention heatmaps on saved dataset.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-train-samples", type=int, default=12000)
    p.add_argument("--max-test-samples", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-examples", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples, sequences = load_dataset(Path(args.dataset_dir))
    cfg = TrainingConfig(
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        dl_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
    train, test = _split_frame(samples)
    train = _balanced_tail(train, cfg.max_train_samples)
    test = _balanced_tail(test, cfg.max_test_samples)
    train_ids = train["sample_id"].to_numpy(dtype=int)
    test_ids = test["sample_id"].to_numpy(dtype=int)

    x_train = sequences[train_ids].astype(np.float32)
    x_test = sequences[test_ids].astype(np.float32)
    y_train = train["y_class"].to_numpy(dtype=np.int64)

    flat = x_train.reshape(-1, x_train.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd[sd == 0] = 1.0
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

    class TransformerWithAttention(nn.Module):
        def __init__(self, n_features: int, d_model: int = 32, n_heads: int = 2, ff: int = 64, layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.input = nn.Linear(n_features, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.attn_layers = nn.ModuleList([
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(layers)
            ])
            self.ff_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d_model)) for _ in range(layers)
            ])
            self.norms_a = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
            self.norms_b = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
            self.head = nn.Linear(d_model, 3)
            self._last_attn: List[torch.Tensor] = []

        def forward(self, x, return_attn: bool = False):
            self._last_attn = []
            z = self.input(x)
            cls = self.cls.expand(z.shape[0], -1, -1)
            z = torch.cat([cls, z], dim=1)
            for attn, ff, n1, n2 in zip(self.attn_layers, self.ff_layers, self.norms_a, self.norms_b):
                out, w = attn(z, z, z, need_weights=True, average_attn_weights=True)
                if return_attn:
                    self._last_attn.append(w.detach().cpu())
                z = n1(z + out)
                z = n2(z + ff(z))
            logits = self.head(z[:, 0, :])
            return logits

    model = TransformerWithAttention(x_train.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for epoch in range(cfg.dl_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        print(f"epoch {epoch+1}/{cfg.dl_epochs}: loss={loss.item():.4f}")

    model.eval()
    # Pick the n-examples test windows with the highest |spread_z| at the last step
    pick = np.argsort(-np.abs(test["latest_spread_z"].to_numpy(dtype=float)))[: args.n_examples]
    with torch.no_grad():
        xb = torch.tensor(x_test[pick]).to(device)
        _ = model(xb, return_attn=True)
        attns = model._last_attn  # list of (B, T+1, T+1), one per layer

    final_layer = attns[-1].numpy()  # final layer averaged over heads
    for i, sample_idx in enumerate(pick):
        cls_row = final_layer[i, 0, 1:]  # attention from CLS to the 168 input steps
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.bar(np.arange(len(cls_row)), cls_row, color="#4c78a8", width=1.0)
        meta = test.iloc[sample_idx]
        ax.set_title(f"Transformer attention from CLS, pair={meta['pair']}, ts={meta['timestamp']}, true={int(meta['y_class'])}")
        ax.set_xlabel("hours back from prediction time (0 = oldest, last = most recent)")
        ax.set_ylabel("attention weight")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        out_path = out_dir / f"attn_{i:02d}_{meta['pair']}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
