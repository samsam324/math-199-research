"""
4-layer transformer with cosine LR and early stopping. Ported from phase 1.
Used to test whether the small transformer's underperformance is a capacity
artifact (it was, in phase 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.modeling import TrainingConfig, _metrics_for_predictions, _torch_device


@dataclass(frozen=True)
class BigTransformerConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff: int = 512
    dropout: float = 0.2
    epochs: int = 20
    warmup_epochs: int = 1
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.10
    patience: int = 5
    seed: int = 7


def _sinusoidal_positions(length: int, d_model: int, torch):
    pe = torch.zeros(length, d_model)
    pos = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _build_model(n_features: int, window: int, cfg: BigTransformerConfig):
    import torch
    import torch.nn as nn

    class BigTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(n_features, cfg.d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.register_buffer("pos", _sinusoidal_positions(window + 1, cfg.d_model, torch))
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ff,
                dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, 3)
        def forward(self, x):
            z = self.proj(x)
            cls = self.cls.expand(z.shape[0], -1, -1)
            z = torch.cat([cls, z], dim=1)
            z = z + self.pos[: z.shape[1]].unsqueeze(0)
            z = self.encoder(z)
            z = self.norm(z[:, 0, :])
            return self.head(z)
    return BigTransformer()


def train_big_transformer_on_split(
    train: pd.DataFrame, test: pd.DataFrame, sequences: np.ndarray,
    cfg: BigTransformerConfig = BigTransformerConfig(),
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    train_ids = train["sample_id"].to_numpy(dtype=int)
    test_ids = test["sample_id"].to_numpy(dtype=int)
    x_train_full = sequences[train_ids].astype(np.float32)
    y_train_full = train["y_class"].to_numpy(dtype=np.int64)
    x_test = sequences[test_ids].astype(np.float32)
    y_test = test["y_class"].to_numpy(dtype=np.int64)

    flat = x_train_full.reshape(-1, x_train_full.shape[-1])
    mu = flat.mean(axis=0); sd = flat.std(axis=0); sd[sd == 0] = 1.0
    x_train_full = (x_train_full - mu) / sd
    x_test = (x_test - mu) / sd

    val_size = max(1, int(len(x_train_full) * cfg.val_fraction))
    x_train, x_val = x_train_full[:-val_size], x_train_full[-val_size:]
    y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]

    device = _torch_device()
    model = _build_model(x_train.shape[-1], window=x_train.shape[1], cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=cfg.batch_size, shuffle=False)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val_loss = float("inf"); best_state = None; epochs_since_improve = 0
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits = model(xb)
            loss = loss_fn(logits, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); scheduler.step()

        model.eval(); val_loss = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += float(loss_fn(logits, yb)) * len(yb); n += len(yb)
        val_loss /= max(1, n)
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval(); preds = []
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=cfg.batch_size, shuffle=False)
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    pred = np.concatenate(preds).astype(int)
    return _metrics_for_predictions("big_transformer", train, test, pred)
