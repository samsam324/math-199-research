"""
Baseline / booster / LSTM / small transformer for the 3-class label.
Ported from phase 1. The architectures don't care about the feature semantics
or the bar cadence; only TABULAR_COLUMNS_DEFAULT changes (phase 1's set was
hourly; phase 2 uses a microstructure-derived set declared in ml_dataset.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainingConfig:
    max_train_samples: int = 20000
    max_test_samples: int = 10000
    dl_epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 7


def _load_xgb_classifier():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            max_depth=4, n_estimators=300, learning_rate=0.05,
            objective="multi:softprob", eval_metric="mlogloss",
            subsample=0.9, colsample_bytree=0.9, random_state=7,
        ), "xgboost"
    except Exception:
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=7), "sklearn_hist_gradient_boosting"


def _split_frame(samples: pd.DataFrame, test_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    samples = samples.sort_values("timestamp").copy()
    samples["timestamp"] = pd.to_datetime(samples["timestamp"], utc=True)
    test_start = samples["timestamp"].max() - pd.Timedelta(days=test_days)
    train = samples[samples["timestamp"] < test_start]
    test = samples[samples["timestamp"] >= test_start]
    if train.empty or test.empty:
        n = len(samples); cut = int(n * 0.8)
        train = samples.iloc[:cut]; test = samples.iloc[cut:]
    return train, test


def _balanced_tail(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sort_values("timestamp").tail(max_rows)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def trading_metrics(test: pd.DataFrame, pred_class: np.ndarray) -> Dict[str, float]:
    """Signal-quality metric. NOT a portfolio backtest; backtest.py is the real thing."""
    current = test["current_spread"].to_numpy(dtype=float)
    change = test["y_regression"].to_numpy(dtype=float)
    spread_sign = np.sign(current)
    signal = np.zeros_like(change)
    signal[pred_class == 0] = -spread_sign[pred_class == 0]
    signal[pred_class == 2] = spread_sign[pred_class == 2]
    pnl = signal * change
    active = signal != 0
    if len(pnl) == 0:
        return {}
    pnl_sd = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    pnl_mean_to_std = float(np.mean(pnl) / pnl_sd) if pnl_sd > 0 else 0.0
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    return {
        "trades": int(active.sum()),
        "mean_pnl": float(np.mean(pnl)),
        "total_pnl": float(np.sum(pnl)),
        "pnl_mean_to_std": pnl_mean_to_std,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((pnl[active] > 0).mean()) if active.any() else 0.0,
    }


def _prediction_frame(test: pd.DataFrame, model_name: str, pred: np.ndarray) -> pd.DataFrame:
    pred_df = test[["sample_id", "pair", "timestamp", "current_spread", "y_regression", "y_class"]].copy()
    pred_df["model"] = model_name
    pred_df["pred_class"] = pred.astype(int)
    return pred_df


def _metrics_for_predictions(model_name: str, train: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_test = test["y_class"].to_numpy(dtype=int)
    pred_df = _prediction_frame(test, model_name, pred)
    metrics = {"model": model_name, "train_samples": int(len(train)), "test_samples": int(len(test))}
    metrics.update(evaluate_predictions(y_test, pred))
    metrics.update(trading_metrics(pred_df, pred))
    return pred_df, metrics


def train_booster_on_split(train: pd.DataFrame, test: pd.DataFrame, tabular_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    x_train = train[tabular_columns].to_numpy(dtype=float)
    y_train = train["y_class"].to_numpy(dtype=int)
    x_test = test[tabular_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model, model_name = _load_xgb_classifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test).astype(int)
    return _metrics_for_predictions(model_name, train, test, pred)


def baseline_persist_class(test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pred = np.ones(len(test), dtype=int)
    return _metrics_for_predictions("persist_class", test.iloc[0:0], test, pred)


def baseline_majority_class(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    majority = int(train["y_class"].mode().iloc[0]) if not train.empty else 1
    return _metrics_for_predictions("majority_class", train, test, np.full(len(test), majority, dtype=int))


def baseline_random_stratified(train: pd.DataFrame, test: pd.DataFrame, seed: int = 7) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    counts = train["y_class"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0).to_numpy(dtype=float)
    probs = counts / counts.sum() if counts.sum() > 0 else np.array([1/3, 1/3, 1/3], dtype=float)
    pred = rng.choice(np.array([0, 1, 2], dtype=int), size=len(test), p=probs)
    return _metrics_for_predictions("random_stratified", train, test, pred)


def baseline_zscore_rule(test: pd.DataFrame, threshold: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, float]]:
    z = test["latest_spread_z"].to_numpy(dtype=float)
    pred = np.ones(len(test), dtype=int)
    pred[np.abs(z) >= threshold] = 0
    return _metrics_for_predictions("zscore_rule", test.iloc[0:0], test, pred)


# -------------------- Torch deep models --------------------

def _require_torch():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, DataLoader, TensorDataset


def _torch_device():
    torch, _, _, _ = _require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_torch_on_split(
    model_name: str, model_factory,
    train: pd.DataFrame, test: pd.DataFrame, sequences: np.ndarray,
    cfg: TrainingConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    train_ids = train["sample_id"].to_numpy(dtype=int)
    test_ids = test["sample_id"].to_numpy(dtype=int)
    x_train = sequences[train_ids].astype(np.float32)
    x_test = sequences[test_ids].astype(np.float32)
    y_train = train["y_class"].to_numpy(dtype=np.int64)
    y_test = test["y_class"].to_numpy(dtype=np.int64)

    flat = x_train.reshape(-1, x_train.shape[-1])
    mu = flat.mean(axis=0); sd = flat.std(axis=0); sd[sd == 0] = 1.0
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    device = _torch_device()
    model = model_factory(x_train.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.dl_epochs):
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()

    model.eval()
    preds: List[np.ndarray] = []
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=cfg.batch_size)
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    pred = np.concatenate(preds).astype(int)
    return _metrics_for_predictions(model_name, train, test, pred)


def _lstm_factory():
    torch, nn, _, _ = _require_torch()

    class LSTMClassifier(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Linear(64, 3)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])
    return LSTMClassifier


def _transformer_factory():
    torch, nn, _, _ = _require_torch()

    class TransformerClassifier(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.input = nn.Linear(n_features, 32)
            self.cls = nn.Parameter(torch.zeros(1, 1, 32))
            layer = nn.TransformerEncoderLayer(d_model=32, nhead=2, dim_feedforward=64, dropout=0.2, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(32, 3)
        def forward(self, x):
            z = self.input(x)
            cls = self.cls.expand(z.shape[0], -1, -1)
            z = torch.cat([cls, z], dim=1)
            z = self.encoder(z)
            return self.head(z[:, 0, :])
    return TransformerClassifier


def train_lstm_on_split(train: pd.DataFrame, test: pd.DataFrame, sequences: np.ndarray, cfg: TrainingConfig = TrainingConfig()) -> Tuple[pd.DataFrame, Dict[str, float]]:
    return _train_torch_on_split("lstm", _lstm_factory(), train, test, sequences, cfg)


def train_transformer_on_split(train: pd.DataFrame, test: pd.DataFrame, sequences: np.ndarray, cfg: TrainingConfig = TrainingConfig()) -> Tuple[pd.DataFrame, Dict[str, float]]:
    return _train_torch_on_split("transformer", _transformer_factory(), train, test, sequences, cfg)


def combine_results(results: Iterable[Tuple[pd.DataFrame, Dict[str, float]]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preds = []; metrics = []
    for pred_df, metric in results:
        preds.append(pred_df); metrics.append(metric)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(metrics)
