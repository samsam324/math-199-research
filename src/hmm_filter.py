"""
HMM regime filter for pair-spread series.

Fit a 2-state Gaussian HMM on spread-derived features (z-score, first
difference, rolling vol) and identify which state is mean-reverting. The
identified state mask is then used as a trade filter on top of any predictor:
only allow trades when the most recent observation is in the mean-reverting
state. The "diverging" state is suppressed to flat (class 1).

The library dependency is hmmlearn; it is intentionally not imported at module
top-level so the rest of the codebase keeps working if it is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


HMM_FEATURE_COLUMNS = ["spread_z_168h", "spread_diff_1h", "spread_vol_24h"]


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 2
    covariance_type: str = "full"
    n_iter: int = 50
    seed: int = 7
    # Walk-forward refit: refit at the start of each test window. None means
    # one fit on the whole training period.
    refit_per_split: bool = True


def _require_hmmlearn():
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except Exception as exc:  # pragma: no cover - import error surfaces to caller
        raise ImportError(
            "hmmlearn is required for HMM regime filtering. Install with: pip install hmmlearn"
        ) from exc
    return GaussianHMM


def _feature_matrix(features: pd.DataFrame, columns: List[str] = HMM_FEATURE_COLUMNS) -> np.ndarray:
    arr = features[columns].to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def fit_hmm(
    train_features: pd.DataFrame,
    cfg: HMMConfig = HMMConfig(),
    columns: List[str] = HMM_FEATURE_COLUMNS,
):
    """
    Fit a Gaussian HMM on a per-pair training slice of the features table.
    Returns the fitted model.
    """
    GaussianHMM = _require_hmmlearn()
    X = _feature_matrix(train_features, columns)
    model = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.covariance_type,
        n_iter=cfg.n_iter,
        random_state=cfg.seed,
    )
    model.fit(X)
    return model


def identify_mean_reverting_state(model, train_features: pd.DataFrame, columns: List[str] = HMM_FEATURE_COLUMNS) -> int:
    """
    Pick which decoded state to call mean-reverting. Rule: the state with the
    lower variance on spread first-differences (more contained moves) is
    treated as mean-reverting. Ties break on absolute z-score mean (lower wins).
    """
    X = _feature_matrix(train_features, columns)
    states = model.predict(X)
    diff_col = columns.index("spread_diff_1h") if "spread_diff_1h" in columns else 0
    z_col = columns.index("spread_z_168h") if "spread_z_168h" in columns else 0
    diff_vol = []
    z_abs = []
    for s in range(model.n_components):
        mask = states == s
        if not mask.any():
            diff_vol.append(np.inf)
            z_abs.append(np.inf)
            continue
        diff_vol.append(float(np.std(X[mask, diff_col])))
        z_abs.append(float(np.mean(np.abs(X[mask, z_col]))))
    diff_vol_arr = np.asarray(diff_vol)
    z_abs_arr = np.asarray(z_abs)
    candidate = int(np.argmin(diff_vol_arr))
    ties = np.flatnonzero(diff_vol_arr == diff_vol_arr[candidate])
    if len(ties) > 1:
        candidate = int(ties[np.argmin(z_abs_arr[ties])])
    return candidate


def decode_states(model, features: pd.DataFrame, columns: List[str] = HMM_FEATURE_COLUMNS) -> np.ndarray:
    X = _feature_matrix(features, columns)
    return model.predict(X).astype(int)


def regime_mask_for_samples(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    cfg: HMMConfig = HMMConfig(),
    columns: List[str] = HMM_FEATURE_COLUMNS,
    split_train_end: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Produce a per-sample boolean mask: True if the timestamp falls inside the
    mean-reverting state, False otherwise. Fits one HMM per pair on the
    training slice (timestamps strictly before split_train_end) and decodes the
    full pair history.

    The returned series is indexed on samples.index.
    """
    if features.empty or samples.empty:
        return pd.Series(False, index=samples.index)

    features = features.copy()
    samples = samples.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    samples["timestamp"] = pd.to_datetime(samples["timestamp"], utc=True)

    mask = pd.Series(False, index=samples.index)
    for pair, pair_features in features.groupby("pair", sort=False):
        pair_samples = samples[samples["pair"] == pair]
        if pair_samples.empty:
            continue

        pair_features = pair_features.sort_values("timestamp").reset_index(drop=True)
        if split_train_end is not None:
            train_slice = pair_features[pair_features["timestamp"] < split_train_end]
        else:
            train_slice = pair_features

        if len(train_slice) < max(50, cfg.n_states * 20):
            continue

        try:
            model = fit_hmm(train_slice, cfg=cfg, columns=columns)
        except Exception:
            continue

        mean_rev_state = identify_mean_reverting_state(model, train_slice, columns=columns)
        decoded = decode_states(model, pair_features, columns=columns)
        ts_to_state = dict(zip(pair_features["timestamp"], decoded))

        for idx, ts in pair_samples["timestamp"].items():
            state = ts_to_state.get(ts)
            if state is not None and int(state) == int(mean_rev_state):
                mask.loc[idx] = True
    return mask


def apply_regime_filter(pred_class: np.ndarray, mean_rev_mask: np.ndarray) -> np.ndarray:
    """
    Suppress non-mean-reverting bars to class 1 (flat). Class 1 is no-trade in
    the existing trading_metrics convention, so this is equivalent to skipping
    the bar.
    """
    out = np.asarray(pred_class, dtype=int).copy()
    flat_mask = ~np.asarray(mean_rev_mask, dtype=bool)
    out[flat_mask] = 1
    return out


def hmm_ablation_for_predictions(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    samples: pd.DataFrame,
    cfg: HMMConfig = HMMConfig(),
    columns: List[str] = HMM_FEATURE_COLUMNS,
    split_train_end: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, pd.Series]:
    """
    Convenience wrapper: given a prediction frame (from any model) with a
    `pred_class` column and `sample_id` / `pair` / `timestamp` columns, build
    the mean-reverting mask for those rows and return filtered predictions.
    """
    sample_lookup = samples.set_index("sample_id")
    aligned = predictions.set_index("sample_id").join(
        sample_lookup[["pair", "timestamp"]],
        rsuffix="_lookup",
        how="left",
    ).reset_index()

    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True)
    mask = regime_mask_for_samples(features, aligned, cfg=cfg, columns=columns, split_train_end=split_train_end)
    filtered = apply_regime_filter(aligned["pred_class"].to_numpy(dtype=int), mask.to_numpy(dtype=bool))
    return filtered, mask
