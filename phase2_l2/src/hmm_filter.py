"""
HMM regime filter for pair-spread series. Ported from phase 1.

The inputs change: in phase 1 we used [spread_z_168h, spread_diff_1h,
spread_vol_24h] computed from hourly bars. In phase 2 we use bar-cadence
analogues over an appropriate lookback (default 1-minute z, 1-bar diff,
24-bar vol). Pass the feature names explicitly to `regime_mask_for_samples`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


HMM_FEATURE_COLUMNS_DEFAULT = ["spread_z", "spread_diff_1b", "spread_vol_short"]


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 2
    covariance_type: str = "full"
    n_iter: int = 50
    seed: int = 7
    refit_per_split: bool = True
    n_init: int = 1
    require_converged: bool = True


def _require_hmmlearn():
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as exc:
        raise ImportError("hmmlearn required: pip install hmmlearn") from exc
    return GaussianHMM


def _feature_matrix(features: pd.DataFrame, columns: List[str]) -> np.ndarray:
    arr = features[columns].to_numpy(dtype=float)
    return np.where(np.isfinite(arr), arr, 0.0)


def fit_hmm(train_features: pd.DataFrame, cfg: HMMConfig = HMMConfig(), columns: List[str] = HMM_FEATURE_COLUMNS_DEFAULT):
    GaussianHMM = _require_hmmlearn()
    X = _feature_matrix(train_features, columns)
    best_model = None; best_ll = -np.inf; best_converged = False
    for k in range(max(1, cfg.n_init)):
        model = GaussianHMM(
            n_components=cfg.n_states, covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter, random_state=cfg.seed + k,
        )
        try:
            model.fit(X)
            ll = float(model.score(X))
            converged = bool(getattr(model.monitor_, "converged", False))
        except Exception:
            continue
        accept = False
        if cfg.require_converged:
            if converged and not best_converged:
                accept = True
            elif converged == best_converged and ll > best_ll:
                accept = True
        else:
            if ll > best_ll:
                accept = True
        if accept:
            best_model = model; best_ll = ll; best_converged = converged
    if best_model is None:
        raise RuntimeError("All HMM fits failed")
    setattr(best_model, "_best_ll", best_ll)
    setattr(best_model, "_best_converged", best_converged)
    return best_model


def identify_mean_reverting_state(model, train_features: pd.DataFrame, columns: List[str] = HMM_FEATURE_COLUMNS_DEFAULT) -> int:
    X = _feature_matrix(train_features, columns)
    states = model.predict(X)
    diff_col = columns.index("spread_diff_1b") if "spread_diff_1b" in columns else 0
    z_col = columns.index("spread_z") if "spread_z" in columns else 0
    diff_vol = []; z_abs = []
    for s in range(model.n_components):
        mask = states == s
        if not mask.any():
            diff_vol.append(np.inf); z_abs.append(np.inf); continue
        diff_vol.append(float(np.std(X[mask, diff_col])))
        z_abs.append(float(np.mean(np.abs(X[mask, z_col]))))
    diff_vol_arr = np.asarray(diff_vol); z_abs_arr = np.asarray(z_abs)
    candidate = int(np.argmin(diff_vol_arr))
    ties = np.flatnonzero(diff_vol_arr == diff_vol_arr[candidate])
    if len(ties) > 1:
        candidate = int(ties[np.argmin(z_abs_arr[ties])])
    return candidate


def decode_states(model, features: pd.DataFrame, columns: List[str] = HMM_FEATURE_COLUMNS_DEFAULT) -> np.ndarray:
    return model.predict(_feature_matrix(features, columns)).astype(int)


def apply_regime_filter(pred_class: np.ndarray, mean_rev_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(pred_class, dtype=int).copy()
    out[~np.asarray(mean_rev_mask, dtype=bool)] = 1  # class 1 = flat/no-trade
    return out


def regime_mask_for_samples(
    features: pd.DataFrame, samples: pd.DataFrame,
    cfg: HMMConfig = HMMConfig(),
    columns: List[str] = HMM_FEATURE_COLUMNS_DEFAULT,
    split_train_end: Optional[pd.Timestamp] = None,
) -> pd.Series:
    if features.empty or samples.empty:
        return pd.Series(False, index=samples.index)
    features = features.copy(); samples = samples.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    samples["timestamp"] = pd.to_datetime(samples["timestamp"], utc=True)
    mask = pd.Series(False, index=samples.index)
    for pair, pair_features in features.groupby("pair", sort=False):
        pair_samples = samples[samples["pair"] == pair]
        if pair_samples.empty:
            continue
        pair_features = pair_features.sort_values("timestamp").reset_index(drop=True)
        train_slice = pair_features[pair_features["timestamp"] < split_train_end] if split_train_end is not None else pair_features
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
