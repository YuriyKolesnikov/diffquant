# data/pipeline.py
"""
Core data pipeline for DiffQuant.

Entry point: load_or_build()
    Orchestrates aggregation → feature engineering → sample slicing → split.
    Results cached on disk keyed by config hash.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from configs.base_config import MasterConfig
from data.aggregator import aggregate
from data.features   import build_feature_matrix
from data.splitter   import split

log = logging.getLogger(__name__)

_VALID_TIMEFRAMES = {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}


# ── Public API ────────────────────────────────────────────────────────────────

def load_or_build(
    source_path: str,
    cfg:         MasterConfig,
    cache_dir:   str = "data_cache/",
) -> dict:
    """
    Return processed splits from cache or build and cache them.

    Each split contains:
        full_sequences   : (N, ctx+hor, F) float32 — full window for sliding rollout
        horizon_closes   : (N, hor+1)  float32  — raw prices for simulator
        timestamps       : (N,)        int64    — Unix ms, context start
        context_len      : int
        horizon_len      : int
        n_features       : int
    """
    _validate_timeframe(cfg.data.timeframe_min)

    cache_key  = _config_hash(cfg, source_path)
    cache_path = Path(cache_dir) / f"{cache_key}.npz"

    if cache_path.exists():
        log.info("Cache hit [%s]", cache_key)
        return _load_cache(cache_path)

    log.info("Building dataset [%s]", cache_key)
    result = _build(source_path, cfg)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _save_cache(result, cache_path)
    log.info("Cached → %s", cache_path)
    return result


# ── Build ─────────────────────────────────────────────────────────────────────

def _build(source_path: str, cfg: MasterConfig) -> dict:
    raw  = np.load(source_path, allow_pickle=True)
    bars = raw["bars"].astype(np.float32)
    ts   = raw["timestamps"].astype(np.int64)

    log.info(
        "Source: %s  bars=%s  %s → %s",
        Path(source_path).name, f"{len(bars):,}",
        _ms_to_str(ts[0]), _ms_to_str(ts[-1]),
    )

    bars, ts = aggregate(bars, ts, cfg)
    log.info("Aggregated to %d-min: %s bars", cfg.data.timeframe_min, f"{len(bars):,}")

    features, ts, closes = build_feature_matrix(bars, ts, cfg)
    log.info("Features: shape=%s  channels=%s", features.shape, cfg.data.active_feature_set)

    full_seq, hor_arr, sample_ts = _make_samples(features, closes, ts, cfg)
    log.info("Samples: %s  stride=%d", f"{len(full_seq):,}", cfg.data.stride)

    # Pass full continuous arrays so splitter can build raw val/test/backtest series.
    return split(full_seq, hor_arr, sample_ts, features, closes, ts, cfg)


# ── Sample slicing ────────────────────────────────────────────────────────────

def _make_samples(
    features: np.ndarray,   # (N, F)
    closes:   np.ndarray,   # (N,)
    ts:       np.ndarray,   # (N,)  Unix ms
    cfg:      MasterConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx    = cfg.data.context_len
    hor    = cfg.data.horizon_len
    stride = cfg.data.stride
    win    = ctx + hor
    bar_ms = cfg.data.timeframe_min * 60_000

    full_out, hor_out, ts_out = [], [], []
    skipped_align = skipped_gap = 0

    i = 0
    while i + win <= len(ts):
        t0 = ts[i]

        if t0 % bar_ms != 0:
            skipped_align += 1
            i += 1
            continue

        window_ts = ts[i : i + win]
        if np.diff(window_ts).max() > bar_ms * 2:
            skipped_gap += 1
            i += stride
            continue

        hor_closes     = np.empty(hor + 1, dtype=np.float32)
        hor_closes[0]  = closes[i + ctx - 1]
        hor_closes[1:] = closes[i + ctx : i + ctx + hor]

        full_out.append(features[i : i + win])   # (ctx+hor, F) — full sequence
        hor_out.append(hor_closes)
        ts_out.append(t0)
        i += stride

    log.info(
        "Slicing: accepted=%s  skipped_gap=%s  skipped_align=%s",
        f"{len(ts_out):,}", f"{skipped_gap:,}", f"{skipped_align:,}",
    )

    return (
        np.array(full_out, dtype=np.float32),   # (S, ctx+hor, F)
        np.array(hor_out,  dtype=np.float32),   # (S, hor+1)
        np.array(ts_out,   dtype=np.int64),     # (S,)
    )


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def _save_cache(result: dict, path: Path) -> None:
    flat = {}
    for split_name, data in result.items():
        for field, val in data.items():
            arr = val if isinstance(val, np.ndarray) else np.array(val)
            flat[f"{split_name}__{field}"] = arr
    np.savez_compressed(path, **flat)


def _load_cache(path: Path) -> dict:
    raw    = np.load(path, allow_pickle=True)
    result: dict = {}
    for key in raw.files:
        split_name, field = key.split("__", 1)
        if split_name not in result:
            result[split_name] = {}
        arr = raw[key]
        result[split_name][field] = arr.item() if arr.ndim == 0 else arr
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_timeframe(tf: int) -> None:
    if tf not in _VALID_TIMEFRAMES:
        raise ValueError(
            f"timeframe_min={tf} must divide 60 evenly. "
            f"Valid: {sorted(_VALID_TIMEFRAMES)}"
        )


def _config_hash(cfg: MasterConfig, source_path: str) -> str:
    key = {
        "timeframe_min":     cfg.data.timeframe_min,
        "context_len":       cfg.data.context_len,
        "horizon_len":       cfg.data.horizon_len,
        "stride":            cfg.data.stride,
        "active_features":   list(cfg.data.active_feature_set),
        "add_time_features": cfg.data.add_time_features,
        "splits": {
            "train_start":  cfg.data.splits.train_start,
            "train_end":    cfg.data.splits.train_end,
            "val_end":      cfg.data.splits.val_end,
            "test_end":     cfg.data.splits.test_end,
            "backtest_end": cfg.data.splits.backtest_end,
        },
        "source_path": source_path,   # реальный путь, не cfg
    }
    return "diffquant_" + hashlib.md5(
        json.dumps(key, sort_keys=True).encode()
    ).hexdigest()[:12]


def _ms_to_str(ts_ms: int) -> str:
    import pandas as pd
    return pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")

