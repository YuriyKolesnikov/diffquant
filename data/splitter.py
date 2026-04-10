# data/splitter.py
"""
Strict temporal split of sample arrays into train / val / test / backtest.
"""

import logging
import numpy as np
import pandas as pd
from configs.base_config import MasterConfig

log = logging.getLogger(__name__)


def split(
    full_sequences: np.ndarray,   # (S, ctx+hor, F)  ← было context_features
    horizon_closes: np.ndarray,   # (S, hor+1)
    timestamps:     np.ndarray,   # (S,) Unix ms
    features_cont:  np.ndarray,   # (N, F)  full continuous feature array
    closes_cont:    np.ndarray,   # (N,)    full continuous close prices
    ts_cont:        np.ndarray,   # (N,)    full continuous timestamps
    cfg:            MasterConfig,
) -> dict:
    s = cfg.data.splits
    boundaries = {
        "train":    (s.train_start, s.train_end),
        "val":      (s.train_end,   s.val_end),
        "test":     (s.val_end,     s.test_end),
        "backtest": (s.test_end,    s.backtest_end),
    }

    result = {}
    for name, (start, end) in boundaries.items():
        mask     = _make_mask(timestamps, start, end)
        mask_raw = _make_mask(ts_cont,   start, end)

        entry = {
            "full_sequences": full_sequences[mask],
            "horizon_closes": horizon_closes[mask],
            "timestamps":     timestamps[mask],
            "context_len":    cfg.data.context_len,
            "horizon_len":    cfg.data.horizon_len,
            "n_features":     cfg.data.resolve_n_features(),
        }

        # Raw continuous arrays for walk-forward evaluation (not needed for train).
        if name != "train":
            entry["raw_features"]   = features_cont[mask_raw]
            entry["raw_closes"]     = closes_cont[mask_raw]
            entry["raw_timestamps"] = ts_cont[mask_raw]

        result[name] = entry
        log.info(
            "  %-10s  %s samples  [%s → %s]",
            name, f"{int(mask.sum()):,}",
            start or "start", end or "end",
        )

    return result


def _make_mask(
    timestamps: np.ndarray,
    start:      str | None,
    end:        str | None,
) -> np.ndarray:
    mask = np.ones(len(timestamps), dtype=bool)
    if start:
        mask &= timestamps >= _to_ms(start)
    if end:
        mask &= timestamps <  _to_ms(end)
    return mask


def _to_ms(date_str: str) -> int:
    """'YYYY-MM-DD' → Unix milliseconds (UTC start of day)."""
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)

