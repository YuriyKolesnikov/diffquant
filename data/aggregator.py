# data/aggregator.py
"""
OHLCV aggregation from 1-min source data to any clock-aligned timeframe.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from configs.base_config import MasterConfig

# Column layout of the source .npz — must match collect_btc_1min.py.
_COLS = ["open", "high", "low", "close", "volume", "num_trades"]


def aggregate(
    bars_1m: np.ndarray,   # (N, 6) float32
    ts_1m:   np.ndarray,   # (N,)   int64  Unix ms UTC
    cfg:     MasterConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample 1-min bars to cfg.data.timeframe_min resolution.

    Returns (bars, timestamps) where timestamps are close-times
    pinned to clock boundaries (:00, :05, … for 5-min).
    Incomplete periods (< timeframe_min input bars) are dropped.
    """
    tf = cfg.data.timeframe_min
    if tf == 1:
        return bars_1m, ts_1m

    df = pd.DataFrame(
        bars_1m,
        columns=_COLS,
        index=pd.to_datetime(ts_1m, unit="ms", utc=True),
    )

    agg = df.resample(
        f"{tf}min", origin="epoch", closed="left", label="right",
    ).agg({
        "open":       "first",
        "high":       "max",
        "low":        "min",
        "close":      "last",
        "volume":     "sum",
        "num_trades": "sum",
    }).dropna(subset=["close"])

    # Keep only fully populated buckets (exactly tf source bars).
    # Partial buckets at data gaps or series boundaries produce misleading OHLC.
    counts = df["close"].resample(
        f"{tf}min", origin="epoch", closed="left", label="right",
    ).count()
    # Use .loc with aligned index to avoid reindex warning
    agg = agg.loc[counts[counts == tf].index]

    ts_out = (agg.index.view(np.int64) // 1_000_000).astype(np.int64)
    return agg.values.astype(np.float32), ts_out


def compute_typical_price(bars: np.ndarray) -> np.ndarray:
    """
    Typical price: (H + L + C) / 3. Not a true VWAP.
    Returns (N,) float32. Only meaningful for timeframe_min > 1.
    """
    h = bars[:, _COLS.index("high")]
    l = bars[:, _COLS.index("low")]
    c = bars[:, _COLS.index("close")]
    return ((h + l + c) / 3.0).astype(np.float32)


def get_close(bars: np.ndarray) -> np.ndarray:
    """Extract raw close price column from aggregated bars."""
    return bars[:, _COLS.index("close")]


def col_indices(names: list[str]) -> list[int]:
    """Map column names to their integer indices in _COLS."""
    return [_COLS.index(n) for n in names]

