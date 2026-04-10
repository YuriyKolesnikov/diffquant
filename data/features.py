# data/features.py
"""
Feature engineering: raw OHLCV bars → stationary model inputs.

All transformations are strictly causal — no information from bar t+1
is used when computing features for bar t.

Transformation rules per channel type:

    Price channels (open, high, low, close):
        log(price_t / close_{t-1})
        Log-return relative to the previous bar's close.
        Stationary by construction; invariant to price level.

    Volume / num_trades:
        log(v_t / rolling_mean(v, window=48) + 1)
        Normalised relative to a causal 48-bar expanding/rolling mean.
        Uses expanding mean for the first 48 bars, then rolling mean.
        Avoids global-mean look-ahead present in naive v / mean(v).

    Typical price (optional):
        log((H+L+C)/3 / close_{t-1})
        Bar midpoint log-return. Enabled by cfg.data.add_typical_price.

    Rolling volatility (optional):
        log(rolling_std(close_log_ret, window=20) + eps)
        Causal estimate of return volatility. Provides the model with
        regime-scale information that survives per-sample z-score
        normalisation. Enabled by cfg.data.add_rolling_vol.

    Time features (optional):
        [sin_hour, cos_hour, sin_dow, cos_dow]
        Cyclic UTC encoding of hour-of-day and day-of-week.
        Enabled by cfg.data.add_time_features.

Output:
    (features, timestamps, closes) — all length N-1, first bar dropped
    because log-return at t=0 requires close_{t-1} which is unavailable.
"""

import numpy as np

from configs.base_config import MasterConfig, FEATURE_PRESETS
from data.normalization  import build_time_features

# Index map for the source .npz column layout.
_I   = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4, "num_trades": 5}
_EPS = 1e-10


def build_feature_matrix(
    bars: np.ndarray,   # (N, 6)  float32 — aggregated OHLCV from aggregator.py
    ts:   np.ndarray,   # (N,)    int64   — Unix ms timestamps (close-time)
    cfg:  MasterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the model input matrix from aggregated bars.

    Returns
    -------
    features   : (N-1, F)  float32
    timestamps : (N-1,)    int64
    closes     : (N-1,)    float32  — raw close prices (used by simulator)
    """
    columns = _resolve_columns(cfg)
    close   = bars[:, _I["close"]]                           # (N,)
    prev_c  = np.concatenate([[np.nan], close[:-1]])         # (N,) NaN at t=0

    parts: list[np.ndarray] = []

    # ── Price channels ────────────────────────────────────────────────────────
    for col in columns:
        if col in ("open", "high", "low", "close"):
            parts.append(
                np.log(bars[:, _I[col]] / (prev_c + _EPS) + _EPS)[:, None]
            )

        elif col in ("volume", "num_trades"):
            parts.append(
                _causal_volume_feature(v=bars[:, _I[col]], window=cfg.data.vol_window)[:, None]
            )

        else:
            raise ValueError(
                f"No feature engineering rule for column {col!r}. "
                f"Add it to features.py or choose a valid preset."
            )

    # ── Optional channels ─────────────────────────────────────────────────────
    if cfg.data.add_typical_price:
        tp = (bars[:, _I["high"]] + bars[:, _I["low"]] + bars[:, _I["close"]]) / 3.0
        parts.append(
            np.log(tp / (prev_c + _EPS) + _EPS)[:, None]
        )

    if getattr(cfg.data, "add_rolling_vol", False):
        close_ret = np.log(close / (prev_c + _EPS) + _EPS)
        close_ret[0] = 0.0   # replace NaN at t=0 with neutral value

        parts.append(
            _causal_rolling_vol(close_ret, window=cfg.data.vol_window_vol)[:, None]
        )

    if cfg.data.add_time_features:
        parts.append(build_time_features(ts))   # (N, 4)

    features = np.concatenate(parts, axis=1).astype(np.float32)

    # Drop first bar — prev_close is undefined there (log-return undefined).
    return features[1:], ts[1:], close[1:].astype(np.float32)


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve_columns(cfg: MasterConfig) -> list[str]:
    """Return the ordered list of base feature columns for this config."""
    if cfg.data.preset == "custom":
        if not cfg.data.feature_columns:
            raise ValueError("preset='custom' requires feature_columns to be non-empty.")
        return list(cfg.data.feature_columns)
    if cfg.data.preset not in FEATURE_PRESETS:
        raise ValueError(
            f"Unknown preset: {cfg.data.preset!r}. "
            f"Valid: {sorted(FEATURE_PRESETS)}"
        )
    return list(FEATURE_PRESETS[cfg.data.preset])


def _causal_volume_feature(v: np.ndarray, window: int = 48) -> np.ndarray:
    """
    Causal relative-volume feature: log(v_t / local_mean_t + 1).

    local_mean_t is computed as an expanding mean for the first `window` bars,
    then a rolling mean of the last `window` bars.  No future data is used.

    Parameters
    ----------
    v      : (N,) raw volume or num_trades array
    window : look-back window for the rolling mean (default 48 bars)

    Returns
    -------
    (N,) float32
    """
    v        = v.astype(np.float64)
    cumsum   = np.concatenate([[0.0], np.cumsum(v)])

    # Expanding mean for bars 0..window-1, rolling mean for bars window..N-1
    expanding = cumsum[1 : window + 1] / np.arange(1, window + 1)
    rolling   = (cumsum[window + 1:] - cumsum[1 : len(v) - window + 1]) / window

    local_mean = np.concatenate([expanding, rolling])   # (N,)
    return np.log(v / (local_mean + _EPS) + 1.0).astype(np.float32)


def _causal_rolling_vol(log_rets: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Causal rolling volatility: log(rolling_std(log_ret, window) + eps).

    Uses expanding std for the first `window` bars (no look-ahead).
    Provides the model with a regime-scale signal that survives
    per-sample z-score normalisation in the trainer.

    Parameters
    ----------
    log_rets : (N,) close log-returns, NaN at t=0 must be replaced before calling
    window   : look-back window (default 20 bars)

    Returns
    -------
    (N,) float32
    """
    N   = len(log_rets)
    out = np.empty(N, dtype=np.float32)

    for i in range(N):
        start = max(0, i - window + 1)
        chunk = log_rets[start : i + 1]
        out[i] = float(np.std(chunk)) if len(chunk) >= 2 else 1e-6

    return np.log(out + _EPS)

