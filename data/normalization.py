# data/normalization.py
"""
Per-sample normalisation and cyclic time-feature encoding.

All statistics are computed exclusively from the context window —
no global stats, no look-ahead into the horizon.
"""

from typing import Tuple
import numpy as np
import torch


def normalize_context(
    context: torch.Tensor,        # (B, T, F) or (T, F)
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Z-score each feature channel using context-window statistics only.

    Returns normalised context, mean (B,1,F), std (B,1,F).
    The returned stats must be used to normalise the horizon window
    in the trainer — ensuring context and horizon share the same scale.
    """
    squeezed = context.dim() == 2
    if squeezed:
        context = context.unsqueeze(0)

    mean = context.mean(dim=1, keepdim=True)
    std  = context.std(dim=1, keepdim=True).clamp(min=eps)
    out  = (context - mean) / std

    if squeezed:
        return out.squeeze(0), mean.squeeze(0), std.squeeze(0)
    return out, mean, std


def apply_stats(
    x:    torch.Tensor,   # (..., F)
    mean: torch.Tensor,
    std:  torch.Tensor,
) -> torch.Tensor:
    """Apply precomputed (mean, std) to any tensor with matching last dim."""
    return (x - mean) / std


def build_time_features(ts_ms: np.ndarray) -> np.ndarray:
    """
    Cyclic encoding of UTC hour-of-day and day-of-week → (N, 4).

    Channels: [sin_hour, cos_hour, sin_dow, cos_dow]
    Encoding ensures temporal proximity is preserved at period boundaries
    (e.g. 23:55 and 00:05 are geometrically close).
    """
    import pandas as pd

    dt   = pd.to_datetime(ts_ms, unit="ms", utc=True)
    hour = (dt.hour + dt.minute / 60.0).values.astype(np.float32)
    dow  = dt.dayofweek.values.astype(np.float32)

    return np.stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow  / 7),
        np.cos(2 * np.pi * dow  / 7),
    ], axis=1).astype(np.float32)

