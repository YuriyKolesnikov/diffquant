# utils/metrics.py
"""
Financial metrics for DiffQuant evaluation reports.

Called by WalkForwardEvaluator after every walk-forward run.
All inputs are numpy arrays — no torch dependency.

Metrics are annualised using the bar frequency of the dataset.
For 5-min bars: bars_per_year = 252 * 288 = 72,576.
"""

import numpy as np
import pandas as pd


def compute_metrics(
    step_pnl:       np.ndarray,
    positions:      np.ndarray,
    prev_positions: np.ndarray,
    comms:          np.ndarray,
    equity:         np.ndarray,
    ts_ms:          np.ndarray,
    tf_min:         int,
    flat_threshold: float = 0.05,
) -> dict:
    bars_per_day  = int(24 * 60 / tf_min)
    ann_factor    = np.sqrt(252 * bars_per_day)

    return {
        **_return_metrics(step_pnl, equity, ts_ms),
        **_risk_metrics(step_pnl, equity, ann_factor),
        **_time_metrics(step_pnl, positions, ts_ms, flat_threshold),
        **_position_metrics(positions, flat_threshold),
        **_direction_metrics(step_pnl, prev_positions, comms, flat_threshold),
        **_cost_metrics(comms),
    }


# ── Return ────────────────────────────────────────────────────────────────────

def _return_metrics(
    step_pnl: np.ndarray,
    equity:   np.ndarray,
    ts_ms:    np.ndarray,
) -> dict:
    dt_index = pd.to_datetime(ts_ms, unit="ms", utc=True)
    daily    = pd.Series(step_pnl, index=dt_index).resample("1D").sum()

    return {
        "final_return_pct":   float((equity[-1] - 1.0) * 100),
        "exp_day_return_pct": float(daily.mean() * 100),
        "max_bar_loss_pct":   float(step_pnl.min() * 100),
        "max_bar_profit_pct": float(step_pnl.max() * 100),
    }


# ── Risk ──────────────────────────────────────────────────────────────────────

def _risk_metrics(
    step_pnl:   np.ndarray,
    equity:     np.ndarray,
    ann_factor: float,
) -> dict:
    mu    = step_pnl.mean()
    sigma = step_pnl.std() + 1e-8
    neg   = step_pnl[step_pnl < 0]

    sharpe  = float(mu / sigma * ann_factor)
    sortino = float(mu / (neg.std() + 1e-8) * ann_factor) if len(neg) > 1 else 0.0

    peak    = np.maximum.accumulate(equity)
    max_dd  = float(((peak - equity) / (peak + 1e-8)).max() * 100)
    calmar  = float((equity[-1] - 1.0) / (max_dd / 100 + 1e-8))

    return {
        "sharpe":           sharpe,
        "sortino":          sortino,
        "calmar":           calmar,
        "max_drawdown_pct": max_dd,
    }


# ── Time ──────────────────────────────────────────────────────────────────────

def _time_metrics(
    step_pnl:  np.ndarray,
    positions: np.ndarray,
    ts_ms:     np.ndarray,
    flat_threshold: float,
) -> dict:
    dt_index = pd.to_datetime(ts_ms, unit="ms", utc=True)
    daily    = pd.Series(step_pnl, index=dt_index).resample("1D").sum()

    profit_days = int((daily > 0).sum())
    total_days  = int(len(daily))
    active_days = int(
        pd.Series(np.abs(positions) > flat_threshold, index=dt_index)
        .resample("1D").max().sum()
    )

    return {
        "total_days":      total_days,
        "profit_days":     profit_days,
        "profit_day_pct":  float(profit_days / max(total_days, 1) * 100),
        "active_days":     active_days,
        "active_day_pct":  float(active_days / max(total_days, 1) * 100),
    }


# ── Position ──────────────────────────────────────────────────────────────────

def _position_metrics(positions: np.ndarray, threshold: float) -> dict:
    deltas = np.abs(np.diff(positions, prepend=0.0))
    return {
        "flat_fraction":     float((np.abs(positions) < threshold).mean()),
        "long_fraction":     float((positions >  threshold).mean()),
        "short_fraction":    float((positions < -threshold).mean()),
        "mean_abs_position": float(np.abs(positions).mean()),
        "turnover":          float(deltas.mean()),
    }


# ── Direction accuracy ────────────────────────────────────────────────────────

def _direction_metrics(
    step_pnl:  np.ndarray,
    prev_positions: np.ndarray,
    comms:     np.ndarray,
    threshold: float,
) -> dict:
    gross     = step_pnl + comms
    long_m    = prev_positions >  threshold
    short_m   = prev_positions < -threshold
    l_correct = (long_m  & (gross > 0)).sum()
    s_correct = (short_m & (gross < 0)).sum()
    l_total   = long_m.sum()
    s_total   = short_m.sum()
    active    = l_total + s_total

    correct_m   = (long_m & (gross > 0)) | (short_m & (gross < 0))
    incorrect_m = (long_m | short_m) & ~correct_m

    return {
        "long_total":             int(l_total),
        "long_correct":           int(l_correct),
        "long_accuracy_pct":      float(l_correct / max(l_total, 1) * 100),
        "short_total":            int(s_total),
        "short_correct":          int(s_correct),
        "short_accuracy_pct":     float(s_correct / max(s_total, 1) * 100),
        "direction_accuracy_pct": float((l_correct + s_correct) / max(active, 1) * 100),
        "correct_avg_ret_pct":    float(gross[correct_m].mean()   * 100) if correct_m.any()   else 0.0,
        "incorrect_avg_ret_pct":  float(gross[incorrect_m].mean() * 100) if incorrect_m.any() else 0.0,
    }


# ── Costs ─────────────────────────────────────────────────────────────────────

def _cost_metrics(comms: np.ndarray) -> dict:
    return {
        "total_commission_pct": float(comms.sum() * 100),
        "avg_commission_pct":   float(comms[comms > 0].mean() * 100) if (comms > 0).any() else 0.0,
    }


# ── Daily breakdown helper (used by logging_utils) ────────────────────────────

def daily_breakdown(
    step_pnl:             np.ndarray,
    positions:            np.ndarray,
    comms:                np.ndarray,
    closes:               np.ndarray,
    gates:                np.ndarray,
    ts_ms:                np.ndarray,
    min_delta_rebalance:  float = 0.05,   # cfg.eval.min_delta_to_rebalance
) -> pd.DataFrame:
    """Returns a daily-aggregated DataFrame for the report log."""
    idx = pd.to_datetime(ts_ms, unit="ms", utc=True)

    df = pd.DataFrame({
        "pnl":      step_pnl,
        "position": positions,
        "comm":     comms,
        "close":    closes,
        "gate":     gates,
    }, index=idx)

    agg = pd.DataFrame({
        "open":    df["close"].resample("1D").first(),
        "close":   df["close"].resample("1D").last(),
        "pnl_pct": df["pnl"].resample("1D").sum() * 100,
        "comm_pct":df["comm"].resample("1D").sum() * 100,
        "avg_pos": df["position"].abs().resample("1D").mean(),
        "gate":    df["gate"].resample("1D").mean(),
        "rebal":   df["position"].resample("1D").apply(
            lambda x: int((np.abs(np.diff(x.values)) >= min_delta_rebalance).sum())
            if len(x) > 1 else 0
        ),
    }).dropna(subset=["open"])

    return agg

