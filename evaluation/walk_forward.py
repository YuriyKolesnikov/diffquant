# evaluation/walk_forward.py
"""
Continuous walk-forward evaluation engine.

Used by:
    - DiffTrainer._validate()  — lightweight, every val_freq epochs
    - backtest.py              — full run with metrics, logs, plots

Mechanics mirror live trading exactly:
    For each bar t in [ctx, N):
        1. window = features[t-ctx : t]          observe past ctx bars only
        2. normalize per-sample (same as training)
        3. model(window, extras) → raw_position
        4. hysteresis → target_position
        5. PnL = prev_pos * ret_t - commission * |Δpos|
        6. carry position to next bar

No episode resets. Position is tracked across the entire period.
"""

import logging
from dataclasses import dataclass, field
from typing      import Optional

import numpy as np
import torch

from configs.base_config      import MasterConfig
from data.normalization        import normalize_context
from model.policy_network     import PolicyNetwork
from utils.metrics import compute_metrics


log = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Single bar record — collected during walk-forward for reporting."""
    timestamp:  int
    close:      float
    raw_pos:    float
    position:   float
    prev_pos:   float
    ret:        float
    gross:      float
    commission: float
    net_pnl:    float
    equity:     float
    gate:       float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated result returned by WalkForwardEvaluator.run()."""
    # Per-step arrays (length = N - ctx)
    timestamps:  np.ndarray
    closes:      np.ndarray
    positions:   np.ndarray
    step_pnl:    np.ndarray
    commissions: np.ndarray
    equity:      np.ndarray
    gates:       np.ndarray
    prev_positions: np.ndarray   # position that earned each step's gross return

    # Summary metrics
    metrics: dict = field(default_factory=dict)

    # Rebalance events (subset of steps where |Δpos| >= min_delta)
    rebalance_records: list = field(default_factory=list)


class WalkForwardEvaluator:
    """
    Single evaluation engine for val, test, and backtest.

    Parameters
    ----------
    model : PolicyNetwork — must already be on the correct device.
    cfg   : MasterConfig
    """

    def __init__(self, model: PolicyNetwork, cfg: MasterConfig):
        self.model  = model
        self.cfg    = cfg
        self.device = next(model.parameters()).device

        ec = cfg.eval
        self._enter_long  = ec.enter_long_thr
        self._enter_short = ec.enter_short_thr
        self._exit        = ec.exit_thr
        self._min_delta   = ec.min_delta_to_rebalance
        self._commission  = (
            cfg.simulator.commission_rate + cfg.simulator.slippage_rate
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(
        self,
        raw: dict,
        mode: str = "val",
    ) -> WalkForwardResult:
        """
        Run walk-forward evaluation on a raw continuous split.

        Parameters
        ----------
        raw  : dict with keys raw_features (N,F), raw_closes (N,),
               raw_timestamps (N,) — from pipeline.load_or_build().
        mode : "val" | "test" | "backtest" — used for logging only.
        """
        self.model.eval()

        features  = raw["raw_features"]    # (N, F) numpy
        closes    = raw["raw_closes"]      # (N,)   numpy
        ts        = raw["raw_timestamps"]  # (N,)   numpy int64
        ctx       = self.cfg.data.context_len
        N         = len(features)

        if N <= ctx:
            raise ValueError(
                f"Series too short for walk-forward: N={N}, ctx={ctx}"
            )

        n_steps   = N - ctx
        positions  = np.zeros(n_steps, dtype=np.float32)
        step_pnl   = np.zeros(n_steps, dtype=np.float32)
        comms      = np.zeros(n_steps, dtype=np.float32)
        gates      = np.zeros(n_steps, dtype=np.float32)
        prev_positions = np.zeros(n_steps, dtype=np.float32)

        prev_pos   = 0.0
        prev_delta = 0.0
        rebalances = []

        for i, t in enumerate(range(ctx, N)):
            # Normalize per-sample on the context window.
            window_raw = torch.from_numpy(
                features[t - ctx : t]
            ).unsqueeze(0).to(self.device)                    # (1, ctx, F)
            window, _, _ = normalize_context(window_raw)

            extras = torch.tensor(
                [[prev_pos, prev_delta, 0.5, 0.5]],
                dtype=torch.float32, device=self.device,
            )

            out      = self.model.forward_with_components(window, extras)
            raw_pos  = out["position"].item()
            gate_val = out["gate"].item()

            target = self._hysteresis(raw_pos, prev_pos)

            # Step PnL
            ret   = _safe_return(closes[t], closes[t - 1])
            gross = prev_pos * ret

            earning_pos = prev_pos          # position that actually earned gross this bar

            delta = abs(target - prev_pos)
            comm  = 0.0
            executed_delta = 0.0
            if delta >= self._min_delta:
                comm = delta * self._commission
                rebalances.append({
                    "bar_idx":    i,
                    "timestamp":  int(ts[t]),
                    "close":      float(closes[t]),
                    "prev_pos":   prev_pos,
                    "new_pos":    target,
                    "delta":      target - prev_pos,
                    "commission": comm * 100,
                    "bar_pnl":    (gross - comm) * 100,
                })
                executed_delta = target - prev_pos
                prev_pos       = target

            prev_delta = executed_delta     # 0.0 if no rebalance — correct signal

            net = gross - comm

            positions[i]      = prev_pos
            prev_positions[i] = earning_pos   # position that earned gross_t
            step_pnl[i]       = net
            comms[i]     = comm
            gates[i]     = gate_val

        equity = np.cumprod(1.0 + step_pnl)

        result = WalkForwardResult(
            timestamps         = ts[ctx:],
            closes             = closes[ctx:],
            positions          = positions,
            step_pnl           = step_pnl,
            commissions        = comms,
            equity             = equity,
            gates              = gates,
            prev_positions     = prev_positions,
            rebalance_records  = rebalances,
        )

        result.metrics = compute_metrics(
            step_pnl       = step_pnl,
            positions      = positions,
            prev_positions = prev_positions,
            comms          = comms,
            equity         = equity,
            ts_ms          = result.timestamps,
            tf_min         = self.cfg.data.timeframe_min,
            flat_threshold = self.cfg.eval.exit_thr,
        )

        log.debug(
            "[%s]  sharpe=%+.3f  sortino=%+.3f  ret=%+.2f%%  "
            "dd=%.2f%%  flat=%.3f  rebalances=%d  total_comm=%.3f%%",
            mode,
            result.metrics["sharpe"],
            result.metrics["sortino"],
            result.metrics["final_return_pct"],
            result.metrics["max_drawdown_pct"],
            result.metrics["flat_fraction"],
            len(rebalances),
            result.metrics["total_commission_pct"],
        )

        return result

    # ── Execution logic ───────────────────────────────────────────────────────

    def _hysteresis(self, raw_pos: float, curr_pos: float) -> float:
        """
        Two-threshold hysteresis prevents rapid oscillation at boundaries.
        Entry threshold > exit threshold creates a dead-band that absorbs
        small signal fluctuations without triggering trades.
        """
        if   raw_pos >  self._enter_long:  return raw_pos
        elif raw_pos < -self._enter_short: return raw_pos
        elif abs(raw_pos) < self._exit:    return 0.0
        else:                               return curr_pos


def _safe_return(close: float, prev_close: float) -> float:
    return (close - prev_close) / (abs(prev_close) + 1e-10)

