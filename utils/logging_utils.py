# utils/logging_utils.py
"""
MetricsLogger: unified logger for all evaluation modes.

Val (every val_freq epochs):
    One JSON-line appended to val_history.log per epoch.
    Enables post-training analysis of learning dynamics.

Test / Backtest (once, after training):
    Full report  → {mode}_report.log
    Daily log    → {mode}_daily.log
    Rebalances   → {mode}_rebalances.log
"""

import json
import logging
from datetime import datetime, timezone
from pathlib  import Path

import numpy as np
import pandas as pd

from evaluation.walk_forward import WalkForwardResult
from utils.metrics           import daily_breakdown

log = logging.getLogger(__name__)

_SEP  = "=" * 64
_SEP2 = "-" * 72


class MetricsLogger:

    def __init__(self, out_dir: str):
        self.out  = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self._val_log  = self.out / "val_history.log"
        self._val_json = self.out / "val_history.json"
        self._history: list[dict] = []

        # Training log handler — writes to file in addition to console.
        log_path    = self.out / "training.log"
        root_logger = logging.getLogger()
        already_added = any(
            isinstance(h, logging.FileHandler) and
            getattr(h, "baseFilename", None) == str(log_path.resolve())
            for h in root_logger.handlers
        )
        if not already_added:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            ))
            root_logger.addHandler(fh)

    # ── Val epoch logging ─────────────────────────────────────────────────────

    def log_val(
        self,
        epoch:        int,
        train_metrics: dict,
        val_metrics:   dict,
    ) -> None:
        """
        Log one validation epoch.
        Console: compact one-liner.
        File:    full JSON record for post-training analysis.
        """
        record = {
            "epoch":     epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **train_metrics,
            **val_metrics,
        }
        self._history.append(record)

        # Append JSONL
        with open(self._val_log, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Rewrite full JSON
        with open(self._val_json, "w") as f:
            json.dump(self._history, f, indent=2)

    # ── Full report (test / backtest) ─────────────────────────────────────────

    def write_report(
        self,
        result:              WalkForwardResult,
        mode:                str,
        min_delta_rebalance: float = 0.05,
    ) -> None:
        """Write summary, daily breakdown, and rebalance log."""
        self._write_summary(result, mode)
        self._write_daily(result, mode, min_delta_rebalance)
        self._write_rebalances(result, mode)

    def _write_summary(self, result: WalkForwardResult, mode: str) -> None:
        m      = result.metrics
        period = _period_str(result.timestamps)

        lines = [
            _SEP,
            f"  {mode.upper()} REPORT  |  {period}",
            _SEP,
            "",
            "  [Performance]",
            f"  :      total commission = {m['total_commission_pct']:>+8.2f}%",
            f"  :      avg commission   = {m['avg_commission_pct']:>+8.4f}%",
            f"  :      max bar loss     = {m['max_bar_loss_pct']:>+8.3f}%",
            f"  :      max bar profit   = {m['max_bar_profit_pct']:>+8.3f}%",
            f"  :      total return     = {m['final_return_pct']:>+8.2f}%",
            f"  :      exp day return   = {m['exp_day_return_pct']:>+8.3f}%",
            f"  :      max drawdown     = {m['max_drawdown_pct']:>+8.2f}%",
            "",
            "  [Risk Metrics]",
            f"  :      sharpe  (ann)    = {m['sharpe']:>8.3f}",
            f"  :      sortino (ann)    = {m['sortino']:>8.3f}",
            f"  :      calmar           = {m['calmar']:>8.3f}",
            "",
            "  [Time]",
            f"  :      total days       = {m['total_days']:>8d}",
            f"  :      profit days      = {m['profit_days']:>4d}  ({m['profit_day_pct']:.1f}%)",
            f"  :      active days      = {m['active_days']:>4d}  ({m['active_day_pct']:.1f}%)",
            "",
            "  [Position]",
            f"  :      flat fraction    = {m['flat_fraction']:>8.3f}",
            f"  :      long fraction    = {m['long_fraction']:>8.3f}",
            f"  :      short fraction   = {m['short_fraction']:>8.3f}",
            f"  :      mean |position|  = {m['mean_abs_position']:>8.3f}",
            f"  :      turnover / bar   = {m['turnover']:>8.4f}",
            f"  :      rebalances       = {len(result.rebalance_records):>8d}",
            "",
            "  [Directional Accuracy]",
            f"  :      long  correct    = {m['long_correct']:>4d}/{m['long_total']:>4d}"
            f"  ({m['long_accuracy_pct']:.1f}%)",
            f"  :      short correct    = {m['short_correct']:>4d}/{m['short_total']:>4d}"
            f"  ({m['short_accuracy_pct']:.1f}%)",
            f"  :      direction acc    = {m['direction_accuracy_pct']:>8.2f}%",
            f"  :      correct avg ret  = {m['correct_avg_ret_pct']:>+8.4f}%",
            f"  :      incorrect avg    = {m['incorrect_avg_ret_pct']:>+8.4f}%",
            "",
            _SEP,
        ]

        text = "\n".join(lines)
        log.info("\n%s", text)
        (self.out / f"{mode}_report.log").write_text(text + "\n")

    def _write_daily(self, result: WalkForwardResult, mode: str, min_delta_rebalance: float = 0.05,) -> None:
        df = daily_breakdown(
            result.step_pnl, result.positions, result.commissions,
            result.closes, result.gates, result.timestamps,
            min_delta_rebalance=min_delta_rebalance,
        )

        header = (
            f"  {'Date':<12} {'Open':>9} {'Close':>9} "
            f"{'PnL%':>8} {'Comm%':>7} {'AvgPos':>8} "
            f"{'Rebal':>6} {'Gate':>7}"
        )
        lines = [_SEP2, header, _SEP2]

        for date, row in df.iterrows():
            lines.append(
                f"  {date.strftime('%Y-%m-%d'):<12} "
                f"{row['open']:>9.1f} "
                f"{row['close']:>9.1f} "
                f"{row['pnl_pct']:>+7.3f}% "
                f"{row['comm_pct']:>+6.3f}% "
                f"{row['avg_pos']:>8.3f} "
                f"{int(row['rebal']):>6d} "
                f"{row['gate']:>7.3f}"
            )

        lines.append(_SEP2)
        text = "\n".join(lines)
        (self.out / f"{mode}_daily.log").write_text(text + "\n")
        log.info("Daily log → %s/%s_daily.log", self.out, mode)

    def _write_rebalances(self, result: WalkForwardResult, mode: str) -> None:
        if not result.rebalance_records:
            return

        header = (
            f"  {'Timestamp':<22} {'Close':>9} "
            f"{'PrevPos':>8} {'NewPos':>8} {'Delta':>8} "
            f"{'BarPnL%':>9} {'Comm%':>8}"
        )
        lines = [_SEP2, header, _SEP2]

        for r in result.rebalance_records:
            ts_str = pd.to_datetime(
                r["timestamp"], unit="ms", utc=True
            ).strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  {ts_str:<22} "
                f"{r['close']:>9.1f} "
                f"{r['prev_pos']:>+8.3f} "
                f"{r['new_pos']:>+8.3f} "
                f"{r['delta']:>+8.3f} "
                f"{r['bar_pnl']:>+8.4f}% "
                f"{r['commission']:>+7.4f}%"
            )

        lines += [_SEP2, f"  Total rebalance events: {len(result.rebalance_records)}"]
        text = "\n".join(lines)
        (self.out / f"{mode}_rebalances.log").write_text(text + "\n")
        log.info(
            "Rebalance log → %s/%s_rebalances.log  (%d events)",
            self.out, mode, len(result.rebalance_records),
        )
    
    def get_history(self) -> list[dict]:
        """Return training history for visualization."""
        return self._history


# ── Helpers ───────────────────────────────────────────────────────────────────

def _period_str(ts_ms: np.ndarray) -> str:
    fmt = "%Y-%m-%d"
    return (
        f"{pd.to_datetime(ts_ms[0],  unit='ms', utc=True).strftime(fmt)}"
        f" → "
        f"{pd.to_datetime(ts_ms[-1], unit='ms', utc=True).strftime(fmt)}"
    )

