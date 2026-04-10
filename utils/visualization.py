# utils/visualization.py
"""
Visualizer: publication-quality training and evaluation plots.

Produces separate figures for clarity and reusability:
    training_history.png    — training dynamics (loss, Sharpe, risk, trading stats)
    {mode}_equity.png       — walk-forward equity report (4 panels)
    {mode}_positions.png    — position distribution analysis (3 panels)

All figures are saved to disk only — no interactive display.
Uses Agg backend for headless server compatibility.
"""

import logging
from pathlib import Path

import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker   as mticker
from matplotlib.lines import Line2D

from evaluation.walk_forward import WalkForwardResult

log = logging.getLogger(__name__)

# ── Design system ─────────────────────────────────────────────────────────────
_BG   = "#0f1117"
_PANEL= "#1a1d2e"
_GRID = "#2a2d3e"
_TEXT = "#e8e9f0"
_SUB  = "#8b8fa8"

_C = {
    "green":  "#00d4aa",
    "red":    "#ff4d6d",
    "blue":   "#4d9fff",
    "purple": "#b388ff",
    "orange": "#ffb74d",
    "yellow": "#ffd54f",
    "teal":   "#40e0d0",
    "gray":   "#4a4d60",
    "white":  "#e8e9f0",
}


def _apply_style(ax, ylabel: str = "", xlabel: str = "Epoch") -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_SUB, labelsize=8)
    ax.set_xlabel(xlabel, color=_SUB, fontsize=8)
    ax.set_ylabel(ylabel, color=_TEXT, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.7)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    log.info("Saved → %s", path)


class Visualizer:

    def __init__(self, out_dir: str):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ── Training history ──────────────────────────────────────────────────────

    def plot_training_history(self, history: list[dict]) -> None:
        """
        Save one PNG per metric — 6 separate files in the plots directory.
        Files: train_loss.png, val_sharpe.png, val_sortino.png,
               val_max_drawdown.png, val_turnover.png, val_flat_frac.png

        Separate files allow embedding individual charts in README / papers.
        Note: straight lines are expected with few validation points (e.g.
        2 points with val_freq=5 and num_epochs=10). More epochs → richer curves.
        """
        if not history:
            log.warning("No training history to plot.")
            return

        epochs = [r["epoch"] for r in history]
        n = len(epochs)

        def _get(key: str) -> np.ndarray:
            return np.array([r.get(key, 0.0) for r in history], dtype=float)

        panels = [
            # (filename, ylabel, title, color, data, hline, invert, pct_fmt)
            ("train_loss",       "Loss",           "Train Loss",
             _C["red"],    _get("train_loss"),       None,  False, False),
            ("val_sharpe",       "Sharpe (ann.)",  "Validation Sharpe  (annualised)",
             _C["green"],  _get("val_sharpe"),        0.0,   False, False),
            ("val_sortino",      "Sortino (ann.)", "Validation Sortino  (annualised)",
             _C["blue"],   _get("val_sortino"),       0.0,   False, False),
            ("val_max_drawdown", "Max DD (%)",     "Validation Max Drawdown",
             _C["red"],    _get("val_max_drawdown"),  None,  True,  True),
            ("val_turnover",     "|Δpos| / bar",   "Validation Turnover per Bar",
             _C["purple"], _get("val_turnover"),      None,  False, False),
            ("val_flat_frac",    "Flat fraction",  "Validation Flat Fraction  (time in cash)",
             _C["orange"], _get("val_flat_frac"),     None,  False, True),
        ]

        for fname, ylabel, title, color, data, hline, invert, pct in panels:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG)
            _apply_style(ax, ylabel=ylabel)
            ax.set_title(title, color=_TEXT, fontsize=12, pad=10)
            fig.suptitle("DiffQuant  ·  Training Dynamics",
                         fontsize=10, color=_SUB, y=0.98)

            # Raw values + markers
            ax.plot(epochs, data, color=color, linewidth=1.5,
                    alpha=0.5 if n >= 5 else 1.0)
            ax.scatter(epochs, data, color=color, s=50, zorder=5,
                       edgecolors=_BG, linewidths=1.0)

            # SMA only when enough points
            if n >= 4:
                w   = max(3, n // 4)
                pad = np.pad(data, (w//2, w - w//2), mode="edge")
                sma = np.convolve(pad, np.ones(w)/w, mode="valid")[:n]
                ax.plot(epochs, sma, color=color, linewidth=2.5,
                        label=f"SMA-{w}")
                ax.legend(fontsize=8, facecolor=_PANEL, labelcolor=_SUB,
                          framealpha=0.7, loc="best")

            # Fill area
            baseline = data.min() if hline is None else hline
            ax.fill_between(epochs, data, baseline,
                            where=data >= baseline,
                            color=color, alpha=0.15)
            if hline is not None:
                ax.fill_between(epochs, data, hline,
                                where=data < hline,
                                color=_C["red"], alpha=0.15)
                ax.axhline(hline, color=_C["gray"],
                           linewidth=1.0, linestyle="--", alpha=0.7)

            # Best point annotation for Sharpe
            if fname == "val_sharpe" and len(data) > 0:
                best_i = int(np.argmax(data))
                spread = data.ptp() if data.ptp() > 0 else abs(data[best_i]) * 0.1 + 0.1
                ax.annotate(
                    f"best={data[best_i]:+.3f}  (ep {epochs[best_i]})",
                    xy=(epochs[best_i], data[best_i]),
                    xytext=(epochs[best_i], data[best_i] + spread * 0.12 + 0.01),
                    fontsize=9, color=_C["yellow"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=_C["yellow"], lw=1.0),
                    ha="center",
                )
                ax.axvline(epochs[best_i], color=_C["yellow"],
                           linewidth=1.0, linestyle=":", alpha=0.7)

            # Annotate each data point with its value
            for ep, val in zip(epochs, data):
                if fname == "val_flat_frac":
                    fmt = f"{val:.1%}"
                elif fname == "val_max_drawdown":
                    fmt = f"{val:.2f}%"
                else:
                    fmt = f"{val:.4f}"
                ax.annotate(fmt, xy=(ep, val),
                            xytext=(0, 8), textcoords="offset points",
                            fontsize=7, color=_SUB, ha="center")

            if pct:
                ax.yaxis.set_major_formatter(
                    mticker.PercentFormatter(xmax=1.0)
                    if fname == "val_flat_frac"
                    else mticker.FormatStrFormatter("%.1f%%")
                )
            if invert:
                ax.invert_yaxis()
            if hline is None and not invert:
                ax.set_ylim(bottom=0)

            # Sparse data note
            if n < 5:
                ax.text(
                    0.99, 0.04,
                    f"Only {n} val points — increase num_epochs or reduce val_freq for smoother curves",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=_SUB, style="italic",
                )

            ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
            plt.tight_layout()
            _save(fig, self.out / f"{fname}.png")

    # ── Equity report ─────────────────────────────────────────────────────────

    def plot_equity_report(
        self,
        result:   WalkForwardResult,
        mode:     str,
        tf_min:   int   = 5,
        exit_thr: float = 0.05,
    ) -> None:
        ts     = pd.to_datetime(result.timestamps, unit="ms", utc=True)
        equity = result.equity
        pnl    = result.step_pnl
        pos    = result.positions
        gates  = result.gates

        ret_pct   = (equity - 1.0) * 100
        daily_pnl = pd.Series(pnl * 100, index=ts).resample("1D").sum()
        peak      = np.maximum.accumulate(equity)
        dd        = (peak - equity) / (peak + 1e-8) * 100

        bars_per_day = int(24 * 60 / tf_min)
        win      = 30 * bars_per_day
        pnl_s    = pd.Series(pnl, index=ts)
        roll_sh  = (pnl_s.rolling(win).mean() /
                    (pnl_s.rolling(win).std() + 1e-8) *
                    np.sqrt(252 * bars_per_day))

        m = result.metrics

        fig = plt.figure(figsize=(18, 14), facecolor=_BG)
        fig.suptitle(
            f"DiffQuant  ·  {mode.upper()} Walk-Forward Evaluation",
            fontsize=15, fontweight="bold", color=_TEXT, y=0.98,
        )

        gs = gridspec.GridSpec(
            4, 1, figure=fig,
            height_ratios=[3, 1.5, 1.5, 1.5],
            hspace=0.40, left=0.07, right=0.95, top=0.93, bottom=0.06,
        )

        # Panel 1 — Equity curve
        ax1 = fig.add_subplot(gs[0])
        _apply_style(ax1, ylabel="Cumulative Return (%)", xlabel="")
        ax1.set_title(
            f"Sharpe={m['sharpe']:+.2f}  ·  Sortino={m['sortino']:+.2f}  ·  "
            f"MaxDD={m['max_drawdown_pct']:.2f}%  ·  "
            f"Return={m['final_return_pct']:+.2f}%",
            color=_SUB, fontsize=9, pad=6,
        )
        ax1.plot(ts, ret_pct, color=_C["white"], linewidth=1.5, zorder=3)
        ax1.fill_between(ts, ret_pct, 0,
                         where=ret_pct >= 0, color=_C["green"], alpha=0.20, zorder=2)
        ax1.fill_between(ts, ret_pct, 0,
                         where=ret_pct <  0, color=_C["red"],   alpha=0.20, zorder=2)
        ax1.axhline(0, color=_C["gray"], linewidth=0.8, linestyle="--")
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))

        ax1b = ax1.twinx()
        ax1b.set_facecolor(_PANEL)
        bar_col = [_C["green"] if v >= 0 else _C["red"]
                   for v in daily_pnl.values]
        ax1b.bar(daily_pnl.index, daily_pnl.values,
                 color=bar_col, alpha=0.25, width=0.7, zorder=1)
        ax1b.set_ylabel("Daily PnL (%)", color=_SUB, fontsize=8)
        ax1b.tick_params(colors=_SUB, labelsize=7)
        ax1b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f%%"))
        for sp in ax1b.spines.values():
            sp.set_edgecolor(_GRID)
        ax1.set_zorder(ax1b.get_zorder() + 1)
        ax1.patch.set_visible(False)

        # Panel 2 — Drawdown
        ax2 = fig.add_subplot(gs[1])
        _apply_style(ax2, ylabel="Drawdown (%)", xlabel="")
        ax2.set_title("Underwater Equity", color=_SUB, fontsize=9, pad=4)
        ax2.fill_between(ts, -dd, 0, color=_C["red"], alpha=0.45)
        ax2.plot(ts, -dd, color=_C["red"], linewidth=0.8)
        ax2.axhline(0, color=_C["gray"], linewidth=0.6, linestyle="--")
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

        # Panel 3 — Position
        ax3 = fig.add_subplot(gs[2])
        _apply_style(ax3, ylabel="Position weight", xlabel="")
        ax3.set_title(
            f"Long={m['long_fraction']*100:.1f}%  ·  "
            f"Short={m['short_fraction']*100:.1f}%  ·  "
            f"Flat={m['flat_fraction']*100:.1f}%  ·  "
            f"Rebalances={len(result.rebalance_records)}",
            color=_SUB, fontsize=9, pad=4,
        )
        pos_s = pd.Series(pos, index=ts)
        ax3.plot(ts, pos, color=_C["gray"], linewidth=0.3, alpha=0.5)
        ax3.fill_between(ts, pos, 0, where=(pos_s > exit_thr).values,
                         color=_C["green"], alpha=0.55, label="Long")
        ax3.fill_between(ts, pos, 0, where=(pos_s < -exit_thr).values,
                         color=_C["red"],   alpha=0.55, label="Short")
        ax3.axhline(0,         color=_C["gray"],  linewidth=0.8, linestyle="--")
        ax3.axhline( exit_thr, color=_C["green"], linewidth=0.5, linestyle=":", alpha=0.5)
        ax3.axhline(-exit_thr, color=_C["red"],   linewidth=0.5, linestyle=":", alpha=0.5)
        ax3.set_ylim(-1.1, 1.1)
        ax3.legend(fontsize=7, facecolor=_PANEL, labelcolor=_SUB,
                   framealpha=0.7, loc="upper right")

        # Panel 4 — Gate + Rolling Sharpe
        ax4 = fig.add_subplot(gs[3])
        _apply_style(ax4, ylabel="Gate activation")
        ax4.set_title("Model Confidence (Gate) & 30-Day Rolling Sharpe",
                      color=_SUB, fontsize=9, pad=4)
        gate_d = pd.Series(gates, index=ts).resample("1D").mean()
        ax4.fill_between(gate_d.index, gate_d.values, 0,
                         color=_C["purple"], alpha=0.20)
        ax4.plot(gate_d.index, gate_d.values,
                 color=_C["purple"], linewidth=1.2, label="Gate (daily mean)")
        ax4.set_ylim(0.0, 1.0)
        ax4.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        ax4b = ax4.twinx()
        ax4b.set_facecolor(_PANEL)
        for sp in ax4b.spines.values():
            sp.set_edgecolor(_GRID)
        ax4b.plot(roll_sh.index, roll_sh.values,
                  color=_C["teal"], linewidth=1.2, alpha=0.85,
                  linestyle="--", label="30D Sharpe")
        ax4b.axhline(0, color=_C["gray"], linewidth=0.6, linestyle=":", alpha=0.5)
        ax4b.set_ylabel("Sharpe (ann.)", color=_SUB, fontsize=8)
        ax4b.tick_params(colors=_SUB, labelsize=7)
        for sp in ax4b.spines.values():
            sp.set_edgecolor(_GRID)

        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=7, facecolor=_PANEL, labelcolor=_SUB,
                   framealpha=0.7, loc="upper left")

        _save(fig, self.out / f"{mode}_equity.png")

    # ── Position distribution ─────────────────────────────────────────────────

    def plot_position_distribution(
        self,
        result:   WalkForwardResult,
        mode:     str,
        exit_thr: float = 0.05,
    ) -> None:
        pos = result.positions
        m   = result.metrics

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=_BG)
        fig.suptitle(
            f"DiffQuant  ·  {mode.upper()} Position Analysis",
            fontsize=13, fontweight="bold", color=_TEXT, y=1.02,
        )

        # Left: histogram
        ax = axes[0]
        _apply_style(ax, ylabel="Bar count", xlabel="Position weight")
        ax.set_title("Position Weight Distribution", color=_TEXT,
                     fontsize=10, pad=6)
        long_pos  = pos[pos >  exit_thr]
        short_pos = pos[pos < -exit_thr]
        flat_pos  = pos[np.abs(pos) <= exit_thr]
        if len(long_pos):
            ax.hist(long_pos,  bins=40, color=_C["green"], alpha=0.75,
                    edgecolor="none", label="Long")
        if len(short_pos):
            ax.hist(short_pos, bins=40, color=_C["red"],   alpha=0.75,
                    edgecolor="none", label="Short")
        if len(flat_pos):
            ax.hist(flat_pos,  bins=20, color=_C["gray"],  alpha=0.60,
                    edgecolor="none", label="Flat")
        ax.axvline(0,          color=_C["white"], linewidth=1.0,
                   linestyle="--", alpha=0.6)
        ax.axvline( exit_thr,  color=_C["green"], linewidth=0.8,
                    linestyle=":", alpha=0.7)
        ax.axvline(-exit_thr,  color=_C["red"],   linewidth=0.8,
                    linestyle=":", alpha=0.7)
        ax.legend(fontsize=8, facecolor=_PANEL, labelcolor=_SUB, framealpha=0.7)
        ax.set_title(
            f"Mean gate={result.gates.mean():.3f}  ·  "
            f"Mean |pos|={m['mean_abs_position']:.3f}",
            color=_SUB, fontsize=8, pad=4,
        )

        # Middle: CDF of |position|
        ax = axes[1]
        _apply_style(ax, ylabel="Cumulative fraction", xlabel="|Position| weight")
        ax.set_title("CDF of |Position|", color=_TEXT, fontsize=10, pad=6)
        abs_pos = np.sort(np.abs(pos))
        cdf     = np.arange(1, len(abs_pos) + 1) / len(abs_pos)
        ax.plot(abs_pos, cdf, color=_C["blue"], linewidth=2.0)
        ax.fill_between(abs_pos, cdf, 0, color=_C["blue"], alpha=0.12)
        ax.axvline(exit_thr, color=_C["orange"], linewidth=1.0,
                   linestyle="--", alpha=0.8,
                   label=f"exit_thr={exit_thr:.2f}")
        ax.axhline(m["flat_fraction"], color=_C["gray"], linewidth=0.8,
                   linestyle=":", alpha=0.7,
                   label=f"flat={m['flat_fraction']:.1%}")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8, facecolor=_PANEL, labelcolor=_SUB, framealpha=0.7)

        # Right: pie
        ax = axes[2]
        ax.set_facecolor(_PANEL)
        ax.set_title("Time Allocation", color=_TEXT, fontsize=10, pad=6)
        fracs  = [m["long_fraction"], m["short_fraction"], m["flat_fraction"]]
        labels = ["Long", "Short", "Flat"]
        colors = [_C["green"], _C["red"], _C["gray"]]
        nz = [(f, l, c) for f, l, c in zip(fracs, labels, colors) if f > 0]
        if nz:
            fracs_, labels_, colors_ = zip(*nz)
            _, _, autotexts = ax.pie(
                fracs_, labels=labels_, colors=colors_,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"linewidth": 0.8, "edgecolor": _BG},
                textprops={"color": _TEXT, "fontsize": 9},
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_color(_BG)
        ax.set_title(
            f"Rebalances={len(result.rebalance_records)}  ·  "
            f"Commission={m['total_commission_pct']:.3f}%",
            color=_SUB, fontsize=8, pad=4,
        )

        plt.tight_layout()
        _save(fig, self.out / f"{mode}_positions.png")

