# compare.py
"""
DiffQuant — experiment comparison entry point.

Scans the output/ directory for completed experiments and prints
a ranked comparison table sorted by best val Sharpe.

Also reads test_report.log and backtest_report.log where available,
so the table shows out-of-sample performance side-by-side.

Usage:
    python compare.py
    python compare.py --output-dir output/
    python compare.py --output-dir output/ --sort sharpe
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(message)s",
    handlers = [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_SEP  = "=" * 110
_SEP2 = "-" * 110


def _parse_report(path: Path, key: str) -> float:
    """Extract a single numeric value from a report log by field name."""
    if not path.exists():
        return float("nan")
    text    = path.read_text()
    pattern = rf"{re.escape(key)}\s*=\s*([+-]?\d+\.?\d*)"
    match   = re.search(pattern, text)
    return float(match.group(1)) if match else float("nan")


def _load_val_history(exp_dir: Path) -> dict:
    """Load val history and return best epoch metrics."""
    json_path = exp_dir / "val_history.json"
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        history = json.load(f)
    if not history:
        return {}
    return max(history, key=lambda r: r.get("val_sharpe", float("-inf")))


def main() -> None:
    parser = argparse.ArgumentParser(description="DiffQuant experiment comparison")
    parser.add_argument("--output-dir", "-o", default="output/",
                        help="Root output directory (default: output/)")
    parser.add_argument("--sort", default="val_sharpe",
                        choices=["val_sharpe", "test_sharpe", "backtest_sharpe"],
                        help="Metric to sort by (default: val_sharpe)")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    if not out_root.exists():
        log.error("Output directory not found: %s", out_root)
        sys.exit(1)

    rows = []
    for exp_dir in sorted(out_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "optuna":
            continue

        best = _load_val_history(exp_dir)
        if not best:
            continue

        row = {
            "name":             exp_dir.name,
            "best_epoch":       best.get("epoch",         "-"),
            "val_sharpe":       best.get("val_sharpe",     float("nan")),
            "val_sortino":      best.get("val_sortino",    float("nan")),
            "val_ret_pct":      best.get("val_final_ret",  float("nan")),
            "val_dd_pct":       best.get("val_max_drawdown", float("nan")),
            "val_flat":         best.get("val_flat_frac",  float("nan")),
            "test_sharpe":      _parse_report(exp_dir / "test_report.log",      "sharpe  (ann)"),
            "test_ret_pct":     _parse_report(exp_dir / "test_report.log",      "total return"),
            "test_dd_pct":      _parse_report(exp_dir / "test_report.log",      "max drawdown"),
            "backtest_sharpe":  _parse_report(exp_dir / "backtest_report.log",  "sharpe  (ann)"),
            "backtest_ret_pct": _parse_report(exp_dir / "backtest_report.log",  "total return"),
            "backtest_dd_pct":  _parse_report(exp_dir / "backtest_report.log",  "max drawdown"),
        }
        rows.append(row)

    if not rows:
        log.info("No completed experiments found in %s", out_root)
        return

    sort_key = args.sort
    rows.sort(key=lambda r: r.get(sort_key, float("-inf")), reverse=True)

    # ── Print table ───────────────────────────────────────────────────────────
    log.info("")
    log.info(_SEP)
    log.info("  DiffQuant — Experiment Comparison  (sorted by %s)", sort_key)
    log.info(_SEP)

    header = (
        f"  {'Experiment':<30} {'Ep':>4} | "
        f"{'vSharpe':>8} {'vSortino':>9} {'vRet%':>7} {'vDD%':>6} {'vFlat':>6} | "
        f"{'tSharpe':>8} {'tRet%':>7} {'tDD%':>6} | "
        f"{'bSharpe':>8} {'bRet%':>7} {'bDD%':>6}"
    )
    log.info(header)
    log.info(_SEP2)

    for r in rows:
        def _f(v, fmt="+.3f"):
            return f"{v:{fmt}}" if v == v else "   n/a"   # NaN check

        log.info(
            "  %-30s %4s | "
            "%8s %9s %7s %6s %6s | "
            "%8s %7s %6s | "
            "%8s %7s %6s",
            r["name"][:30],
            r["best_epoch"],
            _f(r["val_sharpe"]),
            _f(r["val_sortino"]),
            _f(r["val_ret_pct"],  "+.2f"),
            _f(r["val_dd_pct"],   ".2f"),
            _f(r["val_flat"],     ".3f"),
            _f(r["test_sharpe"]),
            _f(r["test_ret_pct"],     "+.2f"),
            _f(r["test_dd_pct"],      ".2f"),
            _f(r["backtest_sharpe"]),
            _f(r["backtest_ret_pct"], "+.2f"),
            _f(r["backtest_dd_pct"],  ".2f"),
        )

    log.info(_SEP2)
    log.info("  Total experiments: %d", len(rows))
    log.info(_SEP)
    log.info("")


if __name__ == "__main__":
    main()

