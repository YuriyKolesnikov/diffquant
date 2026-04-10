# evaluate.py
"""
DiffQuant — evaluation entry point.

Runs walk-forward evaluation on the test and backtest splits using
the best checkpoint saved during training.

Usage:
    python evaluate.py --config configs/experiments/itransformer_hybrid.py
    python evaluate.py --config configs/experiments/itransformer_hybrid.py \\
                       --checkpoint output/itransformer_hybrid/models/best.pth
    python evaluate.py --config configs/experiments/itransformer_hybrid.py \\
                       --mode test        # only test split
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DiffQuant evaluation")
    parser.add_argument("--config",     "-c", required=True)
    parser.add_argument("--checkpoint", "-k", default=None,
                        help="Path to .pth checkpoint. Default: output/<exp>/models/best.pth")
    parser.add_argument("--mode",       "-m", default="both",
                        choices=["test", "backtest", "both"],
                        help="Which splits to evaluate (default: both)")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    from configs.experiments import load_config
    cfg = load_config(args.config)

    # ── Attach file handler — all evaluation logs go to training.log ──────────
    import pathlib
    out_dir  = pathlib.Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path    = out_dir / "training.log"
    root_logger = logging.getLogger()
    already_added = any(
        isinstance(h, logging.FileHandler) and
        getattr(h, "baseFilename", None) == str(log_path.resolve())
        for h in root_logger.handlers
    )
    if not already_added:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        root_logger.addHandler(fh)

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint or Path(cfg.paths.output_dir) / "models" / "best.pth")
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    log.info("Experiment : %s", cfg.experiment_name)
    log.info("Checkpoint : %s", ckpt_path)
    log.info("Mode       : %s", args.mode)

    # ── Load model ────────────────────────────────────────────────────────────
    import torch
    from model.policy_network import PolicyNetwork

    device = torch.device(cfg.device)
    model  = PolicyNetwork(cfg)
    state  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    best_sharpe = state.get("best_val_sharpe", float("nan"))
    log.info("Best val Sharpe from training: %+.4f", best_sharpe)

    # ── Data ──────────────────────────────────────────────────────────────────
    from data.pipeline import load_or_build

    splits = load_or_build(
        source_path = cfg.paths.source_data,
        cfg         = cfg,
        cache_dir   = cfg.paths.cache_dir,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    from evaluation.backtest import run_backtest

    modes = (
        ["test", "backtest"] if args.mode == "both"
        else [args.mode]
    )

    for mode in modes:
        if mode not in splits or "raw_features" not in splits[mode]:
            log.error("Split '%s' not available in processed dataset.", mode)
            continue
        log.info("Running %s evaluation...", mode)
        run_backtest(
            model   = model,
            raw     = splits[mode],
            cfg     = cfg,
            mode    = mode,
        )


if __name__ == "__main__":
    main()

