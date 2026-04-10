# train.py
"""
DiffQuant — training entry point.

Usage:
    python train.py --config configs/experiments/itransformer_hybrid.py
    python train.py --config configs/experiments/lstm_hybrid.py --device cuda
    python train.py --config configs/experiments/itransformer_hybrid.py --skip-sanity
"""

import argparse
import logging
import pathlib
import sys

import torch

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

logging.basicConfig(
    level    = logging.INFO,   # root logger
    handlers = [console_handler],
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DiffQuant training pipeline")
    parser.add_argument("--config",      "-c", required=True,
                        help="Path to experiment config")
    parser.add_argument("--device",      "-d", default=None,
                        help="Override device: cpu | cuda | mps")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Skip sanity checks (not recommended)")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    from configs.experiments import load_config
    cfg = load_config(args.config)

    if args.device:
        cfg.device = args.device

    if cfg.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU")
        cfg.device = "cpu"
    if cfg.device == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS not available — falling back to CPU")
        cfg.device = "cpu"

    torch.manual_seed(cfg.seed)

    # ── Attach file handler immediately — before any other logging ────────────
    # This ensures ALL log messages (sanity checks, data pipeline, model init)
    # are written to training.log, not just those after DiffTrainer.__init__.
    out_dir = pathlib.Path(cfg.paths.output_dir)
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

    # ── Log run parameters ────────────────────────────────────────────────────
    log.info("Experiment : %s", cfg.experiment_name)
    log.info("Device     : %s", cfg.device)
    log.info("Backbone   : %s", cfg.backbone.type)
    log.info("Loss       : %s", cfg.loss.type)
    log.info("Output     : %s", cfg.paths.output_dir)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not args.skip_sanity:
        from model.policy_network     import PolicyNetwork
        from simulator.diff_simulator import DiffSimulator, SimConfig
        from sanity.checks            import run_all_checks

        _model = PolicyNetwork(cfg)
        _sim   = DiffSimulator(SimConfig(
            commission_rate = cfg.simulator.commission_rate,
            slippage_rate   = cfg.simulator.slippage_rate,
        ))
        if not run_all_checks(_model, _sim, cfg):
            log.error("Sanity checks failed. Aborting training.")
            sys.exit(1)
        del _model, _sim
    else:
        log.warning("Sanity checks skipped.")

    # ── Data ──────────────────────────────────────────────────────────────────
    from data.pipeline import load_or_build
    from data.dataset  import TradingDataset

    splits = load_or_build(
        source_path = cfg.paths.source_data,
        cfg         = cfg,
        cache_dir   = cfg.paths.cache_dir,
    )

    train_ds = TradingDataset(splits["train"])
    log.info(
        "Dataset  — train: %s samples | val raw: %s bars",
        f"{len(train_ds):,}",
        f"{len(splits['val']['raw_features']):,}",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    from training.trainer import DiffTrainer

    trainer = DiffTrainer(cfg)
    trainer.train(
        train_dataset = train_ds,
        raw_val       = splits["val"],
    )


if __name__ == "__main__":
    main()

