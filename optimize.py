# optimize.py
"""
DiffQuant — Optuna hyperparameter search entry point.

Runs a full train + val evaluation for each trial.
Optimises for best walk-forward val Sharpe.

Usage:
    python optimize.py --config configs/experiments/itransformer_hybrid.py --trials 50
    python optimize.py --config configs/experiments/itransformer_hybrid.py \\
                       --trials 100 --jobs 1 \\
                       --storage sqlite:///optuna_diffquant.db

Notes:
    - Each trial runs cfg.training.num_epochs epochs.
      Use a reduced epoch count in the config for search
      (e.g. 50–100 epochs), then retrain the best config to convergence.
    - Parallel jobs (--jobs > 1) require a persistent storage backend.
    - The best hyperparameters are printed and saved to
      output/<experiment>/optuna_best.json at the end.
"""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="DiffQuant hyperparameter search")
    parser.add_argument("--config",   "-c", required=True)
    parser.add_argument("--trials",   "-n", type=int, default=50)
    parser.add_argument("--jobs",     "-j", type=int, default=1)
    parser.add_argument("--storage",  "-s", default=None,
                        help="Optuna storage URL. Default: in-memory (not persistent).")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        log.error("optuna not installed. Run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from configs.experiments import load_config

    base_cfg = load_config(args.config)

    def objective(trial: optuna.Trial) -> float:
        """
        One trial = one full training run with sampled hyperparameters.
        Returns val Sharpe (annualised, from walk-forward evaluation).
        Returning -inf on failure allows Optuna to continue gracefully.
        """
        import copy
        cfg = copy.deepcopy(base_cfg)

        # ── Search space ──────────────────────────────────────────────────────
        # Keep the space tight — only parameters with high expected impact.
        cfg.training.lr            = trial.suggest_float("lr",            1e-5, 5e-4, log=True)
        cfg.training.weight_decay  = trial.suggest_float("weight_decay",  1e-6, 1e-3, log=True)
        cfg.training.gradient_clip = trial.suggest_float("gradient_clip", 0.3,  2.0)

        cfg.loss.lambda_turnover = trial.suggest_float("lambda_turnover", 0.001, 0.1,  log=True)
        cfg.loss.lambda_drawdown = trial.suggest_float("lambda_drawdown", 0.001, 0.05, log=True)
        cfg.loss.lambda_terminal = trial.suggest_float("lambda_terminal", 0.0,   0.01)

        cfg.policy.tau_gate      = trial.suggest_float("tau_gate",      0.3, 3.0)
        cfg.policy.tau_direction = trial.suggest_float("tau_direction",  0.3, 3.0)

        # Unique output dir per trial to avoid checkpoint collisions.
        cfg.paths.output_dir = str(
            Path(base_cfg.paths.output_dir) / "optuna" / f"trial_{trial.number}"
        )

        try:
            import torch
            from data.pipeline    import load_or_build
            from data.dataset     import TradingDataset
            from training.trainer import DiffTrainer

            splits   = load_or_build(base_cfg.paths.source_data, cfg, base_cfg.paths.cache_dir)
            train_ds = TradingDataset(splits["train"])
            trainer  = DiffTrainer(cfg)
            trainer.train(train_dataset=train_ds, raw_val=splits["val"])

            return trainer.best_val_sharpe

        except Exception as exc:
            log.warning("Trial %d failed: %s", trial.number, exc)
            return float("-inf")

    # ── Create study ──────────────────────────────────────────────────────────
    study = optuna.create_study(
        study_name    = base_cfg.experiment_name,
        direction     = "maximize",
        storage       = args.storage,
        load_if_exists= True,
        sampler       = optuna.samplers.TPESampler(seed=base_cfg.seed),
        # pruner removed: requires trial.report() calls inside training loop
        # pruner        = optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        objective,
        n_trials  = args.trials,
        n_jobs    = args.jobs,
        show_progress_bar = True,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    best = study.best_trial
    log.info("=" * 52)
    log.info("  OPTUNA SEARCH COMPLETE")
    log.info("  Best trial   : #%d", best.number)
    log.info("  Best Sharpe  : %+.4f", best.value)
    log.info("  Best params  :")
    for k, v in best.params.items():
        log.info("    %-25s = %s", k, v)
    log.info("=" * 52)

    # Save best params
    out_path = Path(base_cfg.paths.output_dir) / "optuna_best.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "best_trial":  best.number,
            "best_sharpe": best.value,
            "params":      best.params,
        }, f, indent=2)
    log.info("Best params saved → %s", out_path)


if __name__ == "__main__":
    main()

