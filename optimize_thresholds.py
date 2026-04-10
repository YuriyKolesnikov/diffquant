# optimize_thresholds.py
"""
DiffQuant — Inference threshold optimisation via Optuna.

Searches for the optimal hysteresis thresholds for a TRAINED model
by running walk-forward evaluation on the VALIDATION split only.
The best thresholds are then applied to test and backtest splits
exactly ONCE — preserving their out-of-sample integrity.

Why threshold optimisation matters:
    The raw model output is a continuous position signal ∈ (−1, +1).
    Four thresholds control how that signal translates into actual trades:
        enter_long_thr  — minimum signal to enter / hold long
        enter_short_thr — minimum signal to enter / hold short
        exit_thr        — signal below which we flatten the position
        min_delta       — minimum position change to trigger a rebalance

    These thresholds interact non-linearly with commission costs, signal
    strength, and regime. Grid search is too coarse; Optuna TPE samples
    the space efficiently even with 200–500 fast inference trials.

Methodological contract:
    Thresholds are optimised on VAL data only.
    TEST and BACKTEST splits are never touched during this search.
    After optimisation, update EvalConfig in the experiment config
    and run evaluate.py exactly once for the final honest assessment.

Usage:
    # Basic — optimise on val, 300 trials, in-memory
    python optimize_thresholds.py \\
        --config     configs/experiments/itransformer_hybrid.py \\
        --checkpoint output/itransformer_hybrid/models/best.pth \\
        --trials     300

    # With persistent storage (resume-able)
    python optimize_thresholds.py \\
        --config     configs/experiments/itransformer_hybrid.py \\
        --checkpoint output/itransformer_hybrid/models/best.pth \\
        --trials     500 \\
        --storage    sqlite:///output/itransformer_hybrid/threshold_search.db

    # Optimise for Sortino instead of Sharpe
    python optimize_thresholds.py \\
        --config     configs/experiments/itransformer_hybrid.py \\
        --checkpoint output/itransformer_hybrid/models/best.pth \\
        --trials     300 \\
        --objective  sortino
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Silence matplotlib / other noisy loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DiffQuant — inference threshold optimisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to experiment config (e.g. configs/experiments/itransformer_hybrid.py)",
    )
    parser.add_argument(
        "--checkpoint", "-k", default=None,
        help="Path to .pth checkpoint. Default: output/<exp>/models/best.pth",
    )
    parser.add_argument(
        "--trials", "-n", type=int, default=300,
        help="Number of Optuna trials (default: 300). "
             "300–500 is sufficient; each trial takes ~1–3 sec.",
    )
    parser.add_argument(
        "--objective", "-o",
        choices=["sharpe", "sortino", "calmar"],
        default="sharpe",
        help="Metric to maximise (default: sharpe).",
    )
    parser.add_argument(
        "--storage", "-s", default=None,
        help="Optuna storage URL for persistence. "
             "Example: sqlite:///output/itransformer_hybrid/threshold_search.db",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1,
        help="Parallel jobs (default: 1). >1 requires persistent storage.",
    )
    args = parser.parse_args()

    # ── Optuna import ─────────────────────────────────────────────────────────
    try:
        import optuna
    except ImportError:
        log.error("optuna not installed. Run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Load config ───────────────────────────────────────────────────────────
    from configs.experiments import load_config
    base_cfg = load_config(args.config)

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    ckpt_path = Path(
        args.checkpoint or
        Path(base_cfg.paths.output_dir) / "models" / "best.pth"
    )
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        log.error("Train the model first with: python train.py --config %s", args.config)
        sys.exit(1)

    # ── Attach file handler — logs go to training.log alongside train logs ────
    out_dir  = Path(base_cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path    = out_dir / "training.log"
    root_logger = logging.getLogger()
    already_added = any(
        isinstance(h, logging.FileHandler) and
        getattr(h, "baseFilename", None) == str(log_path.resolve())
        for h in root_logger.handlers
    )
    if not already_added:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root_logger.addHandler(fh)

    # ── Header ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  THRESHOLD OPTIMISATION")
    log.info("  Experiment  : %s", base_cfg.experiment_name)
    log.info("  Checkpoint  : %s", ckpt_path)
    log.info("  Objective   : %s", args.objective)
    log.info("  Trials      : %d", args.trials)
    log.info("  Split       : val  (test/backtest untouched)")
    log.info("=" * 60)

    # ── Load model once — shared across all trials ────────────────────────────
    from model.policy_network import PolicyNetwork

    device = torch.device(base_cfg.device)
    model  = PolicyNetwork(base_cfg)
    state  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    training_sharpe = state.get("best_val_sharpe", float("nan"))
    log.info(
        "Model loaded — %s params | best_val_sharpe_during_training=%+.4f",
        f"{model.n_params:,}", training_sharpe,
    )

    # ── Load data once — shared across all trials ─────────────────────────────
    from data.pipeline import load_or_build

    splits = load_or_build(
        source_path = base_cfg.paths.source_data,
        cfg         = base_cfg,
        cache_dir   = base_cfg.paths.cache_dir,
    )

    raw_val = splits["val"]
    log.info(
        "Val split loaded — %s bars  [%s → %s]",
        f"{len(raw_val['raw_features']):,}",
        _ms_to_str(raw_val["raw_timestamps"][0]),
        _ms_to_str(raw_val["raw_timestamps"][-1]),
    )

    # ── Optuna objective ──────────────────────────────────────────────────────
    from evaluation.walk_forward import WalkForwardEvaluator

    def objective(trial: optuna.Trial) -> float:
        """
        One trial = one walk-forward pass on val with sampled thresholds.
        Inference only — no training, no gradient computation.
        Each trial takes ~1–3 seconds.
        """
        cfg = copy.deepcopy(base_cfg)

        # ── Threshold search space ────────────────────────────────────────────
        # Constraints:
        #   exit_thr < enter_long_thr   (exit before full entry reversal)
        #   exit_thr < enter_short_thr
        #   min_delta <= exit_thr        (delta cannot exceed exit zone)
        enter_long  = trial.suggest_float("enter_long_thr",        0.05, 0.60)
        enter_short = trial.suggest_float("enter_short_thr",       0.05, 0.60)
        exit_thr    = trial.suggest_float("exit_thr",              0.01, 0.30)
        min_delta   = trial.suggest_float("min_delta_to_rebalance",0.01, 0.20)

        # Enforce logical constraint: exit zone must be inside entry zone.
        # Pruning invalid combinations avoids noisy evaluations.
        if exit_thr >= enter_long or exit_thr >= enter_short:
            raise optuna.TrialPruned()

        if min_delta > exit_thr:
            raise optuna.TrialPruned()

        cfg.eval.enter_long_thr         = enter_long
        cfg.eval.enter_short_thr        = enter_short
        cfg.eval.exit_thr               = exit_thr
        cfg.eval.min_delta_to_rebalance = min_delta

        try:
            evaluator = WalkForwardEvaluator(model, cfg)
            result    = evaluator.run(raw_val, mode="val_opt")
            m         = result.metrics

            # Return chosen objective metric.
            score = m.get(args.objective, m["sharpe"])

            # Penalise degenerate strategies:
            #   - Never trades (all flat)
            #   - Never goes flat (no risk management)
            #   - Tiny number of rebalances (overfitting to noise)
            if m["flat_fraction"] > 0.99:
                return float("-inf")   # model never trades
            if m["flat_fraction"] < 0.01 and m["turnover"] > 0.05:
                return float("-inf")   # churning — commission drain
            if len(result.rebalance_records) < 2:
                return float("-inf")   # too few trades for statistical meaning

            return float(score)

        except Exception as exc:
            log.debug("Trial %d failed: %s", trial.number, exc)
            return float("-inf")

    # ── Create / resume study ─────────────────────────────────────────────────
    study_name = f"{base_cfg.experiment_name}_thresholds"
    study = optuna.create_study(
        study_name     = study_name,
        direction      = "maximize",
        storage        = args.storage,
        load_if_exists = True,
        sampler        = optuna.samplers.TPESampler(
            seed            = base_cfg.seed,
            n_startup_trials= 30,    # random exploration before TPE kicks in
        ),
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=20),
    )

    completed_before = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    if completed_before > 0:
        log.info(
            "Resuming study — %d trials already completed.", completed_before
        )

    # ── Run search ────────────────────────────────────────────────────────────
    log.info("Starting threshold search (%d trials)...", args.trials)

    study.optimize(
        objective,
        n_trials          = args.trials,
        n_jobs            = args.jobs,
        show_progress_bar = True,
        gc_after_trial    = True,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned    = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]

    if not completed:
        log.error("No trials completed successfully. Check constraints or data.")
        sys.exit(1)

    best = study.best_trial

    log.info("=" * 60)
    log.info("  THRESHOLD SEARCH COMPLETE")
    log.info("  Trials completed : %d", len(completed))
    log.info("  Trials pruned    : %d", len(pruned))
    log.info("  Best trial       : #%d", best.number)
    log.info("  Best %-10s  : %+.4f", args.objective, best.value)
    log.info("  Best thresholds  :")
    log.info("    enter_long_thr         = %.4f", best.params["enter_long_thr"])
    log.info("    enter_short_thr        = %.4f", best.params["enter_short_thr"])
    log.info("    exit_thr               = %.4f", best.params["exit_thr"])
    log.info("    min_delta_to_rebalance = %.4f", best.params["min_delta_to_rebalance"])
    log.info("=" * 60)

    # ── Top-5 trials for reference ────────────────────────────────────────────
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    log.info("  Top-5 trials by %s:", args.objective)
    for i, t in enumerate(top5, 1):
        log.info(
            "    #%d  trial=%-4d  %s=%+.4f  "
            "enter=(%.3f/%.3f)  exit=%.3f  delta=%.3f",
            i, t.number, args.objective, t.value,
            t.params["enter_long_thr"],
            t.params["enter_short_thr"],
            t.params["exit_thr"],
            t.params["min_delta_to_rebalance"],
        )

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = out_dir / "threshold_best.json"
    result_dict = {
        "experiment":   base_cfg.experiment_name,
        "checkpoint":   str(ckpt_path),
        "objective":    args.objective,
        "trials_total": len(study.trials),
        "trials_done":  len(completed),
        "best_trial":   best.number,
        "best_score":   best.value,
        "best_params": {
            "enter_long_thr":         best.params["enter_long_thr"],
            "enter_short_thr":        best.params["enter_short_thr"],
            "exit_thr":               best.params["exit_thr"],
            "min_delta_to_rebalance": best.params["min_delta_to_rebalance"],
        },
        "top5": [
            {
                "trial":    t.number,
                "score":    t.value,
                "params":   t.params,
            }
            for t in top5
        ],
    }
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    log.info("Results saved → %s", out_path)

    # ── Config snippet ────────────────────────────────────────────────────────
    log.info("")
    log.info("  Copy these values into your experiment config:")
    log.info("  ─" * 30)
    log.info("  cfg.eval.enter_long_thr         = %.4f", best.params["enter_long_thr"])
    log.info("  cfg.eval.enter_short_thr        = %.4f", best.params["enter_short_thr"])
    log.info("  cfg.eval.exit_thr               = %.4f", best.params["exit_thr"])
    log.info("  cfg.eval.min_delta_to_rebalance = %.4f", best.params["min_delta_to_rebalance"])
    log.info("  ─" * 30)
    log.info("  Then run:")
    log.info("  python evaluate.py --config %s", args.config)
    log.info("")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms_to_str(ts_ms: int) -> str:
    import pandas as pd
    return pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()

