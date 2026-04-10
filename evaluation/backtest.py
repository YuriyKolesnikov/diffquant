# evaluation/backtest.py — финальная версия без дублирования отчётов

"""
Backtest runner: delegates evaluation to WalkForwardEvaluator,
reporting to MetricsLogger, and plots to Visualizer.
"""

import logging
from configs.base_config       import MasterConfig
from evaluation.walk_forward   import WalkForwardEvaluator, WalkForwardResult
from model.policy_network      import PolicyNetwork
from utils.logging_utils       import MetricsLogger
from utils.visualization       import Visualizer

log = logging.getLogger(__name__)


def run_backtest(
    model:   PolicyNetwork,
    raw:     dict,
    cfg:     MasterConfig,
    mode:    str = "backtest",
) -> WalkForwardResult:
    """
    Run walk-forward evaluation with full reporting artifacts.

    Parameters
    ----------
    raw  : dict from pipeline.load_or_build() — val, test, or backtest split.
    mode : controls output file names ("val" | "test" | "backtest").
    """
    evaluator = WalkForwardEvaluator(model, cfg)
    result    = evaluator.run(raw, mode=mode)

    logger = MetricsLogger(cfg.paths.output_dir)
    logger.write_report(result, mode=mode, min_delta_rebalance=cfg.eval.min_delta_to_rebalance)

    viz = Visualizer(cfg.paths.output_dir + "/plots")
    viz.plot_equity_report(
        result,
        mode=mode,
        tf_min=cfg.data.timeframe_min,
        exit_thr=cfg.eval.exit_thr
    )
    viz.plot_position_distribution(result, mode=mode, exit_thr=cfg.eval.exit_thr)

    log.info(
        "[%s] Sharpe=%+.3f  Return=%+.2f%%  MaxDD=%.2f%%  Rebalances=%d",
        mode,
        result.metrics["sharpe"],
        result.metrics["final_return_pct"],
        result.metrics["max_drawdown_pct"],
        len(result.rebalance_records),
    )

    return result

