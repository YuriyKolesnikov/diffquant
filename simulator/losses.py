# simulator/losses.py
"""
DiffQuant training objectives.

Three loss modes selected via cfg.loss.type:

    sharpe  — maximise Sharpe ratio: -(μ / sigma)
    sortino — maximise Sortino ratio: -(μ / sigma_down)
              Fully differentiable: downside std computed via ReLU,
              no boolean masks or graph breaks.
    hybrid  — weighted combination of financial objectives and
              behavioural regularisers (recommended for production).

Hybrid loss structure:

    L = λ_sharpe      · (-Sharpe or -Sortino)
      + λ_turnover    · mean(smooth_abs(Δpos[1:]))
      + λ_drawdown    · mean(log_running_max - log_equity)
      + λ_terminal    · smooth_abs(pos_T)
      + λ_flat_target · (flat_soft - flat_target)²
      + λ_bias        · |mean(position)|

Component notes:

    turnover:
        Δpos starts from index 1 — initial entry from flat is not penalised.
        Penalises only subsequent rebalancing during the episode.

    drawdown:
        Computed in log-equity space: cumsum(log1p(step_pnl)).
        Numerically stable for long horizons; avoids cumprod instability.

    flat_soft:
        Differentiable proxy for flat_fraction:
            sigmoid(flat_k · (flat_eps - |pos|))
        Approaches 1 when |pos| → 0, approaches 0 when |pos| → 1.
        flat_eps should match cfg.eval.exit_thr for loss/eval consistency.

    bias:
        |mean(position)| → 0 for symmetric long/short strategies,
        large for buy&hold or always-short strategies.
        Proved critical for breaking long bias on trending BTC data.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from configs.base_config      import MasterConfig
from simulator.diff_simulator import smooth_abs


# ── Public API ────────────────────────────────────────────────────────────────

def compute_loss(
    step_pnl:  torch.Tensor,   # (B, H)
    positions: torch.Tensor,   # (B, H)
    cfg:       MasterConfig,
) -> torch.Tensor:
    t = cfg.loss.type
    if   t == "sharpe":  return _sharpe(step_pnl, cfg)
    elif t == "sortino": return _sortino(step_pnl, cfg)
    elif t == "hybrid":  return _hybrid(step_pnl, positions, cfg)
    else:
        raise ValueError(f"Unknown loss type: {t!r}. Valid: sharpe | sortino | hybrid")


# ── Core objectives ───────────────────────────────────────────────────────────

def _sharpe(step_pnl: torch.Tensor, cfg: MasterConfig) -> torch.Tensor:
    flat = step_pnl.reshape(-1)
    return -(flat.mean() / (flat.std() + cfg.loss.eps))


def _sortino(step_pnl: torch.Tensor, cfg: MasterConfig) -> torch.Tensor:
    flat = step_pnl.reshape(-1)
    # Downside std via ReLU — fully differentiable, no graph breaks.
    downside_std = torch.sqrt(
        torch.mean(F.relu(-flat) ** 2) + cfg.loss.eps
    )
    return -(flat.mean() / (downside_std + cfg.loss.eps))


# ── Hybrid loss ───────────────────────────────────────────────────────────────

def _hybrid(
    step_pnl:  torch.Tensor,   # (B, H)
    positions: torch.Tensor,   # (B, H)
    cfg:       MasterConfig,
) -> torch.Tensor:
    lcfg    = cfg.loss
    eps_abs = cfg.simulator.smooth_abs_eps

    # 1. Core return-risk objective
    l_core = (
        _sortino(step_pnl, cfg)
        if lcfg.type_hybrid == "sortino_hybrid"
        else _sharpe(step_pnl, cfg)
    )

    # 2. Turnover — skip first step (initial entry from flat is not a rebalance)
    delta      = positions[:, 1:] - positions[:, :-1]   # (B, H-1)
    l_turnover = smooth_abs(delta, eps_abs).mean()

    # 3. Drawdown — log-stable: avoids cumprod numerical instability on long horizons
    log_equity  = torch.cumsum(torch.log1p(step_pnl), dim=1)   # (B, H)
    running_max = torch.cummax(log_equity, dim=1).values
    l_drawdown  = (running_max - log_equity).mean()

    # 4. Terminal position — encourages closing before episode end
    l_terminal = smooth_abs(positions[:, -1], eps_abs).mean()

    # 5. Flat target — differentiable proxy for flat_fraction
    #    flat_soft → 1.0 when |pos| → 0, → 0.0 when |pos| → 1.0
    flat_soft     = torch.sigmoid(
        lcfg.flat_k * (lcfg.flat_eps - positions.abs())
    ).mean()
    l_flat_target = (flat_soft - lcfg.flat_target).pow(2)

    # 6. Directional bias — penalises always-long / always-short
    l_bias = positions.mean().abs()

    return (
        lcfg.lambda_sharpe      * l_core        +
        lcfg.lambda_turnover    * l_turnover    +
        lcfg.lambda_drawdown    * l_drawdown    +
        lcfg.lambda_terminal    * l_terminal    +
        lcfg.lambda_flat_target * l_flat_target +
        lcfg.lambda_bias        * l_bias
    )

