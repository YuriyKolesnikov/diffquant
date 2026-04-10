# simulator/diff_simulator.py
"""
Fully differentiable mark-to-market trading simulator.

All operations are tensor ops — autograd traces the complete graph
from PnL back through positions into model weights.

PnL model per step:
    ret_t     = (close_t - close_{t-1}) / close_{t-1}
    gross_t   = position_{t-1} * ret_t          ← prior position earns current return
    delta_t   = position_t - position_{t-1}
    cost_t    = smooth_abs(delta_t) * (commission + slippage)
    net_pnl_t = gross_t - cost_t

smooth_abs replaces |·| to guarantee C∞ differentiability at zero.
Standard abs() has an undefined subgradient at x=0, causing gradient
instability when the model is near-flat.
"""

from dataclasses import dataclass
from typing      import Tuple, Dict

import torch


def smooth_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable |x|: sqrt(x² + ε). Gradient is 0 at x=0, not undefined."""
    return torch.sqrt(x.pow(2) + eps)


@dataclass
class SimConfig:
    commission_rate:   float = 0.0004   # Binance USDT-M Futures taker
    slippage_rate:     float = 0.0003
    smooth_abs_eps:    float = 1e-6
    market_impact_eta: float = 0.0      # Almgren-Chriss quadratic; 0 = disabled


class DiffSimulator:

    def __init__(self, cfg: SimConfig = None):
        self.cfg = cfg or SimConfig()

    def simulate(
        self,
        closes:    torch.Tensor,   # (B, H+1)  raw close prices
        positions: torch.Tensor,   # (B, H)    policy outputs ∈ (-1, +1)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate H trading steps for a batch of B episodes.

        closes[:, 0]  = last close of the context window (boundary price).
        closes[:, 1:] = H horizon close prices.

        Returns
        -------
        step_pnl : (B, H)  net PnL per step — primary gradient signal
        equity   : (B, H)  cumulative equity curve, starts at 1.0
        costs    : (B,)    total transaction costs per episode
        """
        assert closes.shape[1] == positions.shape[1] + 1, (
            f"closes must have one more timestep than positions. "
            f"Got closes={closes.shape}, positions={positions.shape}"
        )
        
        eps = self.cfg.smooth_abs_eps
        B, H = positions.shape

        # Step returns from H+1 close prices → H returns
        ret = (closes[:, 1:] - closes[:, :-1]) / (closes[:, :-1].abs() + 1e-10)

        # Previous positions: pad left with 0 (enter each episode flat)
        prev_pos = torch.cat([
            torch.zeros(B, 1, device=positions.device, dtype=positions.dtype),
            positions[:, :-1],
        ], dim=1)                                    # (B, H)

        gross   = prev_pos * ret                     # (B, H)
        delta   = positions - prev_pos               # (B, H)
        cost_t  = smooth_abs(delta, eps) * (self.cfg.commission_rate + self.cfg.slippage_rate)

        if self.cfg.market_impact_eta > 0.0:
            cost_t = cost_t + self.cfg.market_impact_eta * delta.pow(2)

        step_pnl = gross - cost_t                    # (B, H)
        equity   = torch.cumprod(1.0 + step_pnl, dim=1)
        costs    = cost_t.sum(dim=1)                 # (B,)

        return step_pnl, equity, costs

    @torch.no_grad()
    def metrics(
        self,
        step_pnl:  torch.Tensor,   # (B, H)
        positions: torch.Tensor,   # (B, H)
    ) -> Dict[str, float]:
        """Financial metrics for logging — detached from computation graph."""
        flat    = step_pnl.reshape(-1)
        ep_pnl  = step_pnl.sum(dim=1)              # (B,)

        prev_p  = torch.cat([
            torch.zeros_like(positions[:, :1]),
            positions[:, :-1],
        ], dim=1)
        turnover = smooth_abs(positions - prev_p, self.cfg.smooth_abs_eps).mean()

        eq       = torch.cumprod(1.0 + step_pnl.mean(0), dim=0)
        peak     = eq.cummax(0).values
        max_dd   = ((peak - eq) / (peak + 1e-8)).max()

        neg = flat[flat < 0]
        sortino = (
            (flat.mean() / (neg.std() + 1e-8)).item()
            if neg.numel() > 1 else 0.0
        )

        return {
            "mean_episode_pnl": ep_pnl.mean().item(),
            "win_rate":         (ep_pnl > 0).float().mean().item(),
            "sharpe":           (flat.mean() / (flat.std() + 1e-8)).item(),
            "sortino":          sortino,
            "max_drawdown":     max_dd.item(),
            "turnover":         turnover.item(),
            "flat_fraction":    (positions.abs() < 0.05).float().mean().item(),
        }

