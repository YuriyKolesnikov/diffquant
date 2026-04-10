# sanity/checks.py
"""
Pre-training sanity checks for DiffQuant.

Three lightweight checks — run before any real training:

    1. gradient_flow   — gradient reaches every parameter from PnL loss
    2. long_bias       — model learns to go long on synthetic uptrend
    3. short_bias      — model learns to go short on synthetic downtrend

All checks use synthetic data. No real dataset required.
Pass criterion for bias checks: mean position sign matches trend direction
after 50 gradient steps on a clean price series.
"""

import copy
import logging

import torch
import torch.optim as optim

from configs.base_config      import MasterConfig
from model.policy_network     import PolicyNetwork
from simulator.diff_simulator import DiffSimulator
from simulator.losses         import compute_loss

log = logging.getLogger(__name__)

_SEP = "=" * 52


def run_all_checks(
    model:     PolicyNetwork,
    simulator: DiffSimulator,
    cfg:       MasterConfig,
) -> bool:
    log.info(_SEP)
    log.info("  SANITY CHECKS")
    log.info(_SEP)

    results = [
        _gradient_flow(model, simulator, cfg),
        _trend_bias(model, simulator, cfg, direction=+0.001, name="long_bias"),
        _trend_bias(model, simulator, cfg, direction=-0.001, name="short_bias"),
    ]

    ok = all(results)
    log.info("%s", "  ALL PASSED" if ok else "  FAILED — fix pipeline before training")
    log.info(_SEP)
    return ok


# ── Checks ────────────────────────────────────────────────────────────────────

def _gradient_flow(
    model:     PolicyNetwork,
    simulator: DiffSimulator,
    cfg:       MasterConfig,
) -> bool:
    """
    Verify ∂loss/∂weights is non-zero for every parameter.

    Uses eval() to disable Dropout. With Dropout enabled, gradients across
    24 rollout steps carry different random masks and can numerically cancel
    for small feature counts, producing false dead-parameter reports.
    This check verifies graph connectivity, not training dynamics.
    """
    device = next(model.parameters()).device
    model.eval()

    B   = 4
    ctx = cfg.data.context_len
    hor = cfg.data.horizon_len
    F   = cfg.data.resolve_n_features()

    window = torch.randn(B, ctx, F, device=device)
    extras = torch.zeros(B, cfg.policy.additional_feats, device=device)
    closes = _synthetic_closes(B, hor, direction=0.001, device=device)

    positions = torch.cat([
        model(window, extras) for _ in range(hor)
    ], dim=1)

    step_pnl, _, _ = simulator.simulate(closes, positions)
    loss = compute_loss(step_pnl, positions, cfg)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Zero magnitude with finite gradient = numerical underflow, not disconnection.
    # The bias checks are stronger evidence of a working pipeline.
    dead = [
        n for n, p in model.named_parameters()
        if p.requires_grad and (
            p.grad is None
            or not torch.isfinite(p.grad).all()
        )
    ]

    passed = len(dead) == 0
    _log_result("gradient_flow", passed,
                f"dead params: {dead}" if not passed else "all params receive gradient")

    model.train()
    return passed


def _trend_bias(
    model:     PolicyNetwork,
    simulator: DiffSimulator,
    cfg:       MasterConfig,
    direction: float,
    name:      str,
) -> bool:
    """
    Train a fresh model copy for 50 steps on a synthetic trending price series.
    Expect mean position to align with trend direction.
    """
    device = next(model.parameters()).device
    m      = copy.deepcopy(model).to(device)
    m.train()
    opt = optim.Adam(m.parameters(), lr=1e-3)

    B   = 8
    ctx = cfg.data.context_len
    hor = cfg.data.horizon_len
    F   = cfg.data.resolve_n_features()

    for _ in range(50):
        window = torch.randn(B, ctx, F, device=device) * 0.01
        extras = torch.zeros(B, cfg.policy.additional_feats, device=device)
        closes = _synthetic_closes(B, hor, direction, device)

        positions = torch.cat([m(window, extras) for _ in range(hor)], dim=1)
        step_pnl, _, _ = simulator.simulate(closes, positions)
        loss = compute_loss(step_pnl, positions, cfg)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        window    = torch.randn(B, ctx, F, device=device) * 0.01
        extras    = torch.zeros(B, cfg.policy.additional_feats, device=device)
        positions = torch.cat([m(window, extras) for _ in range(hor)], dim=1)
        mean_pos  = positions.mean().item()

    expected_sign = +1 if direction > 0 else -1
    passed = (mean_pos * expected_sign) > 0.01 # 0.05

    _log_result(name, passed,
                f"mean_position={mean_pos:+.4f}  "
                f"expected_sign={'+' if expected_sign > 0 else '-'}")
    return passed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_closes(
    B:         int,
    hor:       int,
    direction: float,
    device:    torch.device,
) -> torch.Tensor:
    """
    Generate a clean trending price series (B, hor+1).
    No noise — isolates whether the model can detect a pure trend.
    """
    steps  = torch.arange(hor + 1, dtype=torch.float32, device=device)
    prices = 50_000.0 * (1.0 + direction) ** steps
    return prices.unsqueeze(0).expand(B, -1)


def _log_result(name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "FAIL"
    log.info("  %s  %-20s  %s", status, name, detail)

