# model/policy_head.py
"""
Two-headed policy output: direction * gate.

    position = tanh(direction_raw / τ_dir) * sigmoid(gate_raw / τ_gate)

direction: alpha signal — which way and how strongly to trade ∈ (-1, +1)
gate:      confidence  — whether to trade at all              ∈ ( 0, +1)
position:  risk-adjusted weight sent to the simulator         ∈ (-1, +1)

The gate creates a genuine flat-zone: when gate → 0, position → 0
regardless of direction. This is the differentiable analogue of
action masking from RL systems.

Gate bias is initialised to -1.0 so the model starts near-flat,
preventing large random trades during the first training steps.
"""

import torch
import torch.nn as nn
from typing import Dict
from configs.base_config import MasterConfig


class PolicyHead(nn.Module):

    def __init__(self, in_dim: int, cfg: MasterConfig):
        super().__init__()
        pcfg = cfg.policy

        # Shared trunk — processes combined (backbone + extras) features.
        layers = []
        d = in_dim
        for out_d in pcfg.dense_dims:
            layers += [
                nn.Linear(d, out_d),
                nn.LayerNorm(out_d),
                nn.GELU(),
                nn.Dropout(pcfg.dropout),
            ]
            d = out_d
        self.trunk = nn.Sequential(*layers)

        self.direction_head = nn.Linear(d, 1)
        self.gate_head      = nn.Linear(d, 1)
        self.tau_dir        = pcfg.tau_direction
        self.tau_gate       = pcfg.tau_gate

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # Conservative initialisation: model starts near flat.
        nn.init.constant_(self.gate_head.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim)
        Returns position: (B, 1) ∈ (-1, +1)
        """
        h         = self.trunk(x)
        direction = torch.tanh(self.direction_head(h) / self.tau_dir)
        gate      = torch.sigmoid(self.gate_head(h)  / self.tau_gate)
        return direction * gate

    def forward_with_components(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Returns position, direction, and gate for analysis and logging."""
        h         = self.trunk(x)
        direction = torch.tanh(self.direction_head(h) / self.tau_dir)
        gate      = torch.sigmoid(self.gate_head(h)  / self.tau_gate)
        return {"position": direction * gate, "direction": direction, "gate": gate}

