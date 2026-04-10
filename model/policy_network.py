# model/policy_network.py
"""
PolicyNetwork: backbone + policy head → continuous position ∈ (-1, +1).

Used identically at training time (episode rollout, one step at a time)
and at inference / backtest time (one forward pass per new bar).

Extras vector fed alongside backbone output:
    [prev_position, prev_delta_position, time_elapsed, time_remaining]
    4 scalar channels giving the model path-dependent context without
    any discrete variables that would break backprop.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict
from configs.base_config import MasterConfig
from model.backbone.itransformer import iTransformerEncoder
from model.backbone.lstm_encoder  import LSTMEncoder
from model.policy_head            import PolicyHead

log = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):

    def __init__(self, cfg: MasterConfig):
        super().__init__()

        # Backbone
        if cfg.backbone.type == "itransformer":
            self.backbone: nn.Module = iTransformerEncoder(cfg)
        elif cfg.backbone.type == "lstm":
            self.backbone = LSTMEncoder(cfg)
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone.type!r}")

        backbone_dim = cfg.backbone_output_dim()
        head_in_dim  = backbone_dim + cfg.policy.additional_feats

        self.head = PolicyHead(head_in_dim, cfg)

        n = sum(p.numel() for p in self.parameters())
        self._n_params = n
        
        from utils.utils import millify
        log.info(
            "PolicyNetwork created — backbone=%s  params=%s",
            cfg.backbone.type,
            millify(n, precision=1),
        )

    def forward(
        self,
        market_seq: torch.Tensor,   # (B, T, F)
        extras:     torch.Tensor,   # (B, additional_feats)
    ) -> torch.Tensor:
        assert market_seq.ndim == 3, \
            f"market_seq must be (B, T, F), got shape {market_seq.shape}"
        assert extras.ndim == 2, \
            f"extras must be (B, additional_feats), got shape {extras.shape}"
        features = self.backbone(market_seq)
        combined = torch.cat([features, extras], dim=1)
        return self.head(combined)

    def forward_with_components(
        self,
        market_seq: torch.Tensor,
        extras:     torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Returns position, direction, gate — for logging and analysis."""
        features = self.backbone(market_seq)
        combined = torch.cat([features, extras], dim=1)
        return self.head.forward_with_components(combined)

    @property
    def n_params(self) -> int:
        return self._n_params

