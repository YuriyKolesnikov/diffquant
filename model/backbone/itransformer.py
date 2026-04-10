# model/backbone/itransformer.py
"""
iTransformer encoder.
Reference: Liu et al., "iTransformer: Inverted Transformers are Effective
for Time Series Forecasting", ICLR 2024.

Key inversion: each feature channel is a token, not each timestep.
Attention operates across the variable dimension — capturing inter-channel
dependencies (price/volume relationships) that are more stable than
local temporal patterns in financial data.

Input : (B, T, C)
Output: (B, C * d_model)
"""

import torch
import torch.nn as nn
from configs.base_config import MasterConfig


class iTransformerEncoder(nn.Module):

    def __init__(self, cfg: MasterConfig):
        super().__init__()
        bcfg = cfg.backbone.itransformer
        n_ch = cfg.data.resolve_n_features()

        # Project each channel's time series (length T) into d_model space.
        self.channel_proj = nn.Linear(cfg.data.context_len, bcfg.d_model)
        self.input_norm   = nn.LayerNorm(bcfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = bcfg.d_model,
            nhead           = bcfg.n_heads,
            dim_feedforward = bcfg.d_ff,
            dropout         = bcfg.dropout,
            batch_first     = True,
            norm_first      = True,    # Pre-LN: more stable for financial data
            activation      = "gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers           = bcfg.n_layers,
            enable_nested_tensor = False,
        )
        self.output_norm = nn.LayerNorm(bcfg.d_model)
        self.output_dim  = n_ch * bcfg.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.permute(0, 2, 1)        # (B, C, T)
        x = self.channel_proj(x)      # (B, C, d_model)
        x = self.input_norm(x)
        x = self.transformer(x)       # (B, C, d_model)
        x = self.output_norm(x)
        return x.flatten(1)           # (B, C * d_model)

