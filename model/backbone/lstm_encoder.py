# model/backbone/lstm_encoder.py
"""
Bidirectional LSTM encoder.

Bidirectionality is valid here: the model observes the complete context
window [t-ctx : t] before any trading decision — no sequential prediction
is involved, so processing the window in both directions is leak-free.

Input : (B, T, C)
Output: (B, hidden_size * 2)  — concatenated final states from both directions
"""

import torch
import torch.nn as nn
from configs.base_config import MasterConfig


class LSTMEncoder(nn.Module):

    def __init__(self, cfg: MasterConfig):
        super().__init__()
        bcfg = cfg.backbone.lstm

        self.lstm = nn.LSTM(
            input_size    = cfg.data.resolve_n_features(),
            hidden_size   = bcfg.hidden_size,
            num_layers    = bcfg.num_layers,
            dropout       = bcfg.dropout if bcfg.num_layers > 1 else 0.0,
            bidirectional = bcfg.bidirectional,
            batch_first   = True,
        )

        dirs            = 2 if bcfg.bidirectional else 1
        self.output_dim = bcfg.hidden_size * dirs
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * dirs, B, hidden_size)
        # Take the last layer's hidden states from both directions.
        if self.lstm.bidirectional:
            # Forward final state: h_n[-2], backward: h_n[-1]
            out = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, hidden*2)
        else:
            out = h_n[-1]                                  # (B, hidden)
        return self.output_norm(out)

