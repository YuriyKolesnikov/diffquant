# data/dataset.py  — обновление под full_sequences
"""
TradingDataset: wraps pre-built split arrays from pipeline into PyTorch Dataset.

Each sample returns:
    full_seq : (ctx+hor, F) float32 — complete sequence for sliding-window rollout
    closes   : (hor+1,)     float32 — raw prices for the simulator
               closes[0] = last context close (boundary price)
               closes[1:] = horizon close prices
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class TradingDataset(Dataset):

    def __init__(self, split: dict):
        self._seqs   = torch.from_numpy(split["full_sequences"])  # (N, ctx+hor, F)
        self._closes = torch.from_numpy(split["horizon_closes"])  # (N, hor+1)
        self._ts     = split["timestamps"]                        # (N,) int64

        self.context_len = int(split["context_len"])
        self.horizon_len = int(split["horizon_len"])
        self.n_features  = int(split["n_features"])

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int):
        return self._seqs[idx], self._closes[idx]

    @property
    def timestamps(self) -> np.ndarray:
        return self._ts

