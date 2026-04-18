"""
Temporal-attention forecaster.

Architectural lineage
---------------------
This is a refactor of the temporal attention encoder I trained on five
years of minute-bar equity data for a research project (1.8
walk-forward Sharpe, -4.2% MaxDD).  Here it forecasts the next-revisit
spectral observation for a region of interest given the past N
revisits, plus auxiliary covariates (sun zenith, AOD, day-of-year).

The architecture is intentionally compact - 4 attention heads, 2
encoder blocks, 64-channel embeddings - because we want to be able
to (a) train it on a single GPU and (b) quantize it to INT8 and run
it on the edge under 5 ms.  Both targets fall out of the same
constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalAttentionConfig:
    in_features: int = 8         # bands + covariates
    seq_len: int = 32            # past revisits
    horizon: int = 1             # forecast horizon
    d_model: int = 64
    n_heads: int = 4
    n_blocks: int = 2
    dropout: float = 0.10
    out_features: int = 1        # default: one scalar (e.g. NDVI)


def _sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() *
                    -(torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class _TemporalBlock(nn.Module):
    def __init__(self, cfg: TemporalAttentionConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(a))
        h = self.ff(x)
        x = self.norm2(x + self.drop(h))
        return x


class TemporalAttentionForecaster(nn.Module):
    """Encoder-only transformer for forecasting next observation."""

    def __init__(self, cfg: TemporalAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.in_features, cfg.d_model)
        self.register_buffer("pe", _sinusoidal_pe(cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([_TemporalBlock(cfg) for _ in range(cfg.n_blocks)])
        self.head_mean = nn.Linear(cfg.d_model, cfg.horizon * cfg.out_features)
        self.head_logvar = nn.Linear(cfg.d_model, cfg.horizon * cfg.out_features)

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, F) - last T revisits.
        h = self.input_proj(x) + self.pe[:, : x.size(1)]
        for blk in self.blocks:
            h = blk(h)
        # Pool over the time axis with attention to the final position.
        last = h[:, -1]
        mean = self.head_mean(last).view(-1, self.cfg.horizon, self.cfg.out_features)
        logvar = self.head_logvar(last).view(-1, self.cfg.horizon, self.cfg.out_features)
        logvar = torch.clamp(logvar, min=-8.0, max=4.0)
        return {"mean": mean, "logvar": logvar}

    @staticmethod
    def gaussian_nll(pred: dict, target: torch.Tensor) -> torch.Tensor:
        """Heteroscedastic Gaussian negative log-likelihood loss."""
        mean = pred["mean"]
        logvar = pred["logvar"]
        return 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar)).mean()


def build_model(cfg: TemporalAttentionConfig | None = None) -> TemporalAttentionForecaster:
    cfg = cfg or TemporalAttentionConfig()
    return TemporalAttentionForecaster(cfg)
