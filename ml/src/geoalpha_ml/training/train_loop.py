"""
Training loop with walk-forward train / OOS splits.

Same harness I use to train sequence models on financial data - the
distinction between in-sample and walk-forward OOS is non-negotiable
when the dataset is non-stationary, which both equity returns and
satellite revisit series very much are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..models.temporal_attention import (
    TemporalAttentionConfig,
    TemporalAttentionForecaster,
    build_model,
)


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: str = "cpu"
    train_window: int = 5000
    test_window: int = 1000


def _make_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float()
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=shuffle)


def train_one_epoch(
    model: TemporalAttentionForecaster,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        optim.zero_grad()
        pred = model(x)
        loss = TemporalAttentionForecaster.gaussian_nll(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def walk_forward_train(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    model_cfg: TemporalAttentionConfig | None = None,
) -> dict:
    """Walk-forward training with OOS evaluation.

    Returns
    -------
    dict
        train_loss_history : list[float] per epoch per fold
        oos_preds          : ndarray concatenated OOS predictions
        oos_truth          : ndarray concatenated OOS targets
    """
    n = X.shape[0]
    folds: List[Tuple[slice, slice]] = []
    t = 0
    while t + cfg.train_window + cfg.test_window <= n:
        folds.append((slice(t, t + cfg.train_window),
                      slice(t + cfg.train_window, t + cfg.train_window + cfg.test_window)))
        t += cfg.test_window

    history = []
    oos_preds: List[np.ndarray] = []
    oos_truth: List[np.ndarray] = []
    for tr, te in folds:
        model = build_model(model_cfg).to(cfg.device)
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        train_loader = _make_loader(X[tr], y[tr], cfg.batch_size, shuffle=True)
        fold_hist = []
        for _ in range(cfg.epochs):
            fold_hist.append(train_one_epoch(model, train_loader, optim, cfg))
        history.append(fold_hist)
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X[te]).float().to(cfg.device)
            pred = model(xt)["mean"].cpu().numpy()
        oos_preds.append(pred)
        oos_truth.append(y[te])

    return {
        "train_loss_history": history,
        "oos_preds": np.concatenate(oos_preds, axis=0) if oos_preds else np.empty(0),
        "oos_truth": np.concatenate(oos_truth, axis=0) if oos_truth else np.empty(0),
        "n_folds": len(folds),
    }
