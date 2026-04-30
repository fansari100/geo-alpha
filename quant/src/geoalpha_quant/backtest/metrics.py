"""
Evaluation metrics.

Standard binary detection scores plus the rank-IC / Spearman analogues
I always reach for on the alpha-research side - they're handy for
ranking detector confidence against ground-truth severity.
"""

from __future__ import annotations

import numpy as np


def detection_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Standard precision / recall / F1 + radar-style PD / FAR."""
    pred = np.asarray(pred, dtype=np.int32).ravel()
    truth = np.asarray(truth, dtype=np.int32).ravel()
    if pred.shape != truth.shape:
        raise ValueError("pred and truth must share shape")
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    far = fp / max(fp + tn, 1)
    pd = recall  # synonym in the radar literature
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "far": float(far), "pd": float(pd),
    }


def information_coefficient(scores: np.ndarray, truth: np.ndarray) -> float:
    """Pearson IC - linear correlation between score and truth."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(truth, dtype=np.float64).ravel()
    if s.size < 2:
        return 0.0
    s_centered = s - s.mean()
    t_centered = t - t.mean()
    denom = (s_centered.std() * t_centered.std() * s.size) + 1e-12
    return float(np.dot(s_centered, t_centered) / denom)


def rank_ic(scores: np.ndarray, truth: np.ndarray) -> float:
    """Spearman rank-IC - same idea but rank-based, robust to outliers."""
    s = _rank(np.asarray(scores, dtype=np.float64).ravel())
    t = _rank(np.asarray(truth, dtype=np.float64).ravel())
    return information_coefficient(s, t)


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks
