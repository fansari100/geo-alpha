"""
Walk-forward threshold search for any score-based detector.

This is the geospatial twin of the walk-forward optimizer I built at
Compak: rolling train / out-of-sample test windows, evaluate a
parameter sweep on each in-sample slice, freeze the best parameters
for the next out-of-sample slice, then concatenate the OOS predictions
to compute a single honest performance number.

Why bother with walk-forward instead of cross-validation?  Same reason
as in finance: the data is non-stationary.  A k-fold CV that mixes
post-2020 imagery with pre-2020 imagery in the same fold will flatter
the model in a way that won't survive deployment.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .metrics import detection_metrics


@dataclass
class WalkForwardConfig:
    train_window: int
    test_window: int
    step: int | None = None  # default = test_window (non-overlapping OOS)


@dataclass
class DetectorResult:
    """Evaluation result for one detector / threshold combination."""

    threshold: float
    precision: float
    recall: float
    f1: float
    far: float                  # false alarm rate per pixel
    pd: float                   # probability of detection
    n_alarms: int
    n_truth: int


def walk_forward_threshold_search(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_thresholds: Iterable[float],
    config: WalkForwardConfig,
    metric: str = "f1",
) -> dict:
    """Walk-forward evaluator over a 1-D time series of scores.

    Parameters
    ----------
    scores : (T,) anomaly / detection scores.
    labels : (T,) ground-truth binary labels.
    candidate_thresholds : iterable of thresholds to evaluate in-sample.
    config : window definitions.
    metric : 'f1' | 'pd' | 'precision' | 'recall' to maximise.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must share shape")

    step = config.step if config.step is not None else config.test_window
    T = scores.size
    chosen: list[float] = []
    oos_pred = np.zeros(T, dtype=np.int32)
    coverage = np.zeros(T, dtype=bool)

    t = 0
    while t + config.train_window + config.test_window <= T:
        s_train = slice(t, t + config.train_window)
        s_test = slice(t + config.train_window, t + config.train_window + config.test_window)
        best_thr, _best = _select_threshold(
            scores[s_train], labels[s_train], candidate_thresholds, metric
        )
        chosen.append(best_thr)
        oos_pred[s_test] = (scores[s_test] >= best_thr).astype(np.int32)
        coverage[s_test] = True
        t += step

    # Aggregate honest OOS metrics across the spliced windows.
    metrics_oos = detection_metrics(oos_pred[coverage], labels[coverage])
    return {
        "thresholds": np.asarray(chosen),
        "oos_predictions": oos_pred,
        "oos_coverage": coverage,
        "oos_metrics": metrics_oos,
    }


def _select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    candidates: Iterable[float],
    metric: str,
) -> tuple[float, DetectorResult]:
    best_score = -np.inf
    best_thr = float(np.median(scores))
    best_res: DetectorResult | None = None
    for thr in candidates:
        pred = (scores >= thr).astype(np.int32)
        m = detection_metrics(pred, labels)
        s = float(m.get(metric, 0.0))
        if s > best_score:
            best_score = s
            best_thr = float(thr)
            best_res = DetectorResult(
                threshold=best_thr,
                precision=m["precision"],
                recall=m["recall"],
                f1=m["f1"],
                far=m["far"],
                pd=m["pd"],
                n_alarms=int(pred.sum()),
                n_truth=int(labels.sum()),
            )
    return best_thr, best_res  # type: ignore
