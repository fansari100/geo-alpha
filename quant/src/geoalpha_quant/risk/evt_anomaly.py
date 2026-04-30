"""
Extreme Value Theory anomaly detection on spectral / radiometric data.

At AIG I used Generalised Pareto Distribution fits on the upper tail
of claim severities to estimate one-in-N-year loss thresholds; here
I'm doing the same thing on the upper tail of per-pixel
Mahalanobis-distance scores to estimate one-in-N-pixel anomaly
thresholds.

The advantage of EVT over a flat percentile cut-off is that you get
a *parametric* extrapolation of the tail - so you can pick a target
false-alarm rate (say 1e-6 per pixel) and the GPD fit will hand you
back the corresponding score threshold even if you didn't sample
deeply enough into the tail to see it empirically.

References
----------
Pickands (1975), McNeil & Frey (2000), Coles (2001).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GPDFit:
    """Result of a Peaks-Over-Threshold GPD fit."""

    threshold: float
    xi: float        # shape parameter
    sigma: float     # scale parameter
    n_exceed: int
    n_total: int

    @property
    def exceed_rate(self) -> float:
        return self.n_exceed / max(self.n_total, 1)


def fit_gpd(
    x: np.ndarray,
    threshold_quantile: float = 0.95,
) -> GPDFit:
    """Fit a Generalised Pareto Distribution to the upper tail of x.

    Uses Probability-Weighted Moments (Hosking & Wallis 1987) - a
    closed-form fit that's much more stable than maximum likelihood
    on the small tail samples you typically have in practice.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if not (0.5 < threshold_quantile < 1.0):
        raise ValueError("threshold_quantile must be in (0.5, 1)")

    u = float(np.quantile(x, threshold_quantile))
    excess = x[x > u] - u
    if excess.size < 30:
        raise ValueError(
            f"only {excess.size} exceedances above q={threshold_quantile} "
            "- raise the sample or lower the threshold"
        )

    # Probability-weighted moments.
    n = excess.size
    sorted_excess = np.sort(excess)
    weights = (np.arange(1, n + 1) - 1) / (n - 1)
    b0 = excess.mean()
    b1 = float((weights * sorted_excess).sum() / n)
    sigma = (2 * b0 * b1) / (b0 - 2 * b1)
    xi = 2 - b0 / (b0 - 2 * b1)

    return GPDFit(
        threshold=u,
        xi=float(xi),
        sigma=float(max(sigma, 1e-9)),
        n_exceed=int(excess.size),
        n_total=int(x.size),
    )


def return_period_threshold(fit: GPDFit, target_prob: float) -> float:
    """Score level x such that P(X > x) ~ target_prob.

    Inverts the GPD CDF in the standard POT framing:
        P(X > x | X > u) = (1 + xi (x - u) / sigma)^(-1/xi)
        P(X > x)         = (n_exc / n) * P(X > x | X > u)
    """
    if not (0.0 < target_prob < 1.0):
        raise ValueError("target_prob must be in (0, 1)")
    p_cond = target_prob / fit.exceed_rate
    if abs(fit.xi) < 1e-6:
        return fit.threshold + fit.sigma * (-np.log(p_cond))
    return fit.threshold + (fit.sigma / fit.xi) * (p_cond ** (-fit.xi) - 1.0)


# --------------------------------------------------------------------- #
# Detector wrapper.
# --------------------------------------------------------------------- #

class EVTAnomalyDetector:
    """Two-stage detector.

    1. Compute a Mahalanobis-style anomaly score per pixel (or per
       observation) - the caller is expected to provide the score map.
    2. Fit a GPD to the score's upper tail and pick a threshold that
       targets a desired false-alarm rate.

    This is the same contract as the RX detector but with a calibrated,
    extrapolated threshold instead of an empirical one - so the false
    positive rate stays predictable as the scene changes.
    """

    def __init__(
        self,
        threshold_quantile: float = 0.95,
        target_far: float = 1e-4,
    ):
        self.threshold_quantile = threshold_quantile
        self.target_far = target_far
        self.fit_: GPDFit | None = None
        self.score_threshold_: float | None = None

    def fit(self, scores: np.ndarray) -> EVTAnomalyDetector:
        self.fit_ = fit_gpd(scores, self.threshold_quantile)
        self.score_threshold_ = return_period_threshold(self.fit_, self.target_far)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.score_threshold_ is None:
            raise RuntimeError("call .fit() first")
        return np.asarray(scores) >= self.score_threshold_

    def fit_predict(self, scores: np.ndarray) -> np.ndarray:
        self.fit(scores)
        return self.predict(scores)
