"""
Change-point detectors for streaming geospatial signals.

Two flavours, with very different latency / detection-power trade-offs:

    BayesianOnlineChangePoint  - Adams & MacKay (2007).  Maintains a
        run-length distribution P(r_t | x_{1:t}).  Streaming, prior-
        sensitive, gives you a probabilistic "how confident am I that
        a change just happened" knob for free.  Cost: O(t) per step.
    cusum_change_point         - Page (1954) Cumulative Sum.  Fast,
        boring, very robust.  What you'd run on the on-board processor
        when you can't afford to allocate per-timestep.

Both are stateless w.r.t. each other so it's perfectly fine to wire
them up in parallel and AND/OR the alarms - that's what the dashboard
does for the "satellite revisit" demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# ---------------------------------------------------------------------
# Bayesian online change-point with a Normal-Inverse-Gamma prior.
# ---------------------------------------------------------------------

@dataclass
class _NIGPrior:
    """Normal-Inverse-Gamma prior, conjugate to a Gaussian likelihood."""

    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0


class BayesianOnlineChangePoint:
    """BOCPD on a 1-D Gaussian stream.

    Parameters
    ----------
    hazard_lambda : float
        Mean of the geometric prior on segment length.  Smaller =
        more frequent breaks expected.
    prior : _NIGPrior
        Normal-Inverse-Gamma hyperparameters.
    """

    def __init__(self, hazard_lambda: float = 250.0, prior: _NIGPrior | None = None):
        if hazard_lambda <= 1.0:
            raise ValueError("hazard_lambda must be > 1")
        self.hazard = 1.0 / hazard_lambda
        self.prior = prior or _NIGPrior()

        # Sufficient stats indexed by run length.
        self._mu = np.array([self.prior.mu0])
        self._kappa = np.array([self.prior.kappa0])
        self._alpha = np.array([self.prior.alpha0])
        self._beta = np.array([self.prior.beta0])
        self._R = np.array([1.0])  # P(r_t = 0) = 1 at t = 0

    @staticmethod
    def _student_t_pdf(x: float, mu: np.ndarray, kappa: np.ndarray,
                       alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        # Posterior predictive of a Normal-Inverse-Gamma is a Student-t.
        df = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        z = (x - mu) / scale
        c = (
            np.exp(_lgamma((df + 1.0) / 2.0) - _lgamma(df / 2.0))
            / (np.sqrt(df * np.pi) * scale)
        )
        return c * (1.0 + z * z / df) ** (-(df + 1.0) / 2.0)

    def update(self, x: float) -> dict:
        """Process one observation; return the new run-length distribution."""
        pred = self._student_t_pdf(x, self._mu, self._kappa, self._alpha, self._beta)

        # Growth probabilities: r_t = r_{t-1} + 1
        growth = self._R * pred * (1.0 - self.hazard)
        # Change-point probability: r_t = 0
        cp_mass = float(np.sum(self._R * pred * self.hazard))

        new_R = np.empty(self._R.size + 1)
        new_R[0] = cp_mass
        new_R[1:] = growth
        new_R = new_R / new_R.sum()
        self._R = new_R

        # Sufficient-stat update for the Normal-Inverse-Gamma prior.
        new_mu = np.empty(self._mu.size + 1)
        new_kappa = np.empty(self._kappa.size + 1)
        new_alpha = np.empty(self._alpha.size + 1)
        new_beta = np.empty(self._beta.size + 1)
        new_mu[0] = self.prior.mu0
        new_kappa[0] = self.prior.kappa0
        new_alpha[0] = self.prior.alpha0
        new_beta[0] = self.prior.beta0
        new_mu[1:] = (self._kappa * self._mu + x) / (self._kappa + 1.0)
        new_kappa[1:] = self._kappa + 1.0
        new_alpha[1:] = self._alpha + 0.5
        new_beta[1:] = self._beta + (self._kappa * (x - self._mu) ** 2) / (
            2.0 * (self._kappa + 1.0)
        )
        self._mu, self._kappa, self._alpha, self._beta = new_mu, new_kappa, new_alpha, new_beta

        return {
            "run_length": self._R.copy(),
            "cp_prob": cp_mass,
        }

    def run(self, x: np.ndarray) -> np.ndarray:
        """Process a batch and return the per-step change-point posterior."""
        out = np.empty(x.size)
        for i, xi in enumerate(np.asarray(x, dtype=np.float64).ravel()):
            out[i] = self.update(float(xi))["cp_prob"]
        return out


def _lgamma(x):
    from math import lgamma
    if np.isscalar(x):
        return lgamma(float(x))
    return np.array([lgamma(float(xi)) for xi in np.atleast_1d(x)])


# ---------------------------------------------------------------------
# CUSUM - cheap and cheerful, perfect for on-board edge detection.
# ---------------------------------------------------------------------

def cusum_change_point(
    x: np.ndarray,
    target: float | None = None,
    drift: float = 0.0,
    threshold: float = 5.0,
) -> List[int]:
    """Two-sided CUSUM change-point indices.

    Parameters
    ----------
    x : ndarray
        Observations.
    target : float, optional
        Reference mean.  Defaults to the running mean of x.
    drift : float
        Allowance / slack term (k in the literature).
    threshold : float
        Decision boundary (h).  Higher = fewer false alarms.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mu = target if target is not None else float(x.mean())
    s_pos = 0.0
    s_neg = 0.0
    out: List[int] = []
    for i, xi in enumerate(x):
        s_pos = max(0.0, s_pos + (xi - mu) - drift)
        s_neg = min(0.0, s_neg + (xi - mu) + drift)
        if s_pos > threshold or s_neg < -threshold:
            out.append(i)
            s_pos = 0.0
            s_neg = 0.0
    return out
