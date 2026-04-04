"""
Gaussian HMM with Baum-Welch fitting and Viterbi decoding.

I originally wrote this for vol-regime detection on intraday returns;
the only thing that changes when you point it at a satellite NDVI
series or a per-pixel radiance trace is the *interpretation* of the
states.  All the matrix machinery is identical.

Notation follows Rabiner (1989):

    A   - K x K transition matrix, A[i, j] = P(state j at t+1 | state i at t)
    pi  - K-vector initial state distribution
    mu  - K-vector emission means         (1-D obs in this minimal impl.)
    var - K-vector emission variances
    O   - T-vector of observations

Forward-backward is run in log-space to avoid the underflow that
bites you the moment you cross ~500 observations on float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


_LOG_2PI = float(np.log(2.0 * np.pi))


def _log_gauss_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return -0.5 * (_LOG_2PI + np.log(var) + (x - mu) ** 2 / var)


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    if axis is None:
        a_max = float(np.max(a))
        if not np.isfinite(a_max):
            a_max = 0.0
        return np.log(np.sum(np.exp(a - a_max))) + a_max
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return np.squeeze(out, axis=axis)


@dataclass
class HMMParams:
    pi: np.ndarray   # (K,)
    A: np.ndarray    # (K, K)
    mu: np.ndarray   # (K,)
    var: np.ndarray  # (K,)


class GaussianHMM:
    """Vanilla Gaussian-emission HMM.

    Parameters
    ----------
    n_states : int
        Number of latent regimes.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Stop EM when the log-likelihood improvement falls below tol.
    seed : int
        For reproducible random init.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-4, seed: int = 0):
        if n_states < 2:
            raise ValueError("HMM with <2 states is degenerate")
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.params: HMMParams | None = None
        self.history_: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, obs: np.ndarray) -> "GaussianHMM":
        obs = np.asarray(obs, dtype=np.float64).ravel()
        if obs.size < self.n_states:
            raise ValueError("need at least one obs per state for init")
        rng = np.random.default_rng(self.seed)
        K = self.n_states
        # k-quantile init keeps means well separated and EM stable.
        qs = np.quantile(obs, np.linspace(0.1, 0.9, K))
        self.params = HMMParams(
            pi=np.full(K, 1.0 / K),
            A=self._init_transition(K, rng),
            mu=qs.copy(),
            var=np.full(K, max(obs.var(), 1e-6)),
        )
        prev_ll = -np.inf
        for it in range(self.n_iter):
            log_b = self._log_emission(obs)
            log_alpha, ll = self._forward(log_b)
            log_beta = self._backward(log_b)
            self._m_step(obs, log_alpha, log_beta, log_b)
            self.history_.append(ll)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Most-likely state sequence (Viterbi)."""
        if self.params is None:
            raise RuntimeError("call .fit() first")
        obs = np.asarray(obs, dtype=np.float64).ravel()
        return self._viterbi(obs)

    def posterior(self, obs: np.ndarray) -> np.ndarray:
        """Smoothed P(state_t = k | O), shape (T, K)."""
        if self.params is None:
            raise RuntimeError("call .fit() first")
        obs = np.asarray(obs, dtype=np.float64).ravel()
        log_b = self._log_emission(obs)
        log_alpha, _ = self._forward(log_b)
        log_beta = self._backward(log_b)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        return np.exp(log_gamma)

    def score(self, obs: np.ndarray) -> float:
        if self.params is None:
            raise RuntimeError("call .fit() first")
        obs = np.asarray(obs, dtype=np.float64).ravel()
        log_b = self._log_emission(obs)
        _, ll = self._forward(log_b)
        return float(ll)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _init_transition(K: int, rng: np.random.Generator) -> np.ndarray:
        A = rng.uniform(0.05, 0.15, size=(K, K))
        np.fill_diagonal(A, 0.85)
        A = A / A.sum(axis=1, keepdims=True)
        return A

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        T = obs.size
        K = self.n_states
        log_b = np.empty((T, K))
        for k in range(K):
            log_b[:, k] = _log_gauss_pdf(obs, self.params.mu[k], self.params.var[k])
        return log_b

    def _forward(self, log_b: np.ndarray) -> Tuple[np.ndarray, float]:
        T, K = log_b.shape
        log_A = np.log(self.params.A + 1e-300)
        log_pi = np.log(self.params.pi + 1e-300)
        log_alpha = np.empty((T, K))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            log_alpha[t] = _logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0) + log_b[t]
        ll = float(_logsumexp(log_alpha[-1]))
        return log_alpha, ll

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        T, K = log_b.shape
        log_A = np.log(self.params.A + 1e-300)
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(log_A + log_b[t + 1] + log_beta[t + 1], axis=1)
        return log_beta

    def _m_step(
        self,
        obs: np.ndarray,
        log_alpha: np.ndarray,
        log_beta: np.ndarray,
        log_b: np.ndarray,
    ) -> None:
        T, K = log_alpha.shape
        log_A = np.log(self.params.A + 1e-300)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        # xi_t(i, j) = P(s_t = i, s_{t+1} = j | O)
        log_xi = (
            log_alpha[:-1, :, None]
            + log_A[None, :, :]
            + log_b[1:, None, :]
            + log_beta[1:, None, :]
        )
        log_xi -= _logsumexp(log_xi.reshape(T - 1, -1), axis=1)[:, None, None]
        xi = np.exp(log_xi)

        new_pi = gamma[0] / gamma[0].sum()
        denom_A = gamma[:-1].sum(axis=0)
        new_A = xi.sum(axis=0) / np.where(denom_A[:, None] == 0, 1.0, denom_A[:, None])
        new_A = new_A / new_A.sum(axis=1, keepdims=True)

        denom = gamma.sum(axis=0)
        denom_safe = np.where(denom == 0, 1.0, denom)
        new_mu = (gamma * obs[:, None]).sum(axis=0) / denom_safe
        new_var = (gamma * (obs[:, None] - new_mu) ** 2).sum(axis=0) / denom_safe
        new_var = np.maximum(new_var, 1e-6)

        self.params = HMMParams(pi=new_pi, A=new_A, mu=new_mu, var=new_var)

    def _viterbi(self, obs: np.ndarray) -> np.ndarray:
        T = obs.size
        K = self.n_states
        log_A = np.log(self.params.A + 1e-300)
        log_pi = np.log(self.params.pi + 1e-300)
        log_b = self._log_emission(obs)

        delta = np.empty((T, K))
        psi = np.empty((T, K), dtype=np.int32)
        delta[0] = log_pi + log_b[0]
        for t in range(1, T):
            scores = delta[t - 1, :, None] + log_A
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(K)] + log_b[t]
        path = np.empty(T, dtype=np.int32)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


def hmm_regime_path(obs: np.ndarray, n_states: int = 2, seed: int = 0) -> Tuple[np.ndarray, GaussianHMM]:
    """Convenience wrapper - fit + return Viterbi path + the model.

    Used by the FastAPI endpoint and by the regime panel in the web UI.
    """
    model = GaussianHMM(n_states=n_states, seed=seed).fit(obs)
    return model.predict(obs), model
