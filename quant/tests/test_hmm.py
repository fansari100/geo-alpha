"""HMM regime-decoding sanity checks on synthetic two-regime data."""

from __future__ import annotations

import numpy as np

from geoalpha_quant.regime import GaussianHMM


def _two_regime_series(n_per: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.55, 0.05, size=n_per)
    b = rng.normal(0.30, 0.10, size=n_per)
    return np.concatenate([a, b]).astype(np.float64)


def test_two_state_hmm_identifies_means():
    obs = _two_regime_series()
    hmm = GaussianHMM(n_states=2, seed=1).fit(obs)
    means = sorted(hmm.params.mu)
    assert abs(means[0] - 0.30) < 0.05
    assert abs(means[1] - 0.55) < 0.05


def test_viterbi_path_majority_correct():
    obs = _two_regime_series()
    hmm = GaussianHMM(n_states=2, seed=1).fit(obs)
    path = hmm.predict(obs)
    # Map state ids to "low / high" by their fitted mean.
    high = int(np.argmax(hmm.params.mu))
    high_in_first_half = int((path[:200] == high).sum())
    high_in_second_half = int((path[200:] == high).sum())
    assert high_in_first_half > high_in_second_half


def test_posterior_sums_to_one():
    obs = _two_regime_series()
    hmm = GaussianHMM(n_states=2, seed=2).fit(obs)
    p = hmm.posterior(obs)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
