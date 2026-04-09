"""Monte Carlo uncertainty propagation."""

from __future__ import annotations

import numpy as np

from geoalpha_quant.risk import (
    AtmosphericChain,
    propagate_uncertainty,
    summarize_distribution,
)


def test_propagation_returns_full_summary():
    toa = np.full((4, 4), 0.18, dtype=np.float32)
    res = propagate_uncertainty(toa, n_samples=128, seed=1)
    assert res["outputs"].shape == (128, 4, 4)
    for key in ("mean", "std", "p05", "p50", "p95", "cvar95"):
        assert res[key].shape == toa.shape
    # Quantile ordering invariant.
    assert (res["p05"] <= res["p50"]).all()
    assert (res["p50"] <= res["p95"]).all()


def test_summary_stats_keys():
    arr = np.linspace(0, 1, 256)
    s = summarize_distribution(arr)
    for k in ("mean", "std", "p05", "p50", "p95", "var95", "cvar95", "count"):
        assert k in s


def test_chain_zero_atmosphere_returns_input_scaled_by_geometry():
    toa = np.array([[0.5]], dtype=np.float32)
    chain = AtmosphericChain()
    out = chain.forward(toa, sun_zenith_deg=0.0, aod_550=1e-6, cwv_g_cm2=1e-6)
    # Sun overhead, no attenuation - surface ~ TOA.
    assert abs(out[0, 0] - 0.5) < 0.05
