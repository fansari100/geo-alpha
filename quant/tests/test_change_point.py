"""BOCPD + CUSUM change-point sanity checks."""

from __future__ import annotations

from geoalpha_quant.io import make_synthetic_revisit_series
from geoalpha_quant.regime import BayesianOnlineChangePoint, cusum_change_point


def test_bocpd_spikes_near_planted_change():
    series, _ = make_synthetic_revisit_series(n_obs=300, seed=11, regime_shift_at=150)
    bocpd = BayesianOnlineChangePoint(hazard_lambda=80.0)
    cp = bocpd.run(series)
    # Mass concentrated around the planted change point (give 50-step window).
    window = cp[140:200]
    assert window.max() > cp[:120].max()


def test_cusum_detects_change():
    series, _ = make_synthetic_revisit_series(n_obs=400, seed=4, regime_shift_at=200)
    detections = cusum_change_point(series, drift=0.02, threshold=2.0)
    assert any(180 <= d <= 250 for d in detections)
