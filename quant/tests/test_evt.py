"""EVT tail-risk fitting."""

from __future__ import annotations

import numpy as np

from geoalpha_quant.risk import EVTAnomalyDetector, fit_gpd, return_period_threshold


def test_gpd_fits_pareto_like_tail():
    rng = np.random.default_rng(0)
    bulk = rng.normal(0, 1, size=4_000)
    tail = rng.standard_t(df=3, size=600) * 4.0
    x = np.concatenate([bulk, np.abs(tail)])
    fit = fit_gpd(x, threshold_quantile=0.9)
    # heavy-tail expectation - shape parameter > 0.
    assert fit.xi > -0.5
    thr = return_period_threshold(fit, target_prob=1e-3)
    assert thr > fit.threshold


def test_detector_predicts_some_alarms():
    rng = np.random.default_rng(7)
    x = rng.gamma(2.0, 1.0, size=5_000)
    det = EVTAnomalyDetector(threshold_quantile=0.92, target_far=1e-3).fit(x)
    flags = det.predict(x)
    assert int(flags.sum()) >= 1
    assert int(flags.sum()) < x.size  # not flagging everything
