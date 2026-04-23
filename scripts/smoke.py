#!/usr/bin/env python
"""End-to-end smoke test: every analytic on a single synthetic scene."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "quant" / "src"))

import numpy as np  # noqa: E402

from geoalpha_quant.factors import SpectralUnmixer  # noqa: E402
from geoalpha_quant.io import (  # noqa: E402
    make_synthetic_cube,
    make_synthetic_revisit_series,
)
from geoalpha_quant.optimization import (  # noqa: E402
    SensorTaskingProblem,
    TargetRequest,
    solve_sensor_tasking,
)
from geoalpha_quant.regime import (  # noqa: E402
    BayesianOnlineChangePoint,
    GaussianHMM,
)
from geoalpha_quant.risk import (  # noqa: E402
    EVTAnomalyDetector,
    propagate_uncertainty,
    summarize_distribution,
)


def main() -> None:
    out: dict = {}

    # 1. Regime detection on a synthetic revisit series.
    series, truth = make_synthetic_revisit_series(n_obs=240, seed=11, regime_shift_at=120)
    hmm = GaussianHMM(n_states=2, seed=1).fit(series)
    bocpd = BayesianOnlineChangePoint(hazard_lambda=120.0)
    cp = bocpd.run(series)
    out["regime"] = {
        "log_likelihood": hmm.score(series),
        "fitted_means": hmm.params.mu.tolist(),
        "bocpd_peak_t": int(cp.argmax()),
        "bocpd_peak_p": float(cp.max()),
        "truth_change_at": int(np.where(np.diff(truth))[0][0]) + 1,
    }

    # 2. Convex sensor tasking.
    targets = [
        TargetRequest(f"t{i:02d}", value=10.0 - i * 0.5, dwell_max=30,
                      priority=("FLASH" if i == 0 else "ROUTINE"))
        for i in range(8)
    ]
    res = solve_sensor_tasking(
        SensorTaskingProblem(targets=targets, total_budget_s=90.0, risk_aversion=0.5)
    )
    out["tasking"] = {
        "solver": res.solver,
        "total_value": res.total_value,
        "assignment": res.as_assignment(SensorTaskingProblem(targets, 90.0)),
    }

    # 3. Monte-Carlo uncertainty propagation.
    toa = np.full((4, 4), 0.18, dtype=np.float32)
    mc = propagate_uncertainty(toa, n_samples=512)
    out["uncertainty"] = summarize_distribution(mc["mean"])

    # 4. EVT-calibrated anomaly threshold.
    rng = np.random.default_rng(0)
    bg = np.abs(rng.normal(0, 1, size=4_000))
    tail = np.abs(rng.standard_t(df=3, size=400) * 4.0)
    scores = np.concatenate([bg, tail])
    det = EVTAnomalyDetector(0.92, 1e-4).fit(scores)
    out["anomaly"] = {
        "threshold": float(det.score_threshold_),
        "xi": det.fit_.xi,
        "sigma": det.fit_.sigma,
        "n_alarms": int(det.predict(scores).sum()),
    }

    # 5. Spectral unmixing.
    sc = make_synthetic_cube(height=48, width=48, seed=3)
    unm = SpectralUnmixer(n_endmembers=4, l1=0.0, seed=3).fit_predict(sc.cube)
    out["unmixing"] = {
        "rmse": float(unm["rmse"]),
        "explained_variance_ratio": float(unm["explained_variance_ratio"]),
    }

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
