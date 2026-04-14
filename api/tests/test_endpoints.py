"""HTTP integration tests over the FastAPI gateway."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from geoalpha_api.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_health():
    with _client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"


def test_regime_endpoint():
    rng = np.random.default_rng(0)
    series = np.concatenate(
        [rng.normal(0.5, 0.05, 80), rng.normal(0.2, 0.1, 80)]
    ).tolist()
    with _client() as c:
        r = c.post("/quant/regime", json={"series": series, "n_states": 2})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["states"]) == len(series)
        assert len(body["means"]) == 2


def test_tasking_endpoint():
    body = {
        "targets": [
            {"name": "alpha", "value": 10.0, "dwell_max": 30, "priority": "ROUTINE"},
            {"name": "flash", "value": 4.0, "dwell_max": 30, "priority": "FLASH"},
        ],
        "total_budget_s": 60.0,
    }
    with _client() as c:
        r = c.post("/quant/tasking", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert "assignment" in out
        assert "flash" in out["assignment"]


def test_uncertainty_endpoint():
    with _client() as c:
        r = c.post("/quant/uncertainty", json={"toa": [0.18, 0.21, 0.16], "n_samples": 64})
        assert r.status_code == 200, r.text
        s = r.json()["summary"]
        for k in ("mean", "p50", "cvar95"):
            assert k in s


def test_anomaly_endpoint():
    rng = np.random.default_rng(0)
    scores = rng.gamma(2.0, 1.0, size=2000).tolist()
    with _client() as c:
        r = c.post("/quant/anomaly", json={"scores": scores, "threshold_quantile": 0.92})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["n_alarms"] >= 0
        assert "threshold" in out
