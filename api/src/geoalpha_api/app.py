"""
FastAPI gateway.

The endpoints map roughly 1:1 to the modules in geoalpha_quant; the
goal is for the React dashboard (and any external client) to be able
to drive any analytic in the platform with a single HTTP call.

Endpoints
---------
GET  /health                 - liveness + version probe.
POST /quant/regime           - HMM regime decoding on a 1-D series.
POST /quant/change_point     - BOCPD streaming change-point posterior.
POST /quant/tasking          - solve a sensor-tasking convex program.
POST /quant/uncertainty      - MC uncertainty propagation.
POST /quant/anomaly          - EVT-calibrated anomaly threshold + flags.
WS   /stream/regimes         - live regime posterior stream (demo).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from geoalpha_quant import __version__ as quant_version
from geoalpha_quant.optimization import (
    SensorTaskingProblem,
    TargetRequest,
    solve_sensor_tasking,
)
from geoalpha_quant.regime import (
    BayesianOnlineChangePoint,
    GaussianHMM,
)
from geoalpha_quant.risk import (
    EVTAnomalyDetector,
    propagate_uncertainty,
    summarize_distribution,
)

from .schemas import (
    AnomalyRequest,
    AnomalyResponse,
    ChangePointRequest,
    ChangePointResponse,
    HealthResponse,
    RegimeRequest,
    RegimeResponse,
    TaskingRequest,
    TaskingResponse,
    UncertaintyRequest,
    UncertaintyResponse,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="geo-alpha API",
        description="Quantitative methods for geospatial intelligence.",
        version=quant_version,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        return HealthResponse(version=quant_version)

    @app.post("/quant/regime", response_model=RegimeResponse, tags=["regime"])
    async def regime(req: RegimeRequest) -> RegimeResponse:
        obs = np.asarray(req.series, dtype=np.float64)
        try:
            hmm = GaussianHMM(n_states=req.n_states).fit(obs)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        states = hmm.predict(obs).tolist()
        posterior = hmm.posterior(obs).tolist()
        return RegimeResponse(
            states=states,
            posterior=posterior,
            log_likelihood=hmm.score(obs),
            means=hmm.params.mu.tolist(),
            variances=hmm.params.var.tolist(),
        )

    @app.post("/quant/change_point", response_model=ChangePointResponse, tags=["regime"])
    async def change_point(req: ChangePointRequest) -> ChangePointResponse:
        bocpd = BayesianOnlineChangePoint(hazard_lambda=req.hazard_lambda)
        cp = bocpd.run(np.asarray(req.series, dtype=np.float64))
        return ChangePointResponse(cp_prob=cp.tolist())

    @app.post("/quant/tasking", response_model=TaskingResponse, tags=["tasking"])
    async def tasking(req: TaskingRequest) -> TaskingResponse:
        problem = SensorTaskingProblem(
            targets=[
                TargetRequest(t.name, t.value, t.dwell_min, t.dwell_max, t.priority)
                for t in req.targets
            ],
            total_budget_s=req.total_budget_s,
            risk_aversion=req.risk_aversion,
        )
        result = solve_sensor_tasking(problem)
        return TaskingResponse(
            assignment=result.as_assignment(problem),
            total_value=result.total_value,
            solver=result.solver,
        )

    @app.post("/quant/uncertainty", response_model=UncertaintyResponse, tags=["risk"])
    async def uncertainty(req: UncertaintyRequest) -> UncertaintyResponse:
        toa = np.asarray(req.toa, dtype=np.float32)
        out = propagate_uncertainty(
            toa,
            n_samples=req.n_samples,
            sun_zenith_deg=(req.sun_zenith_mean, req.sun_zenith_std),
        )
        return UncertaintyResponse(summary=summarize_distribution(out["mean"]))

    @app.post("/quant/anomaly", response_model=AnomalyResponse, tags=["risk"])
    async def anomaly(req: AnomalyRequest) -> AnomalyResponse:
        scores = np.asarray(req.scores, dtype=np.float64)
        det = EVTAnomalyDetector(req.threshold_quantile, req.target_far).fit(scores)
        flags = det.predict(scores)
        return AnomalyResponse(
            threshold=det.score_threshold_,
            xi=det.fit_.xi,
            sigma=det.fit_.sigma,
            n_alarms=int(flags.sum()),
        )

    @app.websocket("/stream/regimes")
    async def stream_regimes(ws: WebSocket) -> None:
        """Demo: stream a synthetic series + live BOCPD posterior."""
        await ws.accept()
        bocpd = BayesianOnlineChangePoint(hazard_lambda=120.0)
        try:
            for t in range(2_000):
                v = float(np.sin(t / 25.0) + (np.random.randn() * 0.05))
                if t > 1_000:
                    v += 0.5
                p = bocpd.update(v)["cp_prob"]
                await ws.send_json({"t": t, "value": v, "cp_prob": p})
                await asyncio.sleep(0.02)
        except WebSocketDisconnect:
            return

    return app
