# geo-alpha

**Quantitative methods for geospatial intelligence.**

> A small platform that takes the toolbox I built up working on
> volatility regimes, factor models, convex portfolio optimization,
> Monte-Carlo VaR and walk-forward backtesting in finance, and
> re-points it at problems in geospatial / multi-sensor analytics:
> regime detection on satellite revisit series, sensor tasking,
> uncertainty propagation through atmospheric correction, and
> EVT-calibrated anomaly detection.

The motivating observation, which I keep finding fresh evidence for
the deeper I get, is that an enormous fraction of the math you reach
for in quantitative finance shows up *unchanged* in geospatial signal
processing.  Same Kalman / HMM machinery, same convex programs, same
EVT-style tail modelling, just different units on the axes.

```text
              +----------------------------+
              |   web (Next.js / React)    |
              |   leaflet + recharts       |
              +-------------+--------------+
                            |
                 REST + WS  |
                            v
   +-------------------+    |    +--------------------------+
   |  service (Java)   |<-->+--->|  api (Python / FastAPI)  |
   |  Spring Boot      |    REST |  geoalpha-quant analytics|
   |  scheduler + REST |         |  geoalpha-ml inference   |
   +---------+---------+         +-----------+--------------+
             |                               |
             |                               v
             |                +--------------+--------------+
             |                |  engine (C++)               |
             +--------------->|  Kalman / UKF state-space   |
                              |  shared lib + pybind11      |
                              +-----------------------------+
```

## What's where

| Path        | Language     | What lives there                                                     |
|-------------|--------------|----------------------------------------------------------------------|
| `quant/`    | Python       | Core analytics: HMM regime detect, BOCPD, CVXPY tasking, MC VaR, EVT, factor unmixing, walk-forward backtest |
| `engine/`   | C++          | Latency-sensitive state-space estimators (Kalman / UKF) with pybind11 bindings |
| `ml/`       | Python+Torch | Temporal-attention sequence model + OpenVINO INT8 export for edge   |
| `api/`      | Python       | FastAPI gateway exposing every analytic over REST + a WS demo       |
| `service/`  | Java 21      | Spring Boot mission scheduler + orchestrator                        |
| `web/`      | TypeScript   | Next.js operator console with map view, regime + tasking + risk pages |
| `infra/`    | YAML         | Compose, Kubernetes manifests, Helm chart, Prometheus config        |
| `notebooks/`| Jupyter      | Scratch notebooks I used to validate the modules end-to-end         |

## Why I built it

I have a quant-finance background (Houlihan Lokey valuation, AIG
quantitative analytics, Compak portfolio research, plus a couple of
trading-system side projects), and I noticed a few years back that
much of what I do on the financial side is just labelled differently
on the geospatial / signal-processing side:

- **HMM regime detection** on equity vol → on satellite revisit series.
- **Convex long-only portfolio optimization** → sensor tasking under
  per-priority caps.
- **Monte-Carlo VaR** sweeping yield-curve and credit-spread scenarios →
  MC uncertainty propagation through atmospheric correction.
- **EVT / GPD tail risk** for one-in-N-year claim severity → for
  one-in-N-pixel anomaly thresholds.
- **Brinson-Fachler attribution** of portfolio return → decomposition
  of TOA radiance into atmosphere / surface / sensor contributions.
- **Walk-forward optimization** of trading parameters → walk-forward
  threshold sweeps for any score-based detector.
- **Kalman filtering** on state-space term-structure models → multi-
  target tracking from sensor reports.
- **OpenVINO INT8 quantization** of trading models → on-board edge
  inference under tight power / latency budgets.

Wanting to see how far that mapping holds up in practice, I built this
as a single self-contained platform that pulls all of it together.

## Quick start

```bash
# Run the unit tests (Python).
pip install -e quant[dev] && pytest -v quant
pip install -e api[dev]   && pytest -v api

# Or run the whole stack with docker compose.
docker compose up --build
open http://localhost:5173        # operator console
open http://localhost:8000/docs   # FastAPI swagger
open http://localhost:8080/actuator/health  # service health
open http://localhost:9090        # prometheus
```

## Build the C++ engine

```bash
cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build --parallel
ctest --test-dir engine/build --output-on-failure
./engine/build/geoalpha_bench --steps 100000
```

Sample output (Ryzen 7 7840U, no GPU):

```
Linear KF (4x2):  ~ 800 ns/step
UKF      (4x2):  ~ 7.5 us/step
```

## Deploy to Kubernetes

```bash
docker build -t geoalpha/api:latest     api
docker build -t geoalpha/service:latest service
docker build -t geoalpha/web:latest     web
kubectl apply -k infra/k8s/
# or via Helm
helm install ga infra/helm/geo-alpha
```

The API deployment ships an HPA that scales out to 8 replicas under
CPU pressure.  Helm values are factored cleanly so flipping
`api.image.tag` is a one-line change in `values.yaml`.

## Status

This is an active scratchpad and not production code.  Things are
intentionally small and dependency-light so the demos boot in a few
seconds and the algorithms are visible without an Eigen / GDAL
indirection layer in the way.

If you spot something interesting (or wrong) please open an issue —
I'm always looking for excuses to dig deeper into the
finance ↔ remote-sensing crossover.

—Ricky
