# design notes

These are the notes I scribbled on a whiteboard before I started
writing code, then cleaned up after enough iterations that the shapes
stopped changing.  Treat them as guard-rails, not specs.

## the observation that started it

Most of the math I use day-to-day on the quant side maps almost
unchanged onto something operationally useful in geospatial / signal
processing.  The mapping I used to scope this repo:

| quant tool                       | geospatial counterpart                                  |
|----------------------------------|----------------------------------------------------------|
| HMM vol-regime detection         | regime detect on per-pixel / per-AOI radiometric series  |
| Bayesian online change-point     | streaming break detect on a satellite revisit feed       |
| convex long-only portfolio QP    | sensor tasking (dwell-time allocation)                   |
| Monte-Carlo VaR                  | uncertainty propagation through atmospheric correction   |
| EVT / GPD on tail of severities  | calibrated FAR threshold for spectral anomaly scoring    |
| Brinson-Fachler attribution      | decompose TOA radiance into atmosphere / surface / sensor|
| Fama-French factor exposures     | linear / sparse spectral unmixing                        |
| walk-forward optimization        | walk-forward sweep over detector thresholds              |
| Kalman filtering on yield curves | multi-target tracking from sensor reports                |
| OpenVINO INT8 quantization       | on-board sat / edge node inference                       |

## architectural choices and why

- **Two Python packages, not one.**  `geoalpha-quant` is a pure-Python
  library you can pip-install into a notebook with no FastAPI, no
  Spring, no JS.  `geoalpha-api` depends on it and adds the gateway.
  Keeps the analytics importable in isolation, which is exactly what I
  want when I'm doing scratch work in `notebooks/`.
- **C++ in a separate library, not embedded in Python.**  Same reason -
  I want the engine to be usable from Java (via JNI), Rust (via FFI),
  or as a standalone CLI without dragging the Python ABI into every
  build.  pybind11 is the friendliest path back into Python and lives
  in `engine/bindings/`.
- **Java for the orchestrator instead of more Python.**  Defense and
  finance customers run a lot of Spring; pretending otherwise to keep
  the stack monoglot would be a strange choice.  The orchestrator is
  small enough that the JVM tax is fine, and it gets us actuator
  endpoints, micrometer metrics, and reactive HTTP for free.
- **Next.js, not vanilla React + Vite.**  I want SSR + file-based
  routing for the dashboard pages (`/`, `/regimes`, `/tasking`,
  `/risk`).  The proxy rewrites in `next.config.mjs` mean the React
  components don't need to know about CORS or the two backend hosts.

## things I deliberately kept out

- **A real radiative-transfer model.**  The `AtmosphericChain` in
  `quant/risk/mc_uncertainty.py` is a stylised model.  When I want
  6S / MODTRAN I'll wire it up, but the MC harness is the bit that
  needed to be right and that's what's covered by tests.
- **Eigen / GDAL.**  Both are great.  Both also turn 5-second compiles
  into 60-second compiles and bury the actual algebra under template
  metaprogramming.  Not worth it for a learning / portfolio repo.
- **Real sensor IO.**  Every demo runs on synthetic fixtures defined
  in `quant/io/synthetic.py`.  Reproducibility > realism here.
- **A real model-server for the ML piece.**  ONNX export + OpenVINO
  INT8 are wired up in `ml/edge`, but I don't ship a serving binary -
  the artifacts go into `artifacts/edge/` and you can pick them up
  from a real Triton / OpenVINO server in production.

## what I'd do next

1. Replace the in-memory `MissionRepository` with JPA + PostgreSQL.
2. Pull a real Sentinel-2 NDVI extract for a region with ground-truth
   deforestation polygons and rerun the BOCPD + HMM pipeline.
3. Train the temporal-attention forecaster on that real series and
   compare to the synthetic-data benchmark.
4. Wire mTLS / SPIFFE between the three services.
5. Add a particle filter to `engine/` for non-Gaussian target tracking.
