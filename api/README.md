# geoalpha-api (FastAPI)

REST + WebSocket gateway over the `geoalpha_quant` analytics modules.
Every analytic exposed by the platform is reachable from a single
HTTP call so the operator console (`web/`) and any external client
can drive the stack without touching Python directly.

## endpoints

| Method | Path                  | Backed by                                            |
|--------|-----------------------|------------------------------------------------------|
| GET    | `/health`             | liveness + version probe                             |
| POST   | `/quant/regime`       | `geoalpha_quant.regime.GaussianHMM`                  |
| POST   | `/quant/change_point` | `geoalpha_quant.regime.BayesianOnlineChangePoint`    |
| POST   | `/quant/tasking`      | `geoalpha_quant.optimization.solve_sensor_tasking`   |
| POST   | `/quant/uncertainty`  | `geoalpha_quant.risk.propagate_uncertainty`          |
| POST   | `/quant/anomaly`      | `geoalpha_quant.risk.EVTAnomalyDetector`             |
| WS     | `/stream/regimes`     | live BOCPD posterior over a synthetic series (demo)  |

Schemas in `src/geoalpha_api/schemas.py` (Pydantic v2). OpenAPI is
auto-generated; visit `/docs` once the server is running.

## install

```bash
# from repo root - quant must be installed first because api depends on it.
pip install -e quant
pip install -e api[dev]
```

## run

```bash
uvicorn geoalpha_api.app:create_app --factory --host 0.0.0.0 --port 8000
# or
python -m geoalpha_api
```

## test

```bash
pytest -v
```

## docker

```bash
docker build -t geoalpha/api -f api/Dockerfile .
docker run --rm -p 8000:8000 geoalpha/api
```
