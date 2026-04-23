# quant ↔ geospatial mapping

Quick reference table for the equivalences this repo exploits, with
file pointers so you can jump straight to the implementation.

| Quant technique                   | Geospatial counterpart                          | File |
|-----------------------------------|--------------------------------------------------|------|
| Gaussian HMM (Baum-Welch)         | regime detection on per-AOI radiometric series  | `quant/src/geoalpha_quant/regime/hmm.py` |
| Bayesian online change-point      | streaming break detection                       | `quant/src/geoalpha_quant/regime/change_point.py` |
| CUSUM                             | edge-side change-point detector                 | `quant/src/geoalpha_quant/regime/change_point.py` |
| Long-only constrained portfolio QP | sensor tasking / dwell allocation              | `quant/src/geoalpha_quant/optimization/sensor_tasking.py` |
| MC scenario sweep for VaR/CVaR    | MC propagation through atmospheric chain        | `quant/src/geoalpha_quant/risk/mc_uncertainty.py` |
| EVT / GPD on tail of severities   | calibrated FAR for spectral anomaly             | `quant/src/geoalpha_quant/risk/evt_anomaly.py` |
| Brinson-Fachler attribution       | atmosphere/surface/sensor decomposition         | `quant/src/geoalpha_quant/attribution/signal_attribution.py` |
| PCA-based factor decomposition    | hyperspectral PCA / unmixing                    | `quant/src/geoalpha_quant/factors/spectral_factors.py` |
| Sparse / lasso loadings           | non-negative + L1 unmixing                      | `quant/src/geoalpha_quant/factors/spectral_factors.py` |
| Walk-forward parameter sweep      | walk-forward threshold sweep                    | `quant/src/geoalpha_quant/backtest/walk_forward.py` |
| Sharpe / Sortino style ratios     | precision / recall / F1 / FAR / PD              | `quant/src/geoalpha_quant/backtest/metrics.py` |
| Information coefficient (IC)      | rank correlation of detector confidence vs truth| `quant/src/geoalpha_quant/backtest/metrics.py` |
| Kalman filter (linear)            | constant-velocity target tracking               | `engine/src/kalman.cpp` |
| Unscented Kalman Filter           | nonlinear target dynamics (CT / ballistic)      | `engine/src/ukf.cpp` |
| Temporal attention encoder        | next-revisit forecast on satellite series       | `ml/src/geoalpha_ml/models/temporal_attention.py` |
| OpenVINO INT8 quantization        | INT8 inference for on-board / edge node         | `ml/src/geoalpha_ml/edge/openvino_export.py` |
| Heteroscedastic Gaussian NLL loss | predictive variance for uncertainty-aware fcst  | `ml/src/geoalpha_ml/models/temporal_attention.py` |
