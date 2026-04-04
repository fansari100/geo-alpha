"""
geoalpha_quant
==============

Quantitative-finance methods adapted for geospatial / multi-sensor
time series.  Most of what's in here started life as code I wrote for
volatility regime detection, factor decomposition, tail-risk modeling
and convex portfolio optimization on financial data; the same math
turns out to map almost 1:1 onto the kinds of problems you face when
you're trying to extract signal from streams of pixels and sensor
observations.

Modules
-------
regime         - HMM / Bayesian online change-point detection on
                 spectral or radiometric time series.
optimization   - Sensor tasking via convex optimization (CVXPY).
risk           - Monte-Carlo uncertainty propagation, EVT tail risk
                 for spectral anomaly detection.
factors        - Linear / sparse spectral unmixing, the Fama-French
                 way (just with endmembers instead of value/momentum).
attribution    - Brinson-Fachler style decomposition of an observed
                 signal into atmospheric, surface and sensor pieces.
backtest       - Walk-forward evaluation harness for any detector.
io             - Lightweight I/O for the synthetic geospatial cube
                 fixtures used in the demos and tests.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geoalpha-quant")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
