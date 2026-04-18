"""
geoalpha_ml
===========

Sequence models for geospatial time series.

The flagship model here is a temporal-attention transformer
originally trained on minute-bar equity returns for a research
project of mine.  Re-targeting it to predict next-revisit
NDVI / radiance turned out to need essentially the same architecture -
just swap the input feature schema and the loss head.  Both problems
are "given an irregularly sampled multivariate sequence, forecast the
next value and quantify the uncertainty"; the geometry is the same
even when the units aren't.

Plus an OpenVINO INT8 quantization pass for edge deployment - same
trick I use for low-latency inference in the trading system, applied
to on-board sat payload inference where every milliwatt counts.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geoalpha-ml")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
