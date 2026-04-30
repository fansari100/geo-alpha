"""
Regime detection on geospatial / radiometric time series.

In a vol-regime model on equity returns, the latent state is "calm
vs. crisis" and the observation is daily log-returns.  Swap log-returns
for, say, a per-pixel NDVI series across overpasses, or the integrated
brightness inside a target-of-interest box, and the same Gaussian-HMM
machinery picks out things like pre-monsoon vs. monsoon, vegetation
growth phases, or persistent change in a region of interest.

Two estimators are exposed:

* `GaussianHMM`   - classic Baum-Welch on K Gaussian regimes.  Used
                    when you actually want the latent-state path back.
* `BayesianOnlineChangePoint` - Adams & MacKay 2007 BOCPD.  Cheap,
                    streaming, gives you a run-length distribution
                    over time which is exactly what an analyst wants
                    when watching a feed live.
"""

from .change_point import BayesianOnlineChangePoint, cusum_change_point
from .hmm import GaussianHMM, hmm_regime_path

__all__ = [
    "GaussianHMM",
    "hmm_regime_path",
    "BayesianOnlineChangePoint",
    "cusum_change_point",
]
