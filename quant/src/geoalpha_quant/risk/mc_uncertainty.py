"""
Monte Carlo uncertainty propagation through the atmospheric correction
chain.

This is a direct port of the MC valuation harness I built at Houlihan
- there I was sweeping yield-curve and credit-spread scenarios to get
a posterior on portfolio NAV; here I sweep solar-zenith, AOD and
column-water-vapour to get a posterior on surface reflectance.

The mathematical contract is identical: pull N parameter samples from
a joint distribution, run the deterministic forward model on each,
collect the empirical distribution of the output, report tail
statistics (VaR/CVaR over there, P5/P95/CVaR over here).

The forward model is intentionally stylised - I wanted the MC
machinery to be the focus, not radiative transfer.  Swap
`AtmosphericChain.forward` for a 6S / MODTRAN call and the rest of
this file works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


# --------------------------------------------------------------------- #
# A small, deterministic atmospheric forward model (lookup-driven).
# --------------------------------------------------------------------- #

@dataclass
class AtmosphericChain:
    """Stylised 1-band atmospheric transmittance / path-radiance model.

    Parameters are the operational triplet you'd pull from the MTL
    metadata + an AOD product:

        sun_zenith_deg : 0..89
        aod_550        : > 0    (aerosol optical depth at 550 nm)
        cwv_g_cm2      : > 0    (column water vapour, g/cm^2)

    Returns surface reflectance given a measured TOA reflectance.
    """

    sensor_gain: float = 1.0
    band_centre_um: float = 0.65  # default RED

    def forward(
        self,
        toa: np.ndarray,
        sun_zenith_deg: float,
        aod_550: float,
        cwv_g_cm2: float,
    ) -> np.ndarray:
        # cos(theta) - solar geometry attenuation.
        mu0 = max(np.cos(np.deg2rad(sun_zenith_deg)), 0.05)

        # Aerosol transmittance: Beer-Lambert with a wavelength scaling.
        ang_exp = -1.0  # Angstrom exponent
        tau_a = aod_550 * (self.band_centre_um / 0.55) ** ang_exp
        T_a = np.exp(-tau_a / mu0)

        # Water vapour - very crude, realistic enough to differentiate
        # bands when CWV moves ~0..6 g/cm2.
        T_h2o = np.exp(-0.05 * cwv_g_cm2 / mu0)

        # Lumped path-radiance term, scales with optical depth.
        L_p = 0.04 * (1.0 - np.exp(-tau_a))

        # Invert: rho_surface = (rho_toa / mu0 - L_p) / (T_a * T_h2o * sensor_gain)
        rho_surf = (toa / mu0 - L_p) / np.maximum(T_a * T_h2o * self.sensor_gain, 1e-6)
        return np.clip(rho_surf, 0.0, 1.5)


# --------------------------------------------------------------------- #
# MC driver.
# --------------------------------------------------------------------- #

@dataclass
class _ParamPriors:
    sun_zenith_deg: Tuple[float, float]   # (mean, std)
    aod_550: Tuple[float, float]          # log-normal: (mu, sigma)
    cwv_g_cm2: Tuple[float, float]        # gamma: (shape, scale)


def _draw(priors: _ParamPriors, n: int, rng: np.random.Generator) -> np.ndarray:
    sz = rng.normal(priors.sun_zenith_deg[0], priors.sun_zenith_deg[1], size=n)
    aod = rng.lognormal(priors.aod_550[0], priors.aod_550[1], size=n)
    cwv = rng.gamma(priors.cwv_g_cm2[0], priors.cwv_g_cm2[1], size=n)
    return np.column_stack([np.clip(sz, 0.0, 89.0), aod, cwv])


def propagate_uncertainty(
    toa: np.ndarray,
    chain: AtmosphericChain | None = None,
    n_samples: int = 4096,
    sun_zenith_deg: Tuple[float, float] = (35.0, 1.5),
    aod_550_lognorm: Tuple[float, float] = (np.log(0.15), 0.4),
    cwv_g_cm2_gamma: Tuple[float, float] = (2.0, 0.6),
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    """Push N parameter samples through the atmospheric chain.

    Returns
    -------
    dict with keys:
        samples : (n_samples,) atmospheric parameter draws
        outputs : (n_samples, ...)  surface reflectance per draw
        mean    : posterior mean
        std     : posterior std
        p05/p50/p95
        cvar95  : conditional value at risk at the 95% tail
    """
    chain = chain or AtmosphericChain()
    rng = np.random.default_rng(seed)
    priors = _ParamPriors(sun_zenith_deg, aod_550_lognorm, cwv_g_cm2_gamma)
    draws = _draw(priors, n_samples, rng)

    out = np.empty((n_samples,) + np.shape(toa), dtype=np.float32)
    for i, (sz, aod, cwv) in enumerate(draws):
        out[i] = chain.forward(toa, sz, aod, cwv)

    return {
        "samples": draws,
        "outputs": out,
        "mean": out.mean(axis=0),
        "std": out.std(axis=0),
        "p05": np.percentile(out, 5, axis=0),
        "p50": np.percentile(out, 50, axis=0),
        "p95": np.percentile(out, 95, axis=0),
        "cvar95": _cvar(out, 0.95),
    }


def _cvar(samples: np.ndarray, alpha: float) -> np.ndarray:
    """Conditional VaR at confidence alpha along axis 0."""
    var = np.percentile(samples, 100 * alpha, axis=0)
    mask = samples >= var
    safe = np.where(mask, samples, np.nan)
    return np.nanmean(safe, axis=0)


def summarize_distribution(arr: np.ndarray) -> Dict[str, float]:
    """Compact summary used by the API + dashboard."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "var95": float(np.percentile(finite, 95)),
        "cvar95": float(finite[finite >= np.percentile(finite, 95)].mean()),
    }
