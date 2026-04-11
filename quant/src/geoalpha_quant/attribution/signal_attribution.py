"""
Signal attribution - decomposing observed TOA radiance into the
contributions from atmosphere, surface, and sensor.

Same logic as Brinson-Fachler attribution on a portfolio:

    Brinson-Fachler:
        portfolio_return = allocation_effect + selection_effect + interaction
    Signal attribution:
        observed_radiance = atmospheric_contribution
                          + surface_contribution
                          + sensor_contribution + interaction

The interaction terms are what make both decompositions a little
fiddly - in Brinson it's the cross of allocation x selection; here
it's the cross of atmospheric path radiance modulated by surface
reflectance.

Useful for ground-truth-free QC: a sudden swing in the atmospheric
component while surface is steady is the satellite analogue of an
allocation-driven excess return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class SignalAttribution:
    """Per-pixel decomposition of an observation."""

    atmospheric: np.ndarray
    surface: np.ndarray
    sensor: np.ndarray
    interaction: np.ndarray
    total: np.ndarray
    summary: Dict[str, float] = field(default_factory=dict)


def decompose_observation(
    toa: np.ndarray,
    surface_ref: np.ndarray,
    atm_path: np.ndarray,
    transmittance: np.ndarray,
    sensor_bias: np.ndarray,
) -> SignalAttribution:
    """Brinson-style decomposition of a TOA observation.

    Model:
        toa = atm_path + transmittance * surface_ref + sensor_bias

    so we attribute total = atmospheric + surface + sensor + interaction
    where:
        atmospheric  = atm_path                                  (drift)
        surface      = surface_ref                               (signal)
        sensor       = sensor_bias                               (cal)
        interaction  = (transmittance - 1) * surface_ref         (cross)
    """
    atm = atm_path
    surf = surface_ref
    sensor = sensor_bias
    inter = (transmittance - 1.0) * surface_ref
    total = atm + surf + sensor + inter

    def _f(x): return float(np.nan_to_num(np.mean(x), nan=0.0))
    summary = {
        "mean_total": _f(total),
        "mean_atm": _f(atm),
        "mean_surf": _f(surf),
        "mean_sensor": _f(sensor),
        "mean_interaction": _f(inter),
        "atm_share": _f(np.abs(atm)) / max(_f(np.abs(total)), 1e-9),
        "surf_share": _f(np.abs(surf)) / max(_f(np.abs(total)), 1e-9),
        "sensor_share": _f(np.abs(sensor)) / max(_f(np.abs(total)), 1e-9),
    }
    return SignalAttribution(
        atmospheric=atm,
        surface=surf,
        sensor=sensor,
        interaction=inter,
        total=total,
        summary=summary,
    )
