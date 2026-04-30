"""
Reproducible synthetic data fixtures.

I want every demo, notebook and test to run with no external data so
nothing in this repo is gated on having a Sentinel-2 quota or a
GeoTIFF lying around on disk.  These generators reproduce the gross
spectral and temporal structure you'd see in a real ISR feed:

    make_synthetic_cube           - (B, H, W) hyperspectral cube with
                                    vegetation, water and built-up
                                    spectra at known locations.
    make_synthetic_revisit_series - per-pixel time series of NDVI
                                    across N satellite revisits,
                                    with a planted regime shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticCube:
    cube: np.ndarray                  # (B, H, W) reflectance
    truth_classes: np.ndarray         # (H, W) integer class label
    band_centres_um: np.ndarray       # (B,) wavelengths
    endmember_names: tuple[str, ...]
    truth_anomaly_mask: np.ndarray    # (H, W) bool


# Crude library spectra (vegetation / water / urban / soil / anomaly target)
_LIBRARY = {
    "veg":   np.array([0.05, 0.07, 0.09, 0.06, 0.45, 0.30, 0.18]),
    "water": np.array([0.08, 0.07, 0.05, 0.02, 0.01, 0.01, 0.005]),
    "urban": np.array([0.12, 0.13, 0.15, 0.18, 0.22, 0.28, 0.20]),
    "soil":  np.array([0.10, 0.12, 0.14, 0.18, 0.22, 0.30, 0.25]),
    "anom":  np.array([0.40, 0.45, 0.50, 0.55, 0.65, 0.70, 0.55]),
}


def make_synthetic_cube(
    height: int = 96,
    width: int = 96,
    seed: int = 17,
    n_anomalies: int = 6,
) -> SyntheticCube:
    rng = np.random.default_rng(seed)
    B = next(iter(_LIBRARY.values())).size
    cube = np.zeros((B, height, width), dtype=np.float32)
    classes = np.zeros((height, width), dtype=np.int32)

    # Background tessellation: random Voronoi seeds for each material.
    seeds = rng.uniform(0, 1, size=(4, 2))
    seeds[:, 0] *= height
    seeds[:, 1] *= width
    materials = ["soil", "veg", "urban", "water"]

    yy, xx = np.indices((height, width))
    for y in range(height):
        for x in range(width):
            d = ((seeds[:, 0] - y) ** 2 + (seeds[:, 1] - x) ** 2)
            cls = int(np.argmin(d))
            classes[y, x] = cls
            cube[:, y, x] = _LIBRARY[materials[cls]] * rng.uniform(0.92, 1.08, size=B)

    # Planted anomalies (rare-spectra pixels).
    anomaly_mask = np.zeros((height, width), dtype=bool)
    for _ in range(n_anomalies):
        ay = int(rng.integers(2, height - 2))
        ax = int(rng.integers(2, width - 2))
        radius = int(rng.integers(1, 3))
        y0, y1 = max(0, ay - radius), min(height, ay + radius + 1)
        x0, x1 = max(0, ax - radius), min(width, ax + radius + 1)
        cube[:, y0:y1, x0:x1] = _LIBRARY["anom"][:, None, None] * rng.uniform(0.95, 1.05)
        anomaly_mask[y0:y1, x0:x1] = True

    # Photon noise.
    cube = cube + rng.normal(0, 0.005, size=cube.shape).astype(np.float32)
    cube = np.clip(cube, 0.0, 1.0)

    return SyntheticCube(
        cube=cube,
        truth_classes=classes,
        band_centres_um=np.array([0.45, 0.55, 0.65, 0.85, 1.05, 1.65, 2.20]),
        endmember_names=tuple(materials),
        truth_anomaly_mask=anomaly_mask,
    )


def make_synthetic_revisit_series(
    n_obs: int = 240,
    seed: int = 3,
    regime_shift_at: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel NDVI-like series across satellite revisits.

    Inserts a regime shift (e.g. drought, deforestation, irrigation
    onset) so the HMM and BOCPD detectors have something to find.
    """
    rng = np.random.default_rng(seed)
    if regime_shift_at is None:
        regime_shift_at = n_obs // 2
    seasonal = 0.15 * np.sin(np.linspace(0, 6 * np.pi, n_obs))
    base = np.where(np.arange(n_obs) < regime_shift_at, 0.55, 0.30)
    series = base + seasonal + rng.normal(0, 0.03, size=n_obs)
    truth = (np.arange(n_obs) >= regime_shift_at).astype(np.int32)
    return series.astype(np.float32), truth
