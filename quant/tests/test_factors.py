"""Spectral factor / unmixing checks."""

from __future__ import annotations

import numpy as np

from geoalpha_quant.factors import SpectralUnmixer, fit_pca_factors
from geoalpha_quant.io import make_synthetic_cube


def test_pca_returns_correct_shapes():
    cube = make_synthetic_cube(height=32, width=32, seed=0).cube
    factors, loadings, explained = fit_pca_factors(cube, k=3)
    assert factors.shape == (cube.shape[0], 3)
    assert loadings.shape == (3, 32, 32)
    assert explained.shape == (3,)
    assert (explained >= 0).all()


def test_unmixer_reconstructs_well():
    sc = make_synthetic_cube(height=32, width=32, seed=2)
    out = SpectralUnmixer(n_endmembers=4, l1=0.0, seed=2).fit_predict(sc.cube)
    assert out["abundances"].shape == (4, 32, 32)
    assert out["rmse"] < 0.2  # mostly reconstructable
