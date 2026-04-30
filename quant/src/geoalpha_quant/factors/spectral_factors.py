"""
Spectral factor models for hyperspectral unmixing.

If you squint, hyperspectral unmixing is just Fama-French with the
factors being endmember spectra (vegetation, soil, water, ...) and
the loadings being the per-pixel material abundances.  The constraints
on the loadings - non-negativity and (optionally) sum-to-one - are
exactly the long-only, fully-invested constraints I deal with all the
time on the portfolio side.

Three estimators here:

    fit_pca_factors      - quick PCA decomposition for exploration.
    fit_sparse_unmixing  - non-negative least squares with optional
                            L1 sparsity, like a lasso-on-loadings
                            cross-sectional regression.
    SpectralUnmixer      - opinionated wrapper that picks endmembers
                            via the N-FINDR-style vertex algorithm
                            and then fits abundances per pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------- #
# PCA - the Fama-French of remote sensing.
# --------------------------------------------------------------------- #

def fit_pca_factors(cube: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD-based PCA on a (B, H, W) hyperspectral cube.

    Returns
    -------
    factors     : (B, k)  endmember-like component spectra
    loadings    : (k, H, W)  per-pixel scores
    explained   : (k,) explained-variance ratios
    """
    if cube.ndim != 3:
        raise ValueError("cube must be (B, H, W)")
    B, H, W = cube.shape
    X = cube.reshape(B, H * W).astype(np.float64)
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu

    # SVD on the (B x N) matrix - B is small, N is huge, so this is cheap.
    U, S, _ = np.linalg.svd(Xc @ Xc.T / max(X.shape[1] - 1, 1))
    factors = U[:, :k]
    explained = (S[:k] / S.sum()).astype(np.float64)
    loadings = (factors.T @ Xc).reshape(k, H, W)
    return factors, loadings, explained


# --------------------------------------------------------------------- #
# Sparse non-negative unmixing.
# --------------------------------------------------------------------- #

def _nnls(A: np.ndarray, b: np.ndarray, n_iter: int = 200, l1: float = 0.0) -> np.ndarray:
    """Non-negative least squares with optional L1 via projected gradient.

    Mirrors the long-only, lasso-style portfolio fit.  Step size set
    via a Lipschitz upper bound (||A^T A||_2).
    """
    L = float(np.linalg.norm(A.T @ A, ord=2)) + 1e-9
    x = np.zeros(A.shape[1])
    for _ in range(n_iter):
        grad = A.T @ (A @ x - b) + l1
        x = np.maximum(x - grad / L, 0.0)
    return x


def fit_sparse_unmixing(
    spectra: np.ndarray,
    endmembers: np.ndarray,
    l1: float = 0.01,
) -> np.ndarray:
    """Per-pixel non-negative + L1 unmixing.

    Parameters
    ----------
    spectra : (N, B) observed pixel spectra.
    endmembers : (B, k) reference endmember matrix.
    l1 : sparsity penalty.

    Returns
    -------
    abundances : (N, k) non-negative abundances per pixel.
    """
    N, B = spectra.shape
    k = endmembers.shape[1]
    out = np.zeros((N, k), dtype=np.float64)
    for i in range(N):
        out[i] = _nnls(endmembers, spectra[i], l1=l1)
    return out


# --------------------------------------------------------------------- #
# N-FINDR style endmember picker.
# --------------------------------------------------------------------- #

def _nfindr(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 0) -> np.ndarray:
    """Greedy maximum-volume vertex selection.

    The classical N-FINDR (Winter 1999) - we project to the top k-1
    PCA components first and then iteratively swap vertices to grow
    the simplex volume.
    """
    rng = np.random.default_rng(seed)
    B, N = X.shape
    # Reduce dimensionality to (k - 1) via PCA for a numerically-stable volume.
    Xc = X - X.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(Xc @ Xc.T / max(N - 1, 1))
    proj = U[:, :k - 1].T @ Xc            # (k-1) x N
    proj = np.vstack([proj, np.ones(N)])  # k x N (homogeneous coords)

    init_idx = rng.choice(N, size=k, replace=False)
    E = proj[:, init_idx]
    best_vol = abs(np.linalg.det(E))

    for _ in range(max_iter):
        improved = False
        for j in range(k):
            for i in range(N):
                if i in init_idx:
                    continue
                cand = E.copy()
                cand[:, j] = proj[:, i]
                vol = abs(np.linalg.det(cand))
                if vol > best_vol * 1.001:
                    E = cand
                    init_idx[j] = i
                    best_vol = vol
                    improved = True
        if not improved:
            break
    return X[:, init_idx]


@dataclass
class SpectralUnmixer:
    """End-to-end pipeline.

    1. Pick k endmembers from the cube via N-FINDR.
    2. Fit non-negative abundances per pixel (with optional L1).
    3. Report a residual reconstruction error so the operator can
       sanity-check that the chosen k explains enough of the variance.
    """

    n_endmembers: int = 4
    l1: float = 0.01
    seed: int = 0

    def fit_predict(self, cube: np.ndarray) -> dict:
        if cube.ndim != 3:
            raise ValueError("cube must be (B, H, W)")
        B, H, W = cube.shape
        X = cube.reshape(B, H * W).astype(np.float64)
        endmembers = _nfindr(X, k=self.n_endmembers, seed=self.seed)
        abund = fit_sparse_unmixing(X.T, endmembers, l1=self.l1)
        recon = endmembers @ abund.T
        residual = X - recon
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        return {
            "endmembers": endmembers,                   # (B, k)
            "abundances": abund.T.reshape(self.n_endmembers, H, W),
            "rmse": rmse,
            "explained_variance_ratio": 1.0 - residual.var() / max(X.var(), 1e-12),
        }
