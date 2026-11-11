"""
Distance computation utilities for LCP.

Translates LCPhelpers.R: dist(), LCPdefault_distance().
"""

import numpy as np
from typing import Optional, List, Dict


def pairwise_dist(x: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """
    Compute distance between rows of `x` and a single point `x1`.

    Equivalent to R dist().

    If x is 1-D or single-column, uses absolute difference.
    Otherwise uses normalised Euclidean distance: sqrt(mean((x_j - x1_j)^2)).

    Args:
        x:  shape (N, p) or (N,) or (N, 1).
        x1: shape (p,) or scalar.

    Returns:
        dists: shape (N,).
    """
    if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
        return np.abs(x.ravel() - x1)
    else:
        diff = x - x1.reshape(1, -1)
        return np.sqrt(np.mean(diff ** 2, axis=1))


def _pairwise_matrix(z: np.ndarray) -> np.ndarray:
    """
    Compute all-pairs distance matrix for a set of points.

    Args:
        z: shape (N,) or (N, p).

    Returns:
        D: shape (N, N), where D[i, j] = |z_i - z_j| (1D) or
           sqrt(mean((z_i - z_j)^2)) (multivariate).
    """
    if z.ndim == 1 or (z.ndim == 2 and z.shape[1] == 1):
        z1d = z.ravel()
        return np.abs(z1d[:, None] - z1d[None, :])
    else:
        # Use broadcasting
        diff = z[:, None, :] - z[None, :, :]
        return np.sqrt(np.mean(diff ** 2, axis=2))


def _cross_distance(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """
    Cross-distance matrix between two sets of points.

    Args:
        z1: shape (N1,) or (N1, p).
        z2: shape (N2,) or (N2, p).

    Returns:
        D: shape (N1, N2).
    """
    if z1.ndim == 1 or (z1.ndim == 2 and z1.shape[1] == 1):
        return np.abs(z1.ravel()[:, None] - z2.ravel()[None, :])
    else:
        diff = z1[:, None, :] - z2[None, :, :]
        return np.sqrt(np.mean(diff ** 2, axis=2))


def lcp_default_distance(
    utangent: Optional[np.ndarray],
    uorthogonal: Optional[np.ndarray],
    estimated_sds: List[np.ndarray],
    xtrain: np.ndarray,
    xcalibration: np.ndarray,
    xtest: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Build distance matrices for LCP, combining a variance-component distance
    with tangent- and orthogonal-space distances.

    Equivalent to R LCPdefault_distance().

    Args:
        utangent:    first right-singular vector of CV Jacobian (p,), or None.
        uorthogonal: remaining right-singular vectors (p, r), or None.
        estimated_sds: list of 3 arrays — [cv_sds, cal_sds, test_sds],
                       each shape (N_i,).
        xtrain:      training features, shape (n_train, p).
        xcalibration: calibration features, shape (n_cal, p).
        xtest:       test features, shape (n_test, p).

    Returns:
        dict with keys:
            'Hcv':    CV distance matrix (n_train, n_train).
            'H':      calibration distance matrix (n_cal, n_cal).
            'Hnew':   test → cal distance matrix (n_test, n_cal).
            'HnewT':  cal → test distance matrix = Hnew.T (n_cal, n_test).
    """
    z = estimated_sds[0]
    z0 = estimated_sds[1]
    z1 = estimated_sds[2]
    n = len(z)
    n0 = len(z0)
    m = len(z1)

    # ── Variance-component distance ──
    Hdist1 = np.zeros((n0 + m, n0 + m))
    Hdist1cv = np.zeros((n, n))

    for j in range(n0):
        Hdist1[:n0, j] = pairwise_dist(z0, z0[j])
        Hdist1[n0:, j] = pairwise_dist(z1, z0[j])
    for j in range(m):
        Hdist1[:n0, j + n0] = pairwise_dist(z0, z1[j])
        Hdist1[n0:, j + n0] = pairwise_dist(z1, z1[j])

    for j in range(n):
        Hdist1cv[:, j] = pairwise_dist(z, z[j])

    # ── Tangent-space distance ──
    s02 = 0.0
    Hdist2 = np.zeros((n0 + m, n0 + m))
    Hdist2cv = np.zeros((n, n))

    if utangent is not None:
        z0par = xcalibration @ utangent
        z1par = xtest @ utangent
        zpar = xtrain @ utangent

        for j in range(n0):
            Hdist2[:n0, j] = pairwise_dist(z0par, z0par[j])
            Hdist2[n0:, j] = pairwise_dist(z1par, z0par[j])
        for j in range(m):
            Hdist2[:n0, j + n0] = pairwise_dist(z0par, z1par[j])
            Hdist2[n0:, j + n0] = pairwise_dist(z1par, z1par[j])

        for j in range(n):
            Hdist2cv[:, j] = pairwise_dist(zpar, zpar[j])

        mask = (np.arange(n)[:, None] != np.arange(n)[None, :])
        s02 = np.mean(Hdist2cv[mask])

    # ── Orthogonal-space distance ──
    s03 = 0.0
    Hdist3 = np.zeros((n0 + m, n0 + m))
    Hdist3cv = np.zeros((n, n))

    if uorthogonal is not None and uorthogonal.shape[1] > 0:
        z0orth = xcalibration @ uorthogonal
        z1orth = xtest @ uorthogonal
        zorth = xtrain @ uorthogonal

        for j in range(n0):
            Hdist3[:n0, j] = pairwise_dist(z0orth, z0orth[j])
            Hdist3[n0:, j] = pairwise_dist(z1orth, z0orth[j])
        for j in range(m):
            Hdist3[:n0, j + n0] = pairwise_dist(z0orth, z1orth[j])
            Hdist3[n0:, j + n0] = pairwise_dist(z1orth, z1orth[j])

        for j in range(n):
            Hdist3cv[:, j] = pairwise_dist(zorth, zorth[j])

        mask = (np.arange(n)[:, None] != np.arange(n)[None, :])
        s03 = np.mean(Hdist3cv[mask])

    # ── Combine ──
    mask = (np.arange(n)[:, None] != np.arange(n)[None, :])
    s01 = np.mean(Hdist1cv[mask])

    w = s02 / (s02 + s03) if (s02 + s03) > 0 else 0.0
    if uorthogonal is None:
        w = 0.0

    Hdist23 = (1.0 - w) * Hdist2 + w * Hdist3
    Hdist23cv = (1.0 - w) * Hdist2cv + w * Hdist3cv
    s023 = np.mean(Hdist23cv[mask])

    a = 1.0
    Hcv = a * Hdist1cv / s01 + Hdist23cv / s023
    tmp = a * Hdist1 / s01 + Hdist23 / s023
    H_cal = tmp[:n0, :n0]
    Hnew = tmp[n0:, :n0]
    HnewT = tmp[:n0, n0:]

    return {
        "Hcv": Hcv,
        "H": H_cal,
        "Hnew": Hnew,
        "HnewT": HnewT,
    }
