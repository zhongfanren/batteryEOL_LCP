"""
Core LCP algorithm — direct translation of LCPcpp.cpp to NumPy.

The LCP (Localized Conformal Prediction) algorithm computes locally-weighted
empirical CDFs for each test point, then extracts prediction bands.

Key internal data structures:
    V:     sorted calibration residuals, extended with +Inf at the end (shape: n+1)
    id_low: for each position k, the largest index i such that V[i] < V[k] (shape: n+1)
    q_low:  Qcumsum[i, id_low[i]] — cumulative weight up to the lower bound (shape: n)
    qn:     Qcumsum[i, n-1] — total cumulative weight for row i (shape: n)
    Hnew:   test→cal  kernel weights, rowwise (shape: m × n)
    HnewT:  cal→test kernel weights = Hnew.T (shape: n × m)
"""

import numpy as np
from typing import Tuple, Optional


# ── C++ → NumPy: id_low_search ──────────────────────────────────────────────

def id_low_search(V: np.ndarray) -> np.ndarray:
    """
    For a *sorted* vector V (extended with +Inf at end), compute the
    id_low array where id_low[i] = largest index k such that V[k] < V[i].

    Equivalent to C++ id_low_search().

    Args:
        V: 1-D array, sorted ascending, last element = np.inf, shape (n+1,).

    Returns:
        id_low: 1-D int array, shape (n+1,), values in [-1, n-1].
    """
    n = len(V)
    id_low = np.full(n, -1, dtype=int)
    it = -1
    for i in range(n):
        while it + 1 < n and V[it + 1] < V[i]:
            it += 1
        id_low[i] = it
    return id_low


# ── C++ → NumPy: q_low_compute ──────────────────────────────────────────────

def q_low_compute(id_low: np.ndarray, Qcumsum: np.ndarray) -> np.ndarray:
    """
    Given id_low and the row-wise cumulative sum Qcumsum,
    extract q_low[i] = Qcumsum[i, id_low[i]] (0 if id_low[i] < 0).

    Equivalent to C++ q_low_compute().

    Args:
        id_low: 1-D int array, shape (n,).
        Qcumsum: 2-D float array, shape (n, n).

    Returns:
        q_low: 1-D float array, shape (n,).
    """
    n = len(id_low)
    q_low = np.zeros(n)
    for i in range(n):
        if id_low[i] >= 0:
            q_low[i] = Qcumsum[i, id_low[i]]
    return q_low


# ── C++ → NumPy: lcp_construction_path_distance ─────────────────────────────

def lcp_construction_path_distance(
    V: np.ndarray,
    id_low: np.ndarray,
    q_low: np.ndarray,
    qn: np.ndarray,
    Hnew: np.ndarray,
    HnewT: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core LCP algorithm — compute alpha-values for each test point × each
    calibration rank.  This is the heart of the paper.

    Equivalent to C++ LCP_construction_path_distance().

    Args:
        V:      sorted residuals + Inf, shape (n+1,)  where n = n_cal + 1.
        id_low: from id_low_search(V), shape (n+1,).
        q_low:  from q_low_compute(id_low[:-1], Qcumsum_cal), shape (n,).
        qn:     Qcumsum_cal[:, n-1], shape (n,).
        Hnew:   test→cal kernel matrix, shape (m, n).
        HnewT:  cal→test kernel matrix = Hnew.T, shape (n, m).

    Returns:
        alphas:  shape (m, n) — alphas[i, k] = P(idealised residual ≤ V[k] | test i).
        V:       the input V (pass-through for convenience).
    """
    m = Hnew.shape[0]        # number of test points
    n = V.shape[0]           # n_cal + 1 (last = Inf)
    n0 = n - 1               # number of calibration points

    alphas = np.zeros((m, n))

    # ── Step 1: compute un-normalized theta (localised cumulative mass) ──
    # theta[i, :] accumulates Hnew contributions col-by-col as we walk V.
    theta = np.zeros((m, n))
    for i in range(1, n):
        theta[:, i] = theta[:, i - 1]
        gap = id_low[i] - id_low[i - 1]
        if gap > 0:
            i1_start = id_low[i - 1] + 1
            for i1 in range(i1_start, id_low[i] + 1):
                theta[:, i] += Hnew[:, i1]

    qn_new = theta[:, n0].copy()  # un-normalized total weight per test point
    theta0 = theta.copy()

    # ── Step 2: for each test point, compute its normalised CDF on V ──
    for i in range(m):
        normalizers = qn + HnewT[:, i]
        # p1 = prob mass *with* test→cal connection included
        p1 = (q_low + HnewT[:, i]) / normalizers
        # p2 = prob mass *without* test→cal connection
        p2 = q_low / normalizers

        ttheta_i = theta[i, :].copy() / (qn_new[i] + 1.0)
        ttheta_i0 = theta0[i, :n0].copy() / (qn_new[i] + 1.0)

        # Partition calibration points into three sets based on
        # whether their lower/upper probability bounds straddle ttheta_i0.
        idA1 = np.where(p1 < ttheta_i0)[0]
        idA2 = np.where(p2 >= ttheta_i0)[0]
        idA3 = np.where((p2 < ttheta_i0) & (p1 >= ttheta_i0))[0]

        pA1_sort = np.sort(p1[idA1]) if len(idA1) > 0 else np.array([])
        pA2_sort = np.sort(p2[idA2]) if len(idA2) > 0 else np.array([])
        pA3_sort = np.sort(id_low[idA3]) if len(idA3) > 0 else np.array([])

        itA1, itA2, itA3 = 0, 0, 0
        nA1, nA2, nA3 = len(pA1_sort), len(pA2_sort), len(pA3_sort)

        for k in range(n):
            # Advance pointer in A1
            if nA1 > 0 and itA1 < nA1:
                while itA1 < nA1 - 1 and pA1_sort[itA1] < ttheta_i[k]:
                    itA1 += 1
                if itA1 == nA1 - 1 and pA1_sort[itA1] < ttheta_i[k]:
                    itA1 += 1

            # Advance pointer in A2
            if nA2 > 0 and itA2 < nA2:
                while itA2 < nA2 - 1 and pA2_sort[itA2] < ttheta_i[k]:
                    itA2 += 1
                if itA2 == nA2 - 1 and pA2_sort[itA2] < ttheta_i[k]:
                    itA2 += 1

            # Advance pointer in A3 (compared against id_low(k))
            if nA3 > 0 and itA3 < nA3:
                id_low_k = id_low[k]
                while itA3 < nA3 - 1 and pA3_sort[itA3] < id_low_k:
                    itA3 += 1
                if itA3 == nA3 - 1 and pA3_sort[itA3] < id_low_k:
                    itA3 += 1

            alphas[i, k] = (itA1 + itA2 + itA3) / (n0 + 1.0)

    return alphas, V


# ── Helper: lcp_alpha (extract band from alphas) ───────────────────────────

def lcp_alpha(
    vs: np.ndarray,
    alphas: np.ndarray,
    alpha: float = 0.95,
) -> np.ndarray:
    """
    For each test point, find the smallest value in `vs` such that the
    local empirical CDF (encoded in `alphas`) reaches `alpha`.

    Equivalent to R LCP_alpha().

    Args:
        vs:      calibration residual values, sorted, shape (n,).
        alphas:  from lcp_construction_path_distance, shape (m, n).
        alpha:   target confidence level (e.g. 0.95 for 95% coverage).

    Returns:
        bounds: shape (m,) — the delta-band for each test point.
                np.inf where coverage cannot be met.
    """
    if alphas.ndim == 1:
        idx = np.where(alphas < alpha)[0]
        if len(idx) > 0:
            return vs[idx[-1]]
        return np.inf

    m = alphas.shape[0]
    bounds = np.full(m, np.inf)

    # For each test point, find where alphas[i, :] first reaches >= alpha.
    for i in range(m):
        idx = np.where(alphas[i, :] < alpha)[0]
        if len(idx) > 0:
            bounds[i] = vs[idx[-1]]

    return bounds


# ── Helper: lcp_construction_path (R wrapper) ──────────────────────────────

def lcp_construction_path(
    alpha: float,
    V: np.ndarray,
    id_low: np.ndarray,
    q_low: np.ndarray,
    qn: np.ndarray,
    Hnew: np.ndarray,
    HnewT: np.ndarray,
) -> dict:
    """
    R LCP_construction_path() — thin wrapper calling the distance version.

    Returns dict with keys: 'deltaLCP', 'Smat'.
    """
    alphas, _ = lcp_construction_path_distance(
        V=V, id_low=id_low, q_low=q_low, qn=qn,
        Hnew=Hnew, HnewT=HnewT,
    )
    delta = lcp_alpha(vs=V, alphas=alphas, alpha=1.0 - alpha)
    return {"deltaLCP": delta, "Smat": alphas}


# ── Auto-tuning ──────────────────────────────────────────────────────────────

def auto_tune_distance(
    V: np.ndarray,
    n: int,
    hs: np.ndarray,
    D: np.ndarray,
    alpha: float = 0.05,
    delta: float = 0.05,
    B: int = 10,
    trace: bool = True,
    lmbda: float = 1.0,
) -> dict:
    """
    Auto-tune the bandwidth `h` for the distance-based LCP localizer.

    Equivalent to R autoTune_distance().

    Args:
        V:      calibration residuals (unsorted), shape (n0,).
        n:      LCP internal n (from LCPmodule$n).
        hs:     candidate bandwidths, shape (J,).
        D:      pairwise distance matrix, shape (n0, n0).
        alpha:  target miscoverage (e.g. 0.05).
        delta:  tolerance for probability-of-infinite-bound.
        B:      number of bootstrap replicates.
        trace:  print progress.
        lmbda: regularisation weight for standard deviation penalty.

    Returns:
        dict with keys: 'Ts' (J×3 table), 'h' (chosen bandwidth), 'intermediants'.
    """
    n0 = len(V)
    J = len(hs)
    n1 = min(n0, n + 1)
    B1 = 1 if n1 >= 2 / 3 * n0 else B

    C1 = np.zeros((B1, 2, J))
    thr = np.max(V)

    if trace:
        print("###estimate means######")

    for b in range(B1):
        if trace:
            print(b + 1)
        I = np.sort(np.random.choice(n0, size=n1, replace=False))
        Vb = V[I]
        Db = D[np.ix_(I, I)]

        for j in range(J):
            h = hs[j]
            H_mat = np.exp(-Db / h)
            Qcumsumb = np.cumsum(H_mat, axis=1)
            ret1 = _lcp_construction_distance_loop(Vb, Qcumsumb, H_mat)
            alpha_vals = ret1["alphas"]

            tmp_vals = np.zeros(n0)
            for i in range(n1):
                Vb0 = np.delete(Vb, i)
                tmp_vals[i] = lcp_alpha(
                    vs=np.append(Vb0, np.inf),
                    alphas=alpha_vals[i, :],
                    alpha=1.0 - alpha,
                )
            C1[b, 0, j] = np.mean(tmp_vals > thr)
            C1[b, 1, j] = np.mean(tmp_vals[tmp_vals <= thr])

    C2 = np.zeros((n0, B, J))
    mu = np.zeros((n0, J))
    varphi = np.full((n0, B, J), np.inf)

    if lmbda > 0:
        if trace:
            print("###standard deviation estimation#####")
        for b in range(B):
            if trace:
                print(b + 1)
            I = np.sort(np.random.choice(n0, size=n, replace=True))
            Vb = V[I]
            Db = D[np.ix_(I, I)]
            Dnew = D[:, I]
            DnewT = D[I, :]
            Vb_ext = np.append(Vb, np.inf)
            id_low_b = id_low_search(Vb_ext)

            for j in range(J):
                h = hs[j]
                H_mat = np.exp(-Db / h)
                Hnew_mat = np.exp(-Dnew / h)
                HnewT_mat = np.exp(-DnewT / h)
                Qcumsum_b = np.cumsum(H_mat, axis=1)
                q_low_b = q_low_compute(
                    id_low_b[:-1], Qcumsum_b
                )

                ret = lcp_construction_path(
                    alpha=alpha,
                    V=Vb_ext,
                    id_low=id_low_b,
                    q_low=q_low_b,
                    qn=Qcumsum_b[:, n - 1],  # n = n_cal; 0-indexed
                    Hnew=Hnew_mat,
                    HnewT=HnewT_mat,
                )
                C2[:, b, j] = lcp_alpha(
                    vs=Vb_ext, alphas=ret["Smat"], alpha=1.0 - alpha,
                )

        for j in range(J):
            for i in range(n0):
                tmp_col = C2[i, :, j]
                finite_mask = tmp_col <= thr
                tmp1 = np.mean(finite_mask)
                if tmp1 > 0:
                    mu[i, j] = np.mean(tmp_col[finite_mask])
                for b in range(B):
                    if C2[i, b, j] < np.inf:
                        varphi[i, b, j] = (C2[i, b, j] - mu[i, j]) ** 2

    # Build summary table
    Ts = np.zeros((J, 3))
    Ts[:, 0] = hs
    Ts[:, 1] = np.mean(C1[:, 0, :], axis=0)
    Ts[:, 2] = np.mean(C1[:, 1, :], axis=0)

    intermediants = {"probs": Ts[:, 1].copy(), "means": Ts[:, 2].copy(), "sds": np.zeros(J)}

    if lmbda > 0:
        for j in range(J):
            tmp_phi = varphi[:, :, j]
            intermediants["sds"][j] = np.sqrt(np.mean(tmp_phi[tmp_phi < np.inf]))
            Ts[j, 2] += lmbda * intermediants["sds"][j]

    # Select best h
    idx = np.where(Ts[:, 1] <= delta)[0]
    if len(idx) == 0:
        h_best = np.nan
    else:
        hs_sub = hs[idx]
        c0 = Ts[idx, 2]
        h_best = hs_sub[np.argmin(c0)]

    return {"Ts": Ts, "h": h_best, "intermediants": intermediants}


# ── Internal: lcp_construction_distance_loop (leave-one-out) ───────────────

def _lcp_construction_distance_loop(
    V: np.ndarray,
    Qcumsum: np.ndarray,
    H: np.ndarray,
) -> dict:
    """
    Leave-one-out LCP for auto-tuning.
    Equivalent to C++ LCP_construction_distance_loop().

    Args:
        V:       sorted residuals, shape (n,).
        Qcumsum: row-wise cumulative sum of kernel weights, shape (n, n).
        H:       kernel weight matrix, shape (n, n).

    Returns:
        dict with keys 'alphas' (n×n), 'vs'.
    """
    n = V.shape[0]
    id_low_full = id_low_search(np.append(V, np.inf))
    q_low_full = q_low_compute(id_low_full[:-1], Qcumsum)
    alphas = np.zeros((n, n + 1))

    for i in range(n):
        # Build leave-one-out vectors
        Vi = np.zeros(n)
        id_low_i = np.zeros(n, dtype=int)
        q_low_i = np.zeros(n - 1)
        qn_i = np.zeros(n - 1)
        Hnew_i = np.zeros((1, n - 1))
        HnewT_i = np.zeros((n - 1, 1))

        for i1 in range(n - 1):
            i2 = i1 if i1 < i else i1 + 1
            q_low_i[i1] = q_low_full[i2]
            id_low_i[i1] = id_low_full[i2]
            Vi[i1] = V[i2]

            if i <= id_low_full[i2]:
                q_low_i[i1] -= H[i2, i]
                id_low_i[i1] -= 1

            qn_i[i1] = Qcumsum[i2, n - 1] - H[i2, i]
            Hnew_i[0, i1] = H[i, i2]
            HnewT_i[i1, 0] = H[i2, i]

        Vi[n - 1] = np.inf
        id_low_i[n - 1] = n - 2

        alphas_i, _ = lcp_construction_path_distance(
            V=Vi, id_low=id_low_i, q_low=q_low_i, qn=qn_i,
            Hnew=Hnew_i, HnewT=HnewT_i,
        )
        alphas[i, :] = alphas_i[0, :]

    return {"alphas": alphas, "vs": np.append(V, np.inf)}
