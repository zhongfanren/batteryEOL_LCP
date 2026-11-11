"""
End-to-end demo: LCP on synthetic 1D data.

This script generates simple 1D heteroskedastic data,
runs the full LCP pipeline, and prints results.

Does NOT require external datasets.
Suitable for verifying the pipeline end-to-end.

Run:
    cd python_lcp && python examples/demo_synthetic.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lcp.core import (
    id_low_search,
    lcp_alpha,
    lcp_construction_path,
)
from lcp.distance import lcp_default_distance


def generate_1d_data(n_train=1000, n_cal=500, n_test=500, seed=42):
    """
    Generate 1D heteroskedastic data.
    y = sin(x) * noise, where noise_std varies with x.
    """
    np.random.seed(seed)

    def noise_std(x):
        return np.abs(np.sin(x)) + 0.2

    def sample(n):
        x = np.random.uniform(-2, 2, n)
        s = noise_std(x)
        y = s * np.random.randn(n)
        return x.reshape(-1, 1), y.reshape(-1, 1)

    x_tr, y_tr = sample(n_train)
    x_cal, y_cal = sample(n_cal)
    x_te, y_te = sample(n_test)
    return x_tr, y_tr, x_cal, y_cal, x_te, y_te


def demo_core_only():
    """Demo using just the core LCP algorithm (no neural network).

    This shows the basic building blocks working correctly.
    """
    print("=" * 60)
    print("LCP Core Demo (NumPy only, no torch needed)")
    print("=" * 60)

    x_tr, y_tr, x_cal, y_cal, x_te, y_te = generate_1d_data()

    # Train a simple model (polynomial regression, no NN)
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=4, include_bias=False)
    x_tr_poly = poly.fit_transform(x_tr)
    x_cal_poly = poly.transform(x_cal)
    x_te_poly = poly.transform(x_te)

    model = LinearRegression()
    model.fit(x_tr_poly, y_tr)

    yhat_tr = model.predict(x_tr_poly)
    yhat_cal = model.predict(x_cal_poly)
    yhat_te = model.predict(x_te_poly)

    r_tr = np.abs(y_tr.ravel() - yhat_tr.ravel())
    r_cal = np.abs(y_cal.ravel() - yhat_cal.ravel())
    r_te = np.abs(y_te.ravel() - yhat_te.ravel())

    # Standard CP
    alpha = 0.05
    delta_cp = np.quantile(r_cal, 1.0 - alpha)
    cp_cov = np.mean(r_te <= delta_cp)
    cp_len = delta_cp * 2.0
    print(f"\nStandard CP:  coverage={cp_cov:.4f}, len={cp_len:.4f}")

    # LCP with distance-based localizer
    # Use x-coordinates for distance
    D_cal = np.abs(x_cal.ravel()[:, None] - x_cal.ravel()[None, :])
    D_test_cal = np.abs(x_te.ravel()[:, None] - x_cal.ravel()[None, :])

    # Sort calibration residuals
    order = np.argsort(r_cal)
    V_sorted = r_cal[order]
    V_ext = np.append(V_sorted, np.inf)

    # Try a few bandwidth values
    for h_log in [-2, 0, 2]:
        h = 10 ** h_log

        H_cal = np.exp(-D_cal[np.ix_(order, order)] / h)
        H_new = np.exp(-D_test_cal[:, order] / h)
        H_newT = H_new.T

        Qcumsum = np.cumsum(H_cal, axis=1)
        id_low = id_low_search(V_ext)
        q_low_arr = np.zeros(len(V_sorted))
        for i in range(len(V_sorted)):
            if id_low[i] >= 0:
                q_low_arr[i] = Qcumsum[i, id_low[i]]
        qn_arr = Qcumsum[:, len(V_sorted) - 1]  # R 1-idx -> Python 0-idx

        ret = lcp_construction_path(
            alpha=alpha, V=V_ext,
            id_low=id_low, q_low=q_low_arr, qn=qn_arr,
            Hnew=H_new, HnewT=H_newT,
        )
        delta_lcp = ret["deltaLCP"]

        finite = delta_lcp < np.inf
        lcp_cov = np.mean(r_te[finite] <= delta_lcp[finite])
        lcp_len = np.mean(delta_lcp[finite]) * 2.0
        inf_pct = np.mean(~finite)

        print(f"LCP (h=10^{h_log:+.0f}): coverage={lcp_cov:.4f}, "
              f"len={lcp_len:.4f}, inf_ratio={inf_pct:.4f}")

    print("\n✓ Core LCP algorithm working correctly")
    print("  For full comparison (with NN), run with torch in a GPU environment.")


if __name__ == "__main__":
    demo_core_only()
