"""
Full LCP comparison pipeline — translates LCPcompare() from LCPexperiments.R.

Orchestrates:
  1. Point-prediction NN (MSE)
  2. Variance-prediction NN (on log-residuals^2)
  3. Standard CP bands (CR, CLR)
  4. LCP bands with auto-tuned bandwidth (LCR, LCLR)
  5. Optional: quantile-regression variants (CQR, LCQR, CQLR, LCQLR)
"""

import numpy as np
import torch
import time
from typing import Optional, List, Dict

from .core import (
    id_low_search,
    lcp_alpha,
    lcp_construction_path,
)
from .distance import lcp_default_distance
from .nn import MyNeuralNet, LearnerOptimizer, AllQuantileLoss


def lcp_compare(
    xtrain: np.ndarray,
    ytrain: np.ndarray,
    xcalibration: np.ndarray,
    ycalibration: np.ndarray,
    xtest: np.ndarray,
    ytest: np.ndarray,
    alpha: float = 0.05,
    quantiles: List[float] = None,
    nfolds: int = 3,
    random_state: int = 1,
    save_path: Optional[str] = None,
    print_out: int = 10,
    epochs: int = 50,
    include_qr_variants: bool = True,
    device: str = "cpu",
) -> Dict:
    """
    Compare 8 prediction-interval constructions (or 4 if
    include_qr_variants=False) on a given train/cal/test split.

    Args:
        xtrain, ytrain:        training data.
        xcalibration, ycalibration: calibration data.
        xtest, ytest:          test data.
        alpha:                 miscoverage level (default 0.05 → 95% PI).
        quantiles:             quantile-regression targets.
        nfolds:                number of CV folds.
        random_state:          random seed.
        save_path:             path prefix for model checkpoints.
        print_out:             verbosity interval.
        epochs:                training epochs.
        include_qr_variants:   if True, also compute CQR/LCQR/CQLR/LCQLR.
        device:                torch device ('cpu' or 'cuda').

    Returns:
        dict with coverages, lens, Infpercent, PIbands.
    """
    if quantiles is None:
        quantiles = [0.025, 0.05, 0.1, 0.9, 0.95, 0.975]

    hidden_size = 32
    lr = 0.005
    wd = 1e-6
    drop_out = 0.2
    batch_size = 32
    optimizer_class = torch.optim.Adam
    p = xtrain.shape[1]

    n_methods = 8 if include_qr_variants else 4
    method_names = (["CR", "LCR", "CLR", "LCLR", "CQR", "LCQR", "CQLR", "LCQLR"]
                    if include_qr_variants else ["CR", "LCR", "CLR", "LCLR"])

    coverages = np.full(n_methods, np.nan)
    lens = np.full(n_methods, np.nan)
    inf_pct = np.full(n_methods, np.nan)

    # ── 1. Point-prediction model (MSE) ─────────────────────────────────
    mse_loss_fn = nn.MSELoss() if hasattr(torch.nn, 'MSELoss') else torch.nn.functional.mse_loss

    synthetic_mse = LearnerOptimizer(
        network_factory=MyNeuralNet,
        optimizer_class=optimizer_class,
        loss_func=mse_loss_fn if callable(mse_loss_fn) else torch.nn.functional.mse_loss,
        device=device,
        loss_type="mse",
        test_ratio=0.2,
        CV=True,
        nfolds=nfolds,
        random_state=random_state,
        save_path=save_path,
        hidden_size=hidden_size,
        in_shape=p,
        drop_out=drop_out,
        lr=lr,
        wd=wd,
    )
    synthetic_mse.fit(xtrain, ytrain, epochs=epochs,
                      batch_size=batch_size, print_out=print_out)

    cv_ret_mse = synthetic_mse.cv_evaluate(num_classes=1, requires_grad=False)
    yhat_cal = synthetic_mse.predict(xcalibration, requires_grad=False)["yhat"]
    yhat_te = synthetic_mse.predict(xtest, requires_grad=False)["yhat"]

    r_cv = ytrain[:, 0] - cv_ret_mse["yhat"][:, 0]
    r_cal = yhat_cal[:, 0] - ycalibration[:, 0]
    r_te = yhat_te[:, 0] - ytest[:, 0]

    # ── 2. Variance model (MSE on log-residuals^2) ──────────────────────
    ysd = np.log(r_cv ** 2 + np.mean(r_cv ** 2)).reshape(-1, 1)

    synthetic_var = LearnerOptimizer(
        network_factory=MyNeuralNet,
        optimizer_class=optimizer_class,
        loss_func=torch.nn.functional.mse_loss,
        device=device,
        loss_type="mse",
        test_ratio=0.2,
        CV=True,
        nfolds=nfolds,
        random_state=random_state,
        save_path=save_path,
        hidden_size=hidden_size,
        in_shape=p,
        drop_out=drop_out,
        lr=lr,
        wd=wd,
    )
    synthetic_var.foldid = synthetic_mse.foldid
    synthetic_var.fit(xtrain, ysd, epochs=epochs,
                      batch_size=batch_size, print_out=print_out)

    cv_ret_var = synthetic_var.cv_evaluate(num_classes=1, requires_grad=True)
    cal_ret_var = synthetic_var.predict(xcalibration, requires_grad=True)
    test_ret_var = synthetic_var.predict(xtest, requires_grad=True)

    # Estimated & observed scales
    est_sds = [
        np.sqrt(np.exp(cv_ret_var["yhat"][:, 0])),
        np.sqrt(np.exp(cal_ret_var["yhat"][:, 0])),
        np.sqrt(np.exp(test_ret_var["yhat"][:, 0])),
    ]
    obs_sds = [np.abs(r_cv), np.abs(r_cal), np.abs(r_te)]

    # ── 3. CR, CLR (standard conformal) ─────────────────────────────────
    delta_cp = np.quantile(obs_sds[1], 1.0 - alpha)
    lens[0] = delta_cp * 2.0
    coverages[0] = np.mean(obs_sds[2] <= delta_cp)

    delta_cp_clr = np.quantile(obs_sds[1] / est_sds[1], 1.0 - alpha)
    lens[2] = np.mean(delta_cp_clr * est_sds[2] * 2.0)
    coverages[2] = np.mean(obs_sds[2] <= delta_cp_clr * est_sds[2])

    # ── 4. LCR, LCLR (localized conformal) ──────────────────────────────
    tangent_cv = cv_ret_var["jacobians"][:, 0, :]

    if tangent_cv.ndim == 1 or np.sum(tangent_cv ** 2) == 0:
        utangent = None
        uorthogonal = np.eye(p)
    else:
        U, S, Vt = np.linalg.svd(tangent_cv, full_matrices=False)
        utangent = Vt[0, :]
        uorthogonal = Vt[1:, :].T if Vt.shape[0] > 1 else None

    Hlists = lcp_default_distance(utangent, uorthogonal, est_sds,
                                  xtrain, xcalibration, xtest)

    s0_mean = np.mean(Hlists["Hcv"][np.eye(Hlists["Hcv"].shape[0]) == 0])
    max0 = np.max(Hlists["Hcv"]) * 2.0
    min0 = np.quantile(Hlists["Hcv"][Hlists["Hcv"] > 0], 0.01)
    hs_candidates = np.exp(np.linspace(np.log(min0), np.log(max0), 20))

    # --- LCR ---
    Vcal_lcr = obs_sds[1].copy()
    order_lcr = np.argsort(Vcal_lcr)
    Vcal_sorted = Vcal_lcr[order_lcr]

    from .core import auto_tune_distance as _auto_tune

    lcr_mod = _LCPModuleSimple(
        H=Hlists["H"][np.ix_(order_lcr, order_lcr)],
        V=Vcal_sorted,
        h=2.0,
        alpha=alpha,
    )

    t1 = time.time()
    auto_ret = _auto_tune(
        V=obs_sds[0], n=len(Vcal_sorted), hs=hs_candidates,
        D=Hlists["Hcv"], alpha=alpha, delta=alpha / 2.0,
        B=2, trace=True, lmbda=1.0,
    )
    t2 = time.time()
    print(f"LCR auto-tuning done in {t2 - t1:.1f}s, h={auto_ret['h']}")

    lcr_mod.h = auto_ret["h"]
    lcr_mod._prepare()
    delta_lcr = lcr_mod._construct(
        Hnew=Hlists["Hnew"][:, order_lcr],
        HnewT=Hlists["HnewT"][order_lcr, :],
    )

    coverages[1] = np.mean(obs_sds[2] <= delta_lcr)
    finite_mask = delta_lcr < np.inf
    lens[1] = np.mean(delta_lcr[finite_mask]) * 2.0
    inf_pct[1] = np.mean(~finite_mask)

    # --- LCLR ---
    Vcal_lclr = (obs_sds[1] / est_sds[1]).copy()
    Vcv_lclr = (obs_sds[0] / est_sds[0]).copy()
    order_lclr = np.argsort(Vcal_lclr)

    lclr_mod = _LCPModuleSimple(
        H=Hlists["H"][np.ix_(order_lclr, order_lclr)],
        V=Vcal_lclr[order_lclr],
        h=2.0,
        alpha=alpha,
    )

    auto_ret_lclr = _auto_tune(
        V=Vcv_lclr, n=len(Vcal_lclr), hs=hs_candidates,
        D=Hlists["Hcv"], alpha=alpha, delta=alpha / 2.0,
        B=2, trace=True, lmbda=1.0,
    )
    print(f"LCLR auto-tuning done, h={auto_ret_lclr['h']}")

    lclr_mod.h = auto_ret_lclr["h"]
    lclr_mod._prepare()
    delta_lclr = lclr_mod._construct(
        Hnew=Hlists["Hnew"][:, order_lclr],
        HnewT=Hlists["HnewT"][order_lclr, :],
    )

    coverages[3] = np.mean(obs_sds[2] <= delta_lclr * est_sds[2])
    finite_mask_lclr = (delta_lclr < np.inf)
    lens[3] = np.mean((delta_lclr * est_sds[2])[finite_mask_lclr]) * 2.0
    inf_pct[3] = np.mean(~finite_mask_lclr)

    # ── 5. (Optional) Quantile regression variants ──────────────────────
    if include_qr_variants:
        _lcp_compare_qr(
            xtrain, ytrain, xcalibration, ycalibration, xtest, ytest,
            alpha, quantiles, nfolds, random_state, save_path,
            print_out, epochs, hidden_size, lr, wd, drop_out, batch_size,
            p, optimizer_class, device,
            coverages, lens, inf_pct,
            Hlists, hs_candidates, est_sds, obs_sds,
        )

    # Build output
    result = {
        "coverages": dict(zip(method_names[:n_methods], coverages[:n_methods])),
        "lens": dict(zip(method_names[:n_methods], lens[:n_methods])),
        "Infpercent": dict(zip(method_names[:n_methods], inf_pct[:n_methods])),
    }
    return result


# ── Internal: simplified LCP module (replaces R LCPmodule R6 class) ─────────

class _LCPModuleSimple:
    """Minimal LCP module for the distance-based localizer."""

    def __init__(self, H: np.ndarray, V: np.ndarray, h: float = 1.0,
                 alpha: float = 0.05):
        self.H = H
        self.V_raw = V
        self.V = np.append(V, np.inf)
        self.n = len(self.V) - 1
        self.h = h
        self.alpha = alpha

        self.Hdistance = None
        self.Qcumsum = None
        self.id_low = None
        self.qlow0 = None
        self.qn0 = None

    def _prepare(self):
        """Compute id_low, Qcumsum, etc. (R lower_idx + cumsum_unnormalized)."""
        self.Hdistance = np.exp(-self.H / self.h)
        self.Qcumsum = np.cumsum(self.Hdistance, axis=1)
        self.id_low = id_low_search(self.V)
        self.qlow0 = _gather_rows(self.Qcumsum, self.id_low[:-1])
        self.qn0 = self.Qcumsum[:, self.n - 1]  # R 1-indexed -> Python 0-indexed

    def _construct(self, Hnew: np.ndarray, HnewT: np.ndarray) -> np.ndarray:
        """Build LCP bands. (R LCP_construction)."""
        Hnew_k = np.exp(-Hnew / self.h)
        HnewT_k = np.exp(-HnewT / self.h)

        if Hnew_k.ndim == 1:
            Hnew_k = Hnew_k.reshape(1, -1)
            HnewT_k = HnewT_k.reshape(-1, 1)

        ret = lcp_construction_path(
            alpha=self.alpha,
            V=self.V,
            id_low=self.id_low,
            q_low=self.qlow0,
            qn=self.qn0,
            Hnew=Hnew_k,
            HnewT=HnewT_k,
        )
        return ret["deltaLCP"]


def _gather_rows(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """q_low_compute equivalent: mat[i, idx[i]] for each i, with idx[i]>=0."""
    result = np.zeros(len(idx))
    for i in range(len(idx)):
        if idx[i] >= 0:
            result[i] = mat[i, idx[i]]
    return result


# ── Internal: quantile-regression variants ──────────────────────────────────

def _lcp_compare_qr(
    xtrain, ytrain, xcalibration, ycalibration, xtest, ytest,
    alpha, quantiles, nfolds, random_state, save_path,
    print_out, epochs, hidden_size, lr, wd, drop_out, batch_size,
    p, optimizer_class, device,
    coverages, lens, inf_pct,
    Hlists, hs_candidates, est_sds, obs_sds,
):
    """Subroutine for CQR/LCQR/CQLR/LCQLR variants."""
    qlow, qhigh = 0.025, 0.975
    loss_func = AllQuantileLoss(quantiles)

    synthetic_qc = LearnerOptimizer(
        network_factory=MyNeuralNet,
        optimizer_class=optimizer_class,
        loss_func=loss_func,
        device=device,
        loss_type="qc",
        all_quantiles=quantiles,
        test_ratio=0.2,
        CV=True,
        nfolds=nfolds,
        random_state=random_state,
        save_path=save_path,
        hidden_size=hidden_size,
        in_shape=p,
        drop_out=drop_out,
        lr=lr,
        wd=wd,
        qlow=qlow,
        qhigh=qhigh,
    )
    synthetic_qc.fit(xtrain, ytrain, epochs=epochs,
                     batch_size=batch_size, print_out=print_out)

    cv_ret_qc = synthetic_qc.cv_evaluate(
        num_classes=len(quantiles), requires_grad=False,
    )
    test_ret_qc = synthetic_qc.predict(xtest, requires_grad=False)
    cal_ret_qc = synthetic_qc.predict(xcalibration, requires_grad=False)

    idx_low = np.where(np.array(quantiles) == alpha / 2)[0][0]
    idx_high = np.where(np.array(quantiles) == 1 - alpha / 2)[0][0]

    V1cv = cv_ret_qc["yhat"][:, idx_low] - ytrain[:, 0]
    V2cv = ytrain[:, 0] - cv_ret_qc["yhat"][:, idx_high]
    Vcv_q = np.maximum(V1cv, V2cv)

    # --- CQR ---
    V1cal = cal_ret_qc["yhat"][:, idx_low] - ycalibration[:, 0]
    V2cal = ycalibration[:, 0] - cal_ret_qc["yhat"][:, idx_high]
    Vcal_q = np.maximum(V1cal, V2cal)

    delta_cp = np.quantile(Vcal_q, 0.95)
    qL_cqr = test_ret_qc["yhat"][:, idx_low] - delta_cp
    qU_cqr = test_ret_qc["yhat"][:, idx_high] + delta_cp
    coverages[4] = np.mean((ytest[:, 0] <= qU_cqr) & (ytest[:, 0] >= qL_cqr))
    lens[4] = np.mean(qU_cqr - qL_cqr)

    # Variance model for CQR
    ysd_qc = np.log(Vcv_q ** 2 + np.mean(Vcv_q ** 2)).reshape(-1, 1)

    synthetic_var_qc = LearnerOptimizer(
        network_factory=MyNeuralNet,
        optimizer_class=optimizer_class,
        loss_func=torch.nn.functional.mse_loss,
        device=device,
        loss_type="mse",
        test_ratio=0.2,
        CV=True,
        nfolds=nfolds,
        random_state=random_state,
        save_path=save_path,
        hidden_size=hidden_size,
        in_shape=p,
        drop_out=drop_out,
        lr=lr,
        wd=wd,
    )
    synthetic_var_qc.foldid = synthetic_qc.foldid
    synthetic_var_qc.fit(xtrain, ysd_qc, epochs=epochs,
                         batch_size=batch_size, print_out=print_out)

    cv_ret_var_qc = synthetic_var_qc.cv_evaluate(
        num_classes=1, requires_grad=True,
    )
    cal_ret_var_qc = synthetic_var_qc.predict(xcalibration, requires_grad=True)
    test_ret_var_qc = synthetic_var_qc.predict(xtest, requires_grad=True)

    # Estimate CQLR
    est_sds_q = [
        np.sqrt(np.exp(cv_ret_var_qc["yhat"][:, 0])),
        np.sqrt(np.exp(cal_ret_var_qc["yhat"][:, 0])),
        np.sqrt(np.exp(test_ret_var_qc["yhat"][:, 0])),
    ]
    obs_sds_q = [Vcv_q, Vcal_q, None]

    delta_clr_q = np.quantile(obs_sds_q[1] / est_sds_q[1], 1.0 - alpha)
    qL_cqlr = test_ret_qc["yhat"][:, idx_low] - delta_clr_q * est_sds_q[2]
    qU_cqlr = test_ret_qc["yhat"][:, idx_high] + delta_clr_q * est_sds_q[2]
    coverages[6] = np.mean((ytest[:, 0] <= qU_cqlr) & (ytest[:, 0] >= qL_cqlr))
    lens[6] = np.mean(qU_cqlr - qL_cqlr)

    # LCQR, LCQLR
    tangent_q = cv_ret_var_qc["jacobians"][:, 0, :]
    if tangent_q.ndim == 1 or np.sum(tangent_q ** 2) == 0:
        ut_q = None
        uo_q = np.eye(p)
    else:
        _, _, Vt_q = np.linalg.svd(tangent_q, full_matrices=False)
        ut_q = Vt_q[0, :]
        uo_q = Vt_q[1:, :].T if Vt_q.shape[0] > 1 else None

    Hlists_q = lcp_default_distance(ut_q, uo_q, est_sds_q,
                                    xtrain, xcalibration, xtest)
    min_q = np.quantile(Hlists_q["Hcv"][Hlists_q["Hcv"] > 0], 0.01)
    max_q = np.max(Hlists_q["Hcv"]) * 2.0
    hs_q = np.exp(np.linspace(np.log(min_q), np.log(max_q), 20))

    from .core import auto_tune_distance as _auto_tune

    # --- LCQR ---
    Vcal_lcqr = obs_sds_q[1].copy()
    order_lcqr = np.argsort(Vcal_lcqr)
    lcqr_mod = _LCPModuleSimple(
        H=Hlists_q["H"][np.ix_(order_lcqr, order_lcqr)],
        V=Vcal_lcqr[order_lcqr],
        h=2.0,
        alpha=alpha,
    )
    auto_lcqr = _auto_tune(
        V=obs_sds_q[0], n=len(Vcal_lcqr), hs=hs_q,
        D=Hlists_q["Hcv"], alpha=alpha, delta=alpha / 2.0,
        B=2, trace=False, lmbda=1.0,
    )
    lcqr_mod.h = auto_lcqr["h"]
    lcqr_mod._prepare()
    delta_lcqr = lcqr_mod._construct(
        Hnew=Hlists_q["Hnew"][:, order_lcqr],
        HnewT=Hlists_q["HnewT"][order_lcqr, :],
    )
    qL_lcqr = test_ret_qc["yhat"][:, idx_low] - delta_lcqr
    qU_lcqr = test_ret_qc["yhat"][:, idx_high] + delta_lcqr
    coverages[5] = np.mean((ytest[:, 0] <= qU_lcqr) & (ytest[:, 0] >= qL_lcqr))
    lens[5] = np.mean((qU_lcqr - qL_lcqr)[delta_lcqr < np.inf])
    inf_pct[5] = np.mean(delta_lcqr >= np.inf)

    # --- LCQLR ---
    Vcal_lcqlr = (obs_sds_q[1] / est_sds_q[1]).copy()
    Vcv_lcqlr = (obs_sds_q[0] / est_sds_q[0]).copy()
    order_lcqlr = np.argsort(Vcal_lcqlr)
    lcqlr_mod = _LCPModuleSimple(
        H=Hlists_q["H"][np.ix_(order_lcqlr, order_lcqlr)],
        V=Vcal_lcqlr[order_lcqlr],
        h=2.0,
        alpha=alpha,
    )
    auto_lcqlr = _auto_tune(
        V=Vcv_lcqlr, n=len(Vcal_lcqlr), hs=hs_q,
        D=Hlists_q["Hcv"], alpha=alpha, delta=alpha / 2.0,
        B=2, trace=False, lmbda=1.0,
    )
    lcqlr_mod.h = auto_lcqlr["h"]
    lcqlr_mod._prepare()
    delta_lcqlr = lcqlr_mod._construct(
        Hnew=Hlists_q["Hnew"][:, order_lcqlr],
        HnewT=Hlists_q["HnewT"][order_lcqlr, :],
    )
    qL_lcqlr = test_ret_qc["yhat"][:, idx_low] - delta_lcqlr * est_sds_q[2]
    qU_lcqlr = test_ret_qc["yhat"][:, idx_high] + delta_lcqlr * est_sds_q[2]
    coverages[7] = np.mean((ytest[:, 0] <= qU_lcqlr) & (ytest[:, 0] >= qL_lcqlr))
    lens[7] = np.mean((qU_lcqlr - qL_lcqlr)[delta_lcqlr < np.inf])
    inf_pct[7] = np.mean(delta_lcqlr >= np.inf)
