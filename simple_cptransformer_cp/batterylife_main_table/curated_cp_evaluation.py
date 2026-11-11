"""CPU-only CP diagnostics from frozen curated-pilot predictions.

RLCP follows Hore & Barber, arXiv:2310.07850, equations (13), (15).
All feature scales and method settings are fixed without calibration/test labels.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ALPHA = 0.1
BANDWIDTH = 1.5
RANDOM_SEED = 20261027


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        return x
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def weighted_radius(scores, calibration_weights, test_weight, alpha=ALPHA):
    """Weighted quantile including the candidate test score at +infinity."""
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(calibration_weights, dtype=float)
    if np.any(weights < 0) or not np.all(np.isfinite(weights)) or test_weight < 0:
        raise ValueError("Invalid weights")
    order = np.argsort(scores, kind="stable")
    threshold = (1 - alpha) * (weights.sum() + test_weight)
    index = np.searchsorted(np.cumsum(weights[order]), threshold, side="left")
    return float(scores[order[index]]) if index < len(scores) else float("inf")


def local_radii(cal_z, test_z, scores, randomized, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    radii, effective_sizes = [], []
    for z in test_z:
        center = z + rng.normal(0, BANDWIDTH, size=len(z)) if randomized else z
        log_cal = -np.sum((cal_z - center) ** 2, axis=1) / (2 * BANDWIDTH**2)
        log_test = -np.sum((z - center) ** 2) / (2 * BANDWIDTH**2)
        # The Gaussian normalizing constant cancels; log scaling avoids underflow.
        offset = max(float(log_cal.max()), float(log_test))
        weights, test_weight = np.exp(log_cal - offset), np.exp(log_test - offset)
        radii.append(weighted_radius(scores, weights, test_weight))
        effective_sizes.append(float(weights.sum() ** 2 / np.sum(weights**2)))
    return np.asarray(radii), np.asarray(effective_sizes)


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return None
    return float(spearmanr(a[keep], b[keep]).statistic)


def interval_metrics(rows, name):
    lower, upper = rows[f"{name}_lower"].to_numpy(), rows[f"{name}_upper"].to_numpy()
    y = rows.eol.to_numpy()
    width = upper - lower
    finite = np.isfinite(width)
    score = width + 2 / ALPHA * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
    return {"n_cells": len(rows), "covered": int(((y >= lower) & (y <= upper)).sum()),
            "coverage": float(np.mean((y >= lower) & (y <= upper))),
            "mean_width_cycles": float(np.mean(width)),
            "median_width_cycles": float(np.median(width)),
            "mean_interval_score": float(np.mean(score)),
            "infinite_intervals": int((~finite).sum()),
            "finite_only_mean_width_cycles": float(np.mean(width[finite])) if finite.any() else None,
            "width_error_spearman_finite_only": correlation(width, rows.absolute_error),
            "finite_width_correlation_n": int(finite.sum())}


def point_metrics(rows):
    error = np.abs(rows.eol.to_numpy() - rows.prediction.to_numpy())
    return {"n_cells": len(rows), "rmse_cycles": float(np.sqrt(np.mean(error**2))),
            "mae_cycles": float(np.mean(error)),
            "mape_fraction": float(np.mean(error / rows.eol.to_numpy())),
            "accuracy_15_fraction": float(np.mean(error / rows.eol.to_numpy() <= 0.15))}


def evaluate_frozen(out, manifest, checkpoint):
    out = Path(out)
    tables = {p: pd.read_csv(out / f"{p}_predictions.csv") for p in ("dev", "calibration", "test")}
    metadata = {row["cell_file"]: row for row in manifest["cells"]}
    train_meta = [metadata[name]["metadata"] for name in manifest["split"]["phases"]["train"]]

    def rates(meta):
        return [float(meta[k][0]["rate_in_C"]) for k in ("charge_protocol", "discharge_protocol")]

    train_rates = np.asarray([rates(m) for m in train_meta])
    rate_mean, rate_std = train_rates.mean(axis=0), train_rates.std(axis=0)
    rate_std = np.maximum(rate_std, 1e-6)
    label_mean, label_std = checkpoint["label_mean"], max(checkpoint["label_std"], 1e-6)
    features = {}
    for phase, rows in tables.items():
        if set(rows.cell_file) != set(manifest["split"]["phases"][phase]) or not rows.cell_file.is_unique:
            raise RuntimeError(f"Prediction membership mismatch for {phase}")
        rows["raw_prediction"] = rows.prediction
        rows["prediction"] = np.maximum(0, rows.prediction)
        rows["absolute_error"] = np.abs(rows.eol - rows.prediction)
        rows["condition"] = rows.cell_file.map(lambda n: metadata[n]["condition"])
        pairs = np.asarray([rates(metadata[n]["metadata"]) for n in rows.cell_file])
        rows["charge_rate_C"], rows["discharge_rate_C"] = pairs[:, 0], pairs[:, 1]
        features[phase] = np.column_stack(((rows.prediction.to_numpy() - label_mean) / label_std,
                                          (pairs - rate_mean) / rate_std))

    # Fixed kernel regression of |error| on only development residuals.  This is
    # a noisy expected-error proxy, not the true conditional 90th percentile.
    dev, cal, test = (tables[p] for p in ("dev", "calibration", "test"))
    dev_errors = dev.absolute_error.to_numpy()
    floor = max(1.0, float(np.median(dev_errors)) * 0.1)

    def difficulty(z, leave_one_out=False):
        log_weights = -np.sum((z[:, None, :] - features["dev"][None, :, :])**2, axis=2) / (2 * BANDWIDTH**2)
        if leave_one_out:
            np.fill_diagonal(log_weights, -np.inf)
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        return np.maximum(floor, weights @ dev_errors / weights.sum(axis=1))

    dev_risk = difficulty(features["dev"], leave_one_out=True)
    cutpoints = np.quantile(dev_risk, [1/3, 2/3])
    for phase, rows in tables.items():
        rows["predicted_difficulty"] = difficulty(features[phase])
        rows["difficulty_group"] = np.asarray(["low", "middle", "high"])[
            np.searchsorted(cutpoints, rows.predicted_difficulty.to_numpy(), side="right")]

    cal_scores = cal.absolute_error.to_numpy()
    split_q = weighted_radius(cal_scores, np.ones(len(cal)), 1)
    normalized_q = weighted_radius(cal_scores / cal.predicted_difficulty.to_numpy(), np.ones(len(cal)), 1)
    radii = {"split_cp": np.full(len(test), split_q),
             "normalized_cp": normalized_q * test.predicted_difficulty.to_numpy()}
    method_settings = {
        "split_cp": {"construction": "absolute residual split CP"},
        "normalized_cp": {"construction": "split CP on residual / dev-fitted difficulty"},
    }
    for name, dims, randomized in (("base_lcp_protocol", 3, False),
                                    ("rlcp_prediction", 1, True),
                                    ("rlcp_protocol", 3, True)):
        radii[name], ess = local_radii(features["calibration"][:, :dims], features["test"][:, :dims], cal_scores, randomized)
        test[f"{name}_ess"] = ess
        method_settings[name] = {"construction": "RLCP" if randomized else "exploratory baseLCP; no marginal guarantee",
                                 "feature_dimensions": dims, "bandwidth": BANDWIDTH,
                                 "random_seed": RANDOM_SEED if randomized else None,
                                 "mean_calibration_ess": float(ess.mean()), "min_calibration_ess": float(ess.min())}
    for name, radius in radii.items():
        test[f"{name}_radius"] = radius
        test[f"{name}_lower"] = np.maximum(0, test.prediction.to_numpy() - radius)
        test[f"{name}_upper"] = test.prediction.to_numpy() + radius

    metrics = {name: {**method_settings[name], **interval_metrics(test, name)} for name in radii}
    groups = []
    for group_col in ("difficulty_group", "condition"):
        for group_name, group in test.groupby(group_col):
            for name in radii:
                groups.append({"group_kind": group_col, "group": group_name, "method": name,
                               "mean_absolute_error": float(group.absolute_error.mean()),
                               **interval_metrics(group, name)})
    pd.DataFrame(groups).to_csv(out / "group_diagnostics.csv", index=False)
    cal.to_csv(out / "calibration_scores.csv", index=False)
    test.to_csv(out / "test_intervals.csv", index=False)
    risk_correlation = correlation(test.predicted_difficulty, test.absolute_error)
    constant_baselines = {}
    for name, value in (("train_mean", label_mean),
                        ("train_median", checkpoint.get("label_median", label_mean))):
        baseline = test.copy()
        baseline["prediction"] = value
        constant_baselines[name] = {"value_cycles": value, **point_metrics(baseline)}
    result = {"test_exploratory": True, "alpha": ALPHA, "prediction_cycle": 100,
              "prediction_postprocessing": "max(0, raw_prediction) for all CP scores and reported point metrics; raw values retained",
              "point": {p: point_metrics(rows) for p, rows in tables.items()},
              "constant_point_baselines": constant_baselines,
              "methods": metrics, "difficulty_error_spearman": risk_correlation,
              "difficulty_definition": "Gaussian regression of development absolute errors; fixed feature scales and bandwidth; not conditional-tail oracle",
              "difficulty_group_cutpoints_from_dev": cutpoints.tolist(),
              "scales": {"prediction_mean": label_mean, "prediction_std": label_std,
                         "rate_mean": rate_mean.tolist(), "rate_std": rate_std.tolist()},
              "split_cp_rank": math.ceil((len(cal) + 1) * (1 - ALPHA)),
              "null_metric_meaning": "nonfinite (e.g. an infinite interval) or undefined; inspect infinite_intervals",
              "coverage_scope": "Split/normalized CP: marginal under exchangeability and frozen scoring. RLCP: marginal under paper assumptions, including randomization. No per-condition or pointwise conditional guarantee; historical test exposure makes study exploratory.",
              "rlcp_reference": "https://arxiv.org/html/2310.07850#S3"}
    write_json(out / "results.json", result)
    lines = ["# Curated ISU-ILCC exploratory result", "",
             f"T=100 test MAPE: {result['point']['test']['mape_fraction']:.2%}; n={len(test)}.",
             f"Dev-fitted difficulty vs test |error| Spearman: {risk_correlation}.", "",
             "| Method | Coverage | Mean width (cycles) | Interval score | Infinite |",
             "|---|---:|---:|---:|---:|"]
    for name, values in metrics.items():
        lines.append(f"| {name} | {values['coverage']:.2%} | {values['mean_width_cycles']:.2f} | {values['mean_interval_score']:.2f} | {values['infinite_intervals']} |")
    lines += ["", "Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.",
              "Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULTS={out / 'results.json'}", flush=True)
    return result
