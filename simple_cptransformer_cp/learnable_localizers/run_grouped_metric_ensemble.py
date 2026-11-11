"""Development-only grouped RBF using three-seed ensemble residuals."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import run_comparison as comparison
import run_grouped_metric as grouped
import run_ensemble_disagreement as ensemble


HERE = Path(__file__).resolve().parent
METHODS = grouped.METHODS + [
    "normalized_fixed_group",
    "normalized_learned_group",
]


def metric_bundle(y, prediction, rows, weights, temperatures):
    result = {
        "point": comparison.previous.base.point_metrics(y, prediction),
        "methods": {},
    }
    target = rows["target_log_error"].to_numpy()
    baseline_mse = float(np.mean((target - rows["split_risk"].to_numpy()) ** 2))
    for method in METHODS:
        lower = rows[f"{method}_lower"].to_numpy()
        upper = rows[f"{method}_upper"].to_numpy()
        risk = rows[f"{method}_risk"].to_numpy()
        risk_mse = float(np.mean((target - risk) ** 2))
        interval = comparison.previous.base.interval_metrics(
            y, lower, upper, comparison.ALPHA
        )
        result["methods"][method] = {
            **interval,
            "miss_lower": int(np.sum(y < lower)),
            "miss_upper": int(np.sum(y > upper)),
            "risk_log_mse": risk_mse,
            "risk_mse_improvement_vs_uniform": (
                0.0 if method == "split" else 1.0 - risk_mse / baseline_mse
            ),
            "risk_spearman": (
                None
                if np.ptp(risk) < 1e-12
                else float(spearmanr(target, risk).statistic)
            ),
            **comparison.weight_diagnostics(weights[method], temperatures[method]),
        }
    return result


def oof_ensemble(x, y, development, fold, device):
    prediction = np.full((len(development), len(ensemble.SEEDS)), np.nan)
    epochs = []
    inner = KFold(4, shuffle=True, random_state=12024 + fold)
    for inner_fold, (train_pos, eval_pos) in enumerate(inner.split(development)):
        matrix, selected = ensemble.train_ensemble(
            x,
            y,
            development[train_pos],
            development[eval_pos],
            22024 + 100 * fold + inner_fold,
            device,
        )
        prediction[eval_pos] = matrix
        epochs.append({"role": f"inner_{inner_fold}", "epochs": selected})
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Missing ensemble OOF predictions")
    return prediction, epochs


def evaluate_fold(x, y, names, raw_features, fold, outer_train, outer_eval,
                  device, output):
    development, calibration, evaluation = grouped.fold_partition(
        len(x), fold, outer_train, outer_eval, calibration_size=20
    )
    dev_matrix, epochs = oof_ensemble(x, y, development, fold, device)
    predict_ids = np.concatenate([calibration, evaluation])
    external_matrix, selected = ensemble.train_ensemble(
        x, y, development, predict_ids, 32024 + fold, device
    )
    epochs.append({"role": "final", "epochs": selected})
    dev_prediction = dev_matrix.mean(axis=1)
    external_prediction = external_matrix.mean(axis=1)
    cal_prediction = external_prediction[: len(calibration)]
    eval_prediction = external_prediction[len(calibration):]
    dev_log_error = np.log1p(np.abs(y[development] - dev_prediction)).astype(np.float32)
    cal_error = np.abs(y[calibration] - cal_prediction)
    cal_log_error = np.log1p(cal_error)
    eval_log_error = np.log1p(np.abs(y[evaluation] - eval_prediction))

    median, scale = comparison.fit_robust_scaler(raw_features[development])
    features = comparison.apply_scaler(raw_features, median, scale)
    learned_weights, learned_bandwidth, training = grouped.fit_group_metric(
        features[development], dev_log_error, 424242 + fold, device
    )
    fixed_group_bandwidth = grouped.median_bandwidth(features[development])
    rank = math.ceil((len(calibration) + 1) * (1 - comparison.ALPHA))
    level = min(1.0, rank / len(calibration))
    uniform = np.full(
        (len(evaluation), len(calibration)), 1.0 / len(calibration), dtype=float
    )
    weights = {"split": uniform}
    temperatures = {"split": np.ones(len(evaluation), dtype=float)}

    pca_z, pca_bandwidth = comparison.fixed_rbf_fit(features, development)
    pca_distance2 = np.sum(
        (pca_z[evaluation, None] - pca_z[None, calibration]) ** 2, axis=2
    )
    weights["fixed_rbf_pca"], temperatures["fixed_rbf_pca"] = (
        comparison.weights_with_min_ess(-pca_distance2 / (2 * pca_bandwidth**2))
    )
    weights["fixed_group_rbf"], temperatures["fixed_group_rbf"] = (
        comparison.weights_with_min_ess(
            grouped.grouped_logits(
                features[evaluation],
                features[calibration],
                np.ones(len(grouped.GROUP_NAMES)),
                fixed_group_bandwidth,
            )
        )
    )
    weights["learned_group_rbf"], temperatures["learned_group_rbf"] = (
        comparison.weights_with_min_ess(
            grouped.grouped_logits(
                features[evaluation],
                features[calibration],
                learned_weights,
                learned_bandwidth,
            )
        )
    )

    normalized = {}
    for method, group_weights, bandwidth in [
        ("normalized_fixed_group", np.ones(len(grouped.GROUP_NAMES)), fixed_group_bandwidth),
        ("normalized_learned_group", learned_weights, learned_bandwidth),
    ]:
        cal_weights, _ = comparison.weights_with_min_ess(
            grouped.grouped_logits(
                features[calibration], features[development], group_weights, bandwidth
            )
        )
        eval_weights, eval_temperatures = comparison.weights_with_min_ess(
            grouped.grouped_logits(
                features[evaluation], features[development], group_weights, bandwidth
            )
        )
        cal_log_risk = cal_weights @ dev_log_error
        eval_log_risk = eval_weights @ dev_log_error
        cal_scale = np.maximum(1.0, np.expm1(cal_log_risk))
        eval_scale = np.maximum(1.0, np.expm1(eval_log_risk))
        q_normalized, normalized_rank = comparison.previous.base.conformal_upper_quantile(
            cal_error / cal_scale, comparison.ALPHA
        )
        normalized[method] = {
            "risk": eval_log_risk,
            "radius": q_normalized * eval_scale,
            "q_normalized": q_normalized,
            "rank": normalized_rank,
        }
        weights[method] = eval_weights
        temperatures[method] = eval_temperatures

    rows = pd.DataFrame(
        {
            "cell_id": [names[index][:-4] for index in evaluation],
            "outer_fold": fold,
            "eol": y[evaluation],
            "prediction": eval_prediction,
            "prediction_seed_std": external_matrix[len(calibration):].std(axis=1),
            "target_log_error": eval_log_error,
        }
    )
    for method in grouped.METHODS:
        risk = weights[method] @ cal_log_error
        q = comparison.weighted_quantiles(cal_error, weights[method], level)
        rows[f"{method}_risk"] = risk
        rows[f"{method}_q"] = q
        rows[f"{method}_lower"] = eval_prediction - q
        rows[f"{method}_upper"] = eval_prediction + q

    for method, values in normalized.items():
        radius = values["radius"]
        rows[f"{method}_risk"] = values["risk"]
        rows[f"{method}_q"] = radius
        rows[f"{method}_lower"] = eval_prediction - radius
        rows[f"{method}_upper"] = eval_prediction + radius

    metrics = metric_bundle(
        y[evaluation], eval_prediction, rows, weights, temperatures
    )
    metrics.update(
        {
            "outer_fold": fold,
            "development_n": len(development),
            "calibration_n": len(calibration),
            "evaluation_n": len(evaluation),
            "conformal_rank": rank,
            "weighted_quantile_level": level,
            "pca_bandwidth": pca_bandwidth,
            "fixed_group_bandwidth": fixed_group_bandwidth,
            "learned_group_bandwidth": learned_bandwidth,
            "metric_training": training,
            "normalized_quantiles": {
                method: {
                    "q": values["q_normalized"],
                    "rank": values["rank"],
                }
                for method, values in normalized.items()
            },
            "predictor_epochs": epochs,
        }
    )
    fold_dir = output / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=False)
    rows.to_csv(fold_dir / "evaluation.csv", index=False)
    comparison.save_json(fold_dir / "metrics.json", metrics)
    comparison.save_json(
        fold_dir / "split.json",
        {
            "development": [names[index] for index in development],
            "calibration": [names[index] for index in calibration],
            "evaluation": [names[index] for index in evaluation],
        },
    )
    print(f"ensemble grouped-metric fold {fold + 1}/5", flush=True)
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grouped.METRIC_L2 = 0.0
    output = HERE / "runs_grouped_metric_ensemble" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "experiment": "E: six-weight grouped RBF on three-seed ensemble residuals",
        "T": 100,
        "dataset": "MATR official train only",
        "ensemble_seeds": ensemble.SEEDS,
        "outer_folds": 5,
        "calibration_per_fold": 20,
        "alpha": comparison.ALPHA,
        "methods": METHODS,
        "group_names": grouped.GROUP_NAMES,
        "metric_parameter_count": len(grouped.GROUP_NAMES),
        "metric_l2": grouped.METRIC_L2,
        "metric_loss": "CRPS of leave-one-out ensemble log absolute residual distribution",
        "min_ess": comparison.MIN_ESS,
        "predictor": vars(comparison.previous.CFG),
        "device": device,
        "data_root": str(args.data_root.resolve()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_usage": "MATR validation and test are not loaded",
        "weighted_interval_status": (
            "empirical basic-LCP diagnostic; no new local guarantee claimed"
        ),
    }
    comparison.save_json(output / "config.json", config)
    print(f"RUN_DIR={output}", flush=True)

    names = list(comparison.previous.base.split_recorder.MATR_train_files)
    x, y, audit = comparison.previous.load_data(args.data_root, names)
    comparison.save_json(output / "data_audit.json", audit)
    raw_features = comparison.time_features(x)
    outer = list(KFold(5, shuffle=True, random_state=2024).split(x))
    rows, fold_metrics = [], []
    for fold, (outer_train, outer_eval) in enumerate(outer):
        fold_rows, metrics = evaluate_fold(
            x,
            y,
            names,
            raw_features,
            fold,
            outer_train,
            outer_eval,
            device,
            output,
        )
        rows.append(fold_rows)
        fold_metrics.append(metrics)
    combined = pd.concat(rows, ignore_index=True)
    if len(combined) != 102 or combined["cell_id"].nunique() != 102:
        raise RuntimeError("Outer evaluation must cover every MATR train cell once")
    combined.to_csv(output / "evaluation.csv", index=False)
    metrics = {
        "point": comparison.previous.base.point_metrics(
            combined["eol"].to_numpy(), combined["prediction"].to_numpy()
        ),
        "methods": {},
    }
    target = combined["target_log_error"].to_numpy()
    baseline_mse = float(
        np.mean((target - combined["split_risk"].to_numpy()) ** 2)
    )
    for method in METHODS:
        lower = combined[f"{method}_lower"].to_numpy()
        upper = combined[f"{method}_upper"].to_numpy()
        risk = combined[f"{method}_risk"].to_numpy()
        risk_mse = float(np.mean((target - risk) ** 2))
        interval = comparison.previous.base.interval_metrics(
            combined["eol"].to_numpy(), lower, upper, comparison.ALPHA
        )
        correlations = None if np.ptp(risk) < 1e-12 else float(
            spearmanr(target, risk).statistic
        )
        metrics["methods"][method] = {
            **interval,
            "miss_lower": int(np.sum(combined["eol"].to_numpy() < lower)),
            "miss_upper": int(np.sum(combined["eol"].to_numpy() > upper)),
            "risk_log_mse": risk_mse,
            "risk_mse_improvement_vs_uniform": (
                0.0 if method == "split" else 1.0 - risk_mse / baseline_mse
            ),
            "risk_spearman": correlations,
            "positive_risk_folds": int(
                sum(
                    fold["methods"][method]["risk_log_mse"]
                    < fold["methods"]["split"]["risk_log_mse"]
                    for fold in fold_metrics
                )
            ),
            "ess_mean": float(
                np.mean([fold["methods"][method]["ess_mean"] for fold in fold_metrics])
            ),
        }
    metrics["learned_group_weights_mean"] = {
        name: float(
            np.mean(
                [fold["metric_training"]["group_weights"][name] for fold in fold_metrics]
            )
        )
        for name in grouped.GROUP_NAMES
    }
    comparison.save_json(output / "summary.json", metrics)
    print(f"SUMMARY: {json.dumps(metrics)}", flush=True)


if __name__ == "__main__":
    main()
