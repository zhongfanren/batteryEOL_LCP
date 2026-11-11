"""Development-only low-capacity learnable RBF metric on MATR train cells.

The 57 time features are divided into six interpretable groups.  The learned
localizer has only six non-negative group weights and is trained on OOF point
prediction residuals.  Predictor development, conformal calibration, and outer
evaluation cells are disjoint in every fold.
"""
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
from torch import nn

import run_comparison as comparison


HERE = Path(__file__).resolve().parent
SEEDS = [42, 2021, 2024]
METHODS = ["split", "fixed_rbf_pca", "fixed_group_rbf", "learned_group_rbf"]
GROUP_NAMES = [
    "voltage_trend",
    "current_trend",
    "capacity_trend",
    "channel_std",
    "cycle_difference_std",
    "slope",
]
METRIC_L2 = 0.1


def feature_groups():
    return [
        np.arange(0, 48, 3),
        np.arange(1, 48, 3),
        np.arange(2, 48, 3),
        np.arange(48, 51),
        np.arange(51, 54),
        np.arange(54, 57),
    ]


def group_distance_tensor(query, reference):
    distances = []
    for indices in feature_groups():
        index = torch.tensor(indices, dtype=torch.long, device=query.device)
        difference = query[:, None, index] - reference[None, :, index]
        distances.append(torch.mean(difference**2, dim=2))
    return torch.stack(distances, dim=2)


def group_distance_numpy(query, reference):
    distances = []
    for indices in feature_groups():
        difference = query[:, None, indices] - reference[None, :, indices]
        distances.append(np.mean(difference**2, axis=2))
    return np.stack(distances, axis=2)


def median_bandwidth(features):
    group_distance = group_distance_numpy(features, features)
    distance = np.sqrt(np.sum(group_distance, axis=2))
    upper = distance[np.triu_indices(len(features), 1)]
    return max(float(np.median(upper[upper > 0])), 1e-8)


class GroupMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(len(GROUP_NAMES)))

    def weights(self):
        return len(GROUP_NAMES) * torch.softmax(self.raw_weights, dim=0)

    def logits(self, query, reference, bandwidth):
        distance = group_distance_tensor(query, reference)
        distance2 = torch.sum(distance * self.weights()[None, None, :], dim=2)
        return -distance2 / (2 * bandwidth**2)


def metric_crps(model, query_x, reference_x, query_y, reference_y, bandwidth,
                exclude_self=False, include_regularization=True):
    logits = model.logits(query_x, reference_x, bandwidth)
    if exclude_self:
        logits = logits.masked_fill(
            torch.eye(len(query_x), dtype=torch.bool, device=query_x.device),
            float("-inf"),
        )
    weights = torch.softmax(logits, dim=1)
    residual_loss = comparison.localizer_loss(weights, query_y, reference_y, "crps")
    ess = 1.0 / torch.sum(weights**2, dim=1)
    ess_penalty = torch.mean(torch.relu(comparison.MIN_ESS - ess) ** 2) / (
        comparison.MIN_ESS**2
    )
    regularization = torch.mean((model.weights() - 1.0) ** 2)
    penalty = METRIC_L2 * regularization if include_regularization else 0.0
    return residual_loss + 0.05 * ess_penalty + penalty


def fit_group_metric(features, target, seed, device):
    comparison.previous.base.seed_all(seed)
    order = np.random.default_rng(seed).permutation(len(features))
    n_validation = max(8, int(round(0.2 * len(order))))
    validation, fitting = order[:n_validation], order[n_validation:]
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(target, dtype=torch.float32, device=device)
    fitting_tensor = torch.tensor(fitting, dtype=torch.long, device=device)
    validation_tensor = torch.tensor(validation, dtype=torch.long, device=device)
    selection_bandwidth = median_bandwidth(features[fitting])

    model = GroupMetric().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=comparison.LOCALIZER_LR,
        weight_decay=comparison.LOCALIZER_WEIGHT_DECAY,
    )
    best_loss, best_epoch, stale = float("inf"), 1, 0
    for epoch in range(1, comparison.LOCALIZER_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = metric_crps(
            model,
            x[fitting_tensor],
            x[fitting_tensor],
            y[fitting_tensor],
            y[fitting_tensor],
            selection_bandwidth,
            exclude_self=True,
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                metric_crps(
                    model,
                    x[validation_tensor],
                    x[fitting_tensor],
                    y[validation_tensor],
                    y[fitting_tensor],
                    selection_bandwidth,
                    include_regularization=False,
                ).cpu()
            )
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= comparison.LOCALIZER_PATIENCE:
                break

    comparison.previous.base.seed_all(seed)
    model = GroupMetric().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=comparison.LOCALIZER_LR,
        weight_decay=comparison.LOCALIZER_WEIGHT_DECAY,
    )
    refit_bandwidth = median_bandwidth(features)
    for _ in range(best_epoch):
        model.train()
        optimizer.zero_grad()
        loss = metric_crps(
            model, x, x, y, y, refit_bandwidth, exclude_self=True
        )
        loss.backward()
        optimizer.step()
    learned = model.weights().detach().cpu().numpy()
    return learned, refit_bandwidth, {
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "group_weights": dict(zip(GROUP_NAMES, learned.tolist())),
    }


def grouped_logits(query, reference, group_weights, bandwidth):
    distance = group_distance_numpy(query, reference)
    distance2 = np.sum(distance * np.asarray(group_weights)[None, None, :], axis=2)
    return -distance2 / (2 * bandwidth**2)


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


def fold_partition(n, fold, outer_train, outer_eval, calibration_size):
    pool = np.random.default_rng(2024 + fold).permutation(outer_train)
    calibration, development = pool[:calibration_size], pool[calibration_size:]
    if len(development) < 16:
        raise ValueError("Too few development cells after calibration split")
    if set(development) & set(calibration):
        raise RuntimeError("Development and calibration overlap")
    if set(development) & set(outer_eval) or set(calibration) & set(outer_eval):
        raise RuntimeError("Outer evaluation overlaps development/calibration")
    if len(development) + len(calibration) + len(outer_eval) != n:
        raise RuntimeError("Fold partition does not cover all cells")
    return development, calibration, outer_eval


def evaluate_fold(x, y, names, raw_features, seed, fold, outer_train, outer_eval,
                  device, output, calibration_size, outer_folds):
    development, calibration, evaluation = fold_partition(
        len(x), fold, outer_train, outer_eval, calibration_size
    )
    dev_oof, predictor_epochs = comparison.oof_development_predictions(
        x, y, development, seed, fold, device
    )
    predict_ids = np.concatenate([calibration, evaluation])
    external_prediction, epoch = comparison.robust.train_and_predict(
        x, y, development, predict_ids, seed, 32024 + fold, device
    )
    predictor_epochs.append({"role": "final", "epoch": epoch})
    cal_prediction = external_prediction[: len(calibration)]
    eval_prediction = external_prediction[len(calibration):]
    dev_log_error = np.log1p(np.abs(y[development] - dev_oof)).astype(np.float32)
    cal_error = np.abs(y[calibration] - cal_prediction)
    cal_log_error = np.log1p(cal_error)
    eval_log_error = np.log1p(np.abs(y[evaluation] - eval_prediction))

    median, scale = comparison.fit_robust_scaler(raw_features[development])
    features = comparison.apply_scaler(raw_features, median, scale)
    metric_seed = seed * 1000 + fold
    learned_weights, learned_bandwidth, training = fit_group_metric(
        features[development], dev_log_error, metric_seed, device
    )
    fixed_group_bandwidth = median_bandwidth(features[development])

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
            grouped_logits(
                features[evaluation],
                features[calibration],
                np.ones(len(GROUP_NAMES)),
                fixed_group_bandwidth,
            )
        )
    )
    weights["learned_group_rbf"], temperatures["learned_group_rbf"] = (
        comparison.weights_with_min_ess(
            grouped_logits(
                features[evaluation],
                features[calibration],
                learned_weights,
                learned_bandwidth,
            )
        )
    )

    rows = pd.DataFrame(
        {
            "cell_id": [names[index][:-4] for index in evaluation],
            "seed": seed,
            "outer_fold": fold,
            "eol": y[evaluation],
            "prediction": eval_prediction,
            "target_log_error": eval_log_error,
        }
    )
    for method in METHODS:
        risk = weights[method] @ cal_log_error
        q = comparison.weighted_quantiles(cal_error, weights[method], level)
        rows[f"{method}_risk"] = risk
        rows[f"{method}_q"] = q
        rows[f"{method}_lower"] = eval_prediction - q
        rows[f"{method}_upper"] = eval_prediction + q

    metrics = metric_bundle(y[evaluation], eval_prediction, rows, weights, temperatures)
    metrics.update(
        {
            "seed": seed,
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
            "predictor_epochs": predictor_epochs,
        }
    )
    fold_dir = output / f"seed_{seed}" / f"fold_{fold}"
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
    print(f"seed {seed}: grouped-metric fold {fold + 1}/{outer_folds}", flush=True)
    return rows, metrics


def aggregate_seed(rows, fold_metrics):
    y = rows["eol"].to_numpy()
    prediction = rows["prediction"].to_numpy()
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
            "positive_risk_folds": int(
                sum(
                    record["methods"][method]["risk_log_mse"]
                    < record["methods"]["split"]["risk_log_mse"]
                    for record in fold_metrics
                )
            ),
            "ess_mean": float(
                np.mean([record["methods"][method]["ess_mean"] for record in fold_metrics])
            ),
        }
    result["learned_group_weights_mean"] = {
        name: float(
            np.mean(
                [record["metric_training"]["group_weights"][name] for record in fold_metrics]
            )
        )
        for name in GROUP_NAMES
    }
    return result


def aggregate_all(seed_metrics):
    result = {}
    for method in METHODS:
        values = [record["methods"][method] for record in seed_metrics]
        correlations = [
            value["risk_spearman"]
            for value in values
            if value["risk_spearman"] is not None
        ]
        result[method] = {
            "coverage_mean": float(np.mean([value["coverage"] for value in values])),
            "mean_width_mean": float(np.mean([value["mean_width"] for value in values])),
            "median_width_mean": float(
                np.mean([value["median_width"] for value in values])
            ),
            "interval_score_mean": float(
                np.mean([value["mean_interval_score"] for value in values])
            ),
            "risk_improvement_mean": float(
                np.mean([value["risk_mse_improvement_vs_uniform"] for value in values])
            ),
            "risk_spearman_mean": (
                None if not correlations else float(np.mean(correlations))
            ),
            "seeds_with_positive_risk_improvement": int(
                sum(value["risk_mse_improvement_vs_uniform"] > 0 for value in values)
            ),
            "positive_risk_folds_mean": float(
                np.mean([value["positive_risk_folds"] for value in values])
            ),
            "ess_mean": float(np.mean([value["ess_mean"] for value in values])),
        }
    result["learned_group_weights_mean"] = {
        name: float(
            np.mean([record["learned_group_weights_mean"][name] for record in seed_metrics])
        )
        for name in GROUP_NAMES
    }
    return result


def main():
    global METRIC_L2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--metric-l2", type=float, default=METRIC_L2)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--calibration-size", type=int, default=20)
    args = parser.parse_args()
    if args.metric_l2 < 0:
        raise ValueError("--metric-l2 must be non-negative")
    if args.outer_folds < 2 or args.calibration_size < 1:
        raise ValueError("Invalid fold or calibration size")
    METRIC_L2 = args.metric_l2
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = HERE / "runs_grouped_metric" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "experiment": "D: development-only six-weight learnable grouped RBF",
        "T": 100,
        "dataset": "MATR official train only",
        "outer_folds": args.outer_folds,
        "outer_split_seed": 2024,
        "calibration_per_fold": args.calibration_size,
        "seeds": args.seeds,
        "alpha": comparison.ALPHA,
        "methods": METHODS,
        "group_names": GROUP_NAMES,
        "metric_parameter_count": len(GROUP_NAMES),
        "metric_l2_toward_equal_weights": METRIC_L2,
        "selection_loss": "held-out CRPS plus ESS penalty; excludes training L2",
        "metric_loss": "CRPS of leave-one-out log absolute residual distribution",
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
    outer = list(KFold(args.outer_folds, shuffle=True, random_state=2024).split(x))
    seed_metrics = []
    for seed in args.seeds:
        rows_for_seed, metrics_for_seed = [], []
        for fold, (outer_train, outer_eval) in enumerate(outer):
            rows, metrics = evaluate_fold(
                x,
                y,
                names,
                raw_features,
                seed,
                fold,
                outer_train,
                outer_eval,
                device,
                output,
                args.calibration_size,
                args.outer_folds,
            )
            rows_for_seed.append(rows)
            metrics_for_seed.append(metrics)
        combined = pd.concat(rows_for_seed, ignore_index=True)
        if len(combined) != 102 or combined["cell_id"].nunique() != 102:
            raise RuntimeError("Outer evaluation must cover every MATR train cell once")
        seed_dir = output / f"seed_{seed}"
        combined.to_csv(seed_dir / "evaluation.csv", index=False)
        metrics = aggregate_seed(combined, metrics_for_seed)
        comparison.save_json(seed_dir / "metrics.json", metrics)
        seed_metrics.append(metrics)
        print(f"SEED {seed}: {json.dumps(metrics['methods'])}", flush=True)
    summary = aggregate_all(seed_metrics)
    comparison.save_json(output / "summary.json", summary)
    print(f"SUMMARY: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
