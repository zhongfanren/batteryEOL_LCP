"""Experiment A: independent fixed-T=100 replication on HUST.

The official HUST train split develops the CPTransformer and localizer, the
official validation split calibrates intervals, and the official test split is
used once for evaluation.  Hyperparameters are inherited unchanged from the
MATR localizer comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import run_comparison as comparison


HERE = Path(__file__).resolve().parent
METHODS = ["split", "fixed_rbf", "hopfield_time"]
SEEDS = [42, 2021, 2024]


def load_hust_data(root: Path, names: list[str]):
    """Load fixed-T=100 tensors and verify them against upstream preprocessing."""
    labels_path = root / "Life labels" / "HUST_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(
        Path(comparison.previous.base.__file__).read_bytes()
        + (comparison.previous.base.UPSTREAM / "data_provider/data_loader.py").read_bytes()
    ).hexdigest()[:16]
    cache = HERE / "cache" / "HUST" / source_hash
    cache.mkdir(parents=True, exist_ok=True)
    reader = comparison.previous.upstream_reader(root)
    xs, ys, audit = [], [], []
    for index, name in enumerate(names):
        path = root / "HUST" / name
        if name not in labels or not path.exists():
            raise ValueError(f"Missing source/label for {name}; official split must not change")
        eol = float(labels[name])
        if not np.isfinite(eol) or eol <= 100:
            raise ValueError(f"Ineligible EOL for {name}: {eol}")
        stat = path.stat()
        key = hashlib.sha256(
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{eol}".encode()
        ).hexdigest()[:16]
        target = cache / f"{path.stem}_{key}.npz"
        if target.exists():
            with np.load(target) as record:
                x = record["x"]
                delta = float(record["max_abs_difference"])
        else:
            x = comparison.previous.base.load_cell(path, eol)
            with path.open("rb") as handle:
                raw = pickle.load(handle)
            reader.read_cell_data_according_to_prefix = (
                lambda _, raw=raw, eol=eol: (raw, eol)
            )
            original = reader.read_cell_df(name)[1].astype(np.float32)
            delta = float(np.max(np.abs(x - original)))
            np.testing.assert_allclose(x, original, rtol=1e-6, atol=1e-6, err_msg=name)
            np.savez_compressed(target, x=x, max_abs_difference=delta)
        if x.shape != (100, 3, 300) or not np.isfinite(x).all():
            raise ValueError(f"Invalid fixed-T tensor for {name}: {x.shape}")
        xs.append(x)
        ys.append(eol)
        audit.append({"cell_id": path.stem, "max_abs_difference": delta})
        if (index + 1) % 20 == 0 or index + 1 == len(names):
            print(f"data {index + 1}/{len(names)} verified", flush=True)
    return np.stack(xs), np.asarray(ys, dtype=float), audit


def metric_bundle(y, prediction, rows, weights, temperatures, cal_log_error):
    result = {"point": comparison.previous.base.point_metrics(y, prediction), "methods": {}}
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
                None if np.ptp(risk) < 1e-12
                else float(spearmanr(target, risk).statistic)
            ),
            **comparison.weight_diagnostics(weights[method], temperatures[method]),
        }
    return result


def run_seed(x, y, names, development, calibration, evaluation, seed, device, out):
    dev_oof, epochs = comparison.oof_development_predictions(
        x, y, development, seed, fold=0, device=device
    )
    predict_ids = np.concatenate([calibration, evaluation])
    external_prediction, epoch = comparison.robust.train_and_predict(
        x, y, development, predict_ids, seed, 32024, device
    )
    epochs.append({"role": "final", "epoch": epoch})
    cal_prediction = external_prediction[: len(calibration)]
    test_prediction = external_prediction[len(calibration) :]
    dev_log_error = np.log1p(np.abs(y[development] - dev_oof)).astype(np.float32)
    cal_error = np.abs(y[calibration] - cal_prediction)
    cal_log_error = np.log1p(cal_error)
    test_log_error = np.log1p(np.abs(y[evaluation] - test_prediction))

    raw_features = comparison.time_features(x)
    median, scale = comparison.fit_robust_scaler(raw_features[development])
    features = comparison.apply_scaler(raw_features, median, scale)
    localizer, training = comparison.fit_localizer(
        features[development], dev_log_error, seed * 1000, device, "crps"
    )

    rank = math.ceil((len(calibration) + 1) * (1 - comparison.ALPHA))
    level = min(1.0, rank / len(calibration))
    uniform = np.full(
        (len(evaluation), len(calibration)), 1.0 / len(calibration), dtype=float
    )
    weights = {"split": uniform}
    temperatures = {"split": np.ones(len(evaluation), dtype=float)}

    z, bandwidth = comparison.fixed_rbf_fit(features, development)
    distance2 = np.sum(
        (z[evaluation, None] - z[None, calibration]) ** 2, axis=2
    )
    weights["fixed_rbf"], temperatures["fixed_rbf"] = (
        comparison.weights_with_min_ess(-distance2 / (2 * bandwidth**2))
    )
    logits = comparison.learned_logits(
        localizer, features[evaluation], features[calibration], device
    )
    weights["hopfield_time"], temperatures["hopfield_time"] = (
        comparison.weights_with_min_ess(logits)
    )

    rows = pd.DataFrame(
        {
            "cell_id": [names[index][:-4] for index in evaluation],
            "seed": seed,
            "eol": y[evaluation],
            "prediction": test_prediction,
            "target_log_error": test_log_error,
        }
    )
    for method in METHODS:
        risk = weights[method] @ cal_log_error
        q = comparison.weighted_quantiles(cal_error, weights[method], level)
        rows[f"{method}_risk"] = risk
        rows[f"{method}_q"] = q
        rows[f"{method}_lower"] = test_prediction - q
        rows[f"{method}_upper"] = test_prediction + q

    # With n_cal=15 and alpha=0.1, the finite-sample rank is 15 and every
    # empirical weighted quantile is the maximum residual, independent of weights.
    quantile_degenerate = level == 1.0
    if quantile_degenerate:
        reference = rows["split_q"].to_numpy()
        for method in METHODS[1:]:
            np.testing.assert_allclose(rows[f"{method}_q"], reference)

    metrics = metric_bundle(
        y[evaluation], test_prediction, rows, weights, temperatures, cal_log_error
    )
    metrics.update(
        {
            "seed": seed,
            "development_n": len(development),
            "calibration_n": len(calibration),
            "evaluation_n": len(evaluation),
            "conformal_rank": rank,
            "weighted_quantile_level": level,
            "quantile_degenerate_at_maximum": quantile_degenerate,
            "fixed_rbf_bandwidth": bandwidth,
            "localizer_training": training,
            "predictor_epochs": epochs,
        }
    )
    seed_dir = out / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    rows.to_csv(seed_dir / "evaluation.csv", index=False)
    comparison.save_json(seed_dir / "metrics.json", metrics)
    print(f"SEED {seed}: {json.dumps(metrics['methods'])}", flush=True)
    return metrics


def aggregate(metrics):
    result = {}
    for method in METHODS:
        values = [record["methods"][method] for record in metrics]
        correlations = [
            value["risk_spearman"]
            for value in values
            if value["risk_spearman"] is not None
        ]
        result[method] = {
            "coverage_mean": float(np.mean([value["coverage"] for value in values])),
            "mean_width_mean": float(np.mean([value["mean_width"] for value in values])),
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
            "ess_mean": float(np.mean([value["ess_mean"] for value in values])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = HERE / "runs_hust" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)

    split = comparison.previous.base.split_recorder
    train_names = list(split.HUST_train_files)
    calibration_names = list(split.HUST_val_files)
    test_names = list(split.HUST_test_files)
    names = train_names + calibration_names + test_names
    if len(names) != len(set(names)):
        raise ValueError("Official HUST train/validation/test splits overlap")

    config = {
        "experiment": "A: independent HUST replication",
        "T": 100,
        "dataset": "HUST",
        "official_split_sizes": {"train": 47, "validation": 15, "test": 15},
        "seeds": args.seeds,
        "alpha": comparison.ALPHA,
        "methods": METHODS,
        "embedding_dim": comparison.EMBED_DIM,
        "hidden_dim": comparison.HIDDEN_DIM,
        "min_ess": comparison.MIN_ESS,
        "localizer_loss": "crps",
        "predictor": vars(comparison.previous.CFG),
        "device": device,
        "data_root": str(args.data_root.resolve()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_usage": "used once after MATR hyperparameters were frozen",
        "weighted_interval_status": (
            "empirical basic-LCP diagnostic; no new local guarantee claimed"
        ),
    }
    comparison.save_json(output / "config.json", config)
    comparison.save_json(
        output / "split.json",
        {
            "development": train_names,
            "calibration": calibration_names,
            "evaluation": test_names,
        },
    )
    print(f"RUN_DIR={output}", flush=True)

    x, y, audit = load_hust_data(args.data_root, names)
    comparison.save_json(output / "data_audit.json", audit)
    development = np.arange(0, len(train_names))
    calibration = np.arange(len(train_names), len(train_names) + len(calibration_names))
    evaluation = np.arange(len(train_names) + len(calibration_names), len(names))

    all_metrics = []
    for seed in args.seeds:
        all_metrics.append(
            run_seed(
                x,
                y,
                names,
                development,
                calibration,
                evaluation,
                seed,
                device,
                output,
            )
        )
    summary = aggregate(all_metrics)
    summary["protocol_note"] = {
        "calibration_n": len(calibration),
        "conformal_rank": math.ceil(
            (len(calibration) + 1) * (1 - comparison.ALPHA)
        ),
        "weighted_quantile_level": 1.0,
        "interval_comparison_identifiable": False,
        "reason": (
            "At alpha=0.1 with 15 calibration cells, all empirical weighted "
            "quantiles equal the maximum calibration residual."
        ),
    }
    comparison.save_json(output / "summary.json", summary)
    print(f"SUMMARY: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
