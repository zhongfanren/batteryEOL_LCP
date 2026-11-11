"""Nested cross-fit error predictability with robust, bounded feature transforms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import run as previous

METHODS = ["prediction_only", "simple", "multiscale"]
SEEDS = previous.SEEDS
CFG = previous.CFG


def split_for_training(indices, split_seed):
    shuffled = np.random.default_rng(split_seed).permutation(indices)
    return {
        "predictor": shuffled.tolist(),
        "validation": shuffled[:8].tolist(),
        "selection_train": shuffled[8:].tolist(),
    }


def train_and_predict(x, y, train_ids, predict_ids, model_seed, split_seed, device):
    split = split_for_training(np.asarray(train_ids), split_seed)
    selection, _, _, epoch, _ = previous.train_predictor(
        x, y, split["selection_train"], split["validation"], model_seed, device)
    del selection
    model, mean, scale, _, _ = previous.train_predictor(
        x, y, split["predictor"], [], model_seed, device, fixed_epochs=epoch)
    pred = previous.base.infer(model, x[predict_ids], device, CFG.batch_size)[0]
    pred = pred.astype(float) * scale + mean
    del model
    if not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite point prediction")
    return pred, epoch


def stable_features(x):
    simple, multiscale = previous.features(x)
    energy = np.log1p(np.maximum(multiscale[:, simple.shape[1]:], 0.0))
    return simple, np.column_stack([simple, energy])


def risk_inputs(point_predictions, simple, multiscale):
    return {
        "prediction_only": point_predictions.reshape(-1, 1),
        "simple": np.column_stack([point_predictions, simple]),
        "multiscale": np.column_stack([point_predictions, multiscale]),
    }


def fit_transformer(values):
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    iqr = q75 - q25
    scale = np.where(iqr > 1e-8, iqr, 1.0)
    return median, scale


def transform(values, median, scale):
    return np.clip((values - median) / scale, -5.0, 5.0)


def fit_risk_models(train_inputs, target):
    fitted = {}
    for name in METHODS:
        median, scale = fit_transformer(train_inputs[name])
        model = Ridge(alpha=10.0).fit(transform(train_inputs[name], median, scale), target)
        fitted[name] = {
            "median": median, "scale": scale, "model": model,
            "target_min": float(target.min()), "target_max": float(target.max()),
        }
    return fitted


def predict_risk(fitted, inputs):
    result = {}
    for name, bundle in fitted.items():
        value = bundle["model"].predict(transform(inputs[name], bundle["median"], bundle["scale"]))
        result[name] = np.clip(value, bundle["target_min"], bundle["target_max"])
    return result


def metric_table(rows):
    target = rows["target_log_error"].to_numpy()
    base_mse = float(np.mean((target - rows["constant"]) ** 2))
    result = {}
    for name in ["constant"] + METHODS:
        estimate = rows[name].to_numpy()
        mse = float(np.mean((target - estimate) ** 2))
        rho = None if np.ptp(estimate) == 0 else float(spearmanr(target, estimate).statistic)
        result[name] = {
            "log_mse": mse,
            "improvement": 0.0 if name == "constant" else 1.0 - mse / base_mse,
            "spearman": rho,
            "scale_mae": float(np.mean(np.abs(np.expm1(target) - np.maximum(1.0, np.expm1(estimate))))),
        }
    return result


def diagnose_seed(x, y, names, simple, multiscale, seed, device, out):
    outer = KFold(5, shuffle=True, random_state=2024)
    result_rows = []
    epochs = []
    for outer_fold, (outer_train, outer_eval) in enumerate(outer.split(x)):
        oof_pred = np.full(len(outer_train), np.nan)
        inner = KFold(4, shuffle=True, random_state=12024 + outer_fold)
        for inner_fold, (inner_train_pos, inner_eval_pos) in enumerate(inner.split(outer_train)):
            inner_train = outer_train[inner_train_pos]
            inner_eval = outer_train[inner_eval_pos]
            pred, epoch = train_and_predict(
                x, y, inner_train, inner_eval, seed,
                22024 + 100 * outer_fold + inner_fold, device)
            oof_pred[inner_eval_pos] = pred
            epochs.append({"outer_fold": outer_fold, "role": f"inner_{inner_fold}", "epoch": epoch})
        assert np.isfinite(oof_pred).all()
        eval_pred, epoch = train_and_predict(
            x, y, outer_train, outer_eval, seed, 32024 + outer_fold, device)
        epochs.append({"outer_fold": outer_fold, "role": "outer", "epoch": epoch})

        train_inputs = risk_inputs(oof_pred, simple[outer_train], multiscale[outer_train])
        eval_inputs = risk_inputs(eval_pred, simple[outer_eval], multiscale[outer_eval])
        target = np.log1p(np.abs(y[outer_train] - oof_pred))
        fitted = fit_risk_models(train_inputs, target)
        risk = predict_risk(fitted, eval_inputs)
        fold_rows = pd.DataFrame({
            "cell_id": [names[i][:-4] for i in outer_eval],
            "outer_fold": outer_fold,
            "eol": y[outer_eval],
            "prediction": eval_pred,
            "target_log_error": np.log1p(np.abs(y[outer_eval] - eval_pred)),
            "constant": float(target.mean()),
        })
        for name in METHODS:
            fold_rows[name] = risk[name]
        result_rows.append(fold_rows)
        print(f"seed {seed}: robust outer fold {outer_fold + 1}/5", flush=True)
    rows = pd.concat(result_rows, ignore_index=True)
    assert len(rows) == 102 and rows.cell_id.nunique() == 102
    rows.to_csv(out / f"seed_{seed}_predictions.csv", index=False)
    metrics = {
        "point": previous.base.point_metrics(rows.eol.to_numpy(), rows.prediction.to_numpy()),
        "risk": metric_table(rows),
        "selected_epochs": epochs,
    }
    previous.save_json(out / f"seed_{seed}_metrics.json", metrics)
    return rows, metrics


def decide(metrics):
    candidates = {}
    for name in METHODS:
        values = [m["risk"][name] for m in metrics]
        mean_improvement = float(np.mean([v["improvement"] for v in values]))
        positive_seeds = sum(v["improvement"] > 0 and v["spearman"] is not None and v["spearman"] > 0
                             for v in values)
        candidates[name] = {
            "mean_improvement": mean_improvement,
            "positive_seeds": positive_seeds,
            "passes": mean_improvement >= 0.05 and positive_seeds >= 2,
        }
    passing = [name for name in METHODS if candidates[name]["passes"]]
    selected = None
    if passing:
        best = max(candidates[name]["mean_improvement"] for name in passing)
        selected = next(name for name in passing if best - candidates[name]["mean_improvement"] < 0.01)
    return {"candidates": candidates, "selected": selected}


def interval_metrics(y, lo, hi, alpha):
    result = previous.base.interval_metrics(y, lo, hi, alpha)
    result.update({"miss_lower": int(np.sum(y < lo)), "miss_upper": int(np.sum(y > hi))})
    return result


def run_intervals(data_root, x_dev, y_dev, names_dev, simple_dev, multiscale_dev,
                  diagnostic_rows, selected, seed, device, out):
    extra_names = list(previous.base.split_recorder.MATR_val_files) + list(previous.base.split_recorder.MATR_test_files)
    x_extra, y_extra, audit = previous.load_data(data_root, extra_names)
    ncal = len(previous.base.split_recorder.MATR_val_files)
    cal_slice, test_slice = slice(0, ncal), slice(ncal, len(extra_names))
    simple_extra, multiscale_extra = stable_features(x_extra)

    # All development-cell error targets are OOF; final calibration/test labels remain untouched.
    oof_pred = diagnostic_rows.prediction.to_numpy()
    order = {cell[:-4]: i for i, cell in enumerate(names_dev)}
    diagnostic_rows = diagnostic_rows.assign(order=diagnostic_rows.cell_id.map(order)).sort_values("order")
    oof_pred = diagnostic_rows.prediction.to_numpy()
    target = diagnostic_rows.target_log_error.to_numpy()
    train_inputs = risk_inputs(oof_pred, simple_dev, multiscale_dev)
    fitted = fit_risk_models(train_inputs, target)

    all_dev = np.arange(len(x_dev))
    # Select the epoch on development cells, then fit once on all development cells.
    split = split_for_training(all_dev, 42024)
    selection, _, _, selected_epoch, _ = previous.train_predictor(
        x_dev, y_dev, split["selection_train"], split["validation"], seed, device)
    del selection
    model, mean, scale, _, history = previous.train_predictor(
        x_dev, y_dev, all_dev, [], seed, device, fixed_epochs=selected_epoch)
    external_pred = previous.base.infer(model, x_extra, device, CFG.batch_size)[0].astype(float) * scale + mean
    torch.save({"state_dict": model.state_dict(), "label_mean": mean, "label_scale": scale,
                "epoch": selected_epoch}, out / f"interval_seed_{seed}_model.pt")
    del model
    inputs_extra = risk_inputs(external_pred, simple_extra, multiscale_extra)
    risk_log = predict_risk(fitted, inputs_extra)[selected]
    g = np.maximum(1.0, np.expm1(risk_log))
    cal_y, cal_pred = y_extra[cal_slice], external_pred[cal_slice]
    test_y, test_pred = y_extra[test_slice], external_pred[test_slice]
    q_abs, rank = previous.base.conformal_upper_quantile(np.abs(cal_y - cal_pred), 0.1)
    q_norm, _ = previous.base.conformal_upper_quantile(np.abs(cal_y - cal_pred) / g[cal_slice], 0.1)
    radii = {"split": np.full(len(test_y), q_abs), "normalized": q_norm * g[test_slice]}
    metrics = {"method": selected, "selected_epoch": selected_epoch, "rank": rank,
               "point": previous.base.point_metrics(test_y, test_pred), "historical_test_previously_viewed": True}
    table = pd.DataFrame({"cell_id": [n[:-4] for n in extra_names[ncal:]], "eol": test_y,
                          "prediction": test_pred, "predicted_scale": g[test_slice]})
    for name, radius in radii.items():
        lo, hi = test_pred - radius, test_pred + radius
        metrics[name] = {**interval_metrics(test_y, lo, hi, 0.1),
                         "q": q_abs if name == "split" else q_norm}
        table[name + "_lower"], table[name + "_upper"] = lo, hi
    table.to_csv(out / f"interval_seed_{seed}_predictions.csv", index=False)
    joblib.dump({"method": selected, "risk_model": fitted[selected]}, out / f"interval_seed_{seed}_risk.joblib")
    previous.save_json(out / f"interval_seed_{seed}_metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = previous.HERE / "runs_robust_crossfit" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    config = {"model_seeds": SEEDS, "outer_folds": 5, "inner_folds": 4,
              "outer_split_seed": 2024, "ridge_alpha": 10.0,
              "robust_scale": "median/IQR", "feature_clip": [-5, 5],
              "risk_prediction_clip": "training target range", "device": device,
              "data_root": str(args.data_root.resolve()), "predictor": vars(CFG),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    previous.save_json(out / "config.json", config)
    print(f"RUN_DIR={out}", flush=True)
    names = list(previous.base.split_recorder.MATR_train_files)
    x, y, audit = previous.load_data(args.data_root, names)
    previous.save_json(out / "data_audit.json", audit)
    simple, multiscale = stable_features(x)
    rows, metrics = [], []
    for seed in SEEDS:
        seed_rows, seed_metrics = diagnose_seed(x, y, names, simple, multiscale, seed, device, out)
        rows.append(seed_rows); metrics.append(seed_metrics)
        print(f"SEED {seed}: {json.dumps(seed_metrics['risk'])}", flush=True)
    decision = decide(metrics)
    previous.save_json(out / "decision.json", decision)
    print(f"DECISION: {json.dumps(decision)}", flush=True)
    if decision["selected"]:
        interval = {}
        for seed, seed_rows in zip(SEEDS, rows):
            interval[str(seed)] = run_intervals(args.data_root, x, y, names, simple, multiscale,
                                                 seed_rows, decision["selected"], seed, device, out)
        previous.save_json(out / "interval_summary.json", interval)
    else:
        print("Gate not passed; normalized interval experiment not run.", flush=True)


if __name__ == "__main__":
    main()
