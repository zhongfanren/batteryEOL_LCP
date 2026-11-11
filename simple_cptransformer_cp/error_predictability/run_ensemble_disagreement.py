"""Nested T=100 experiment: ensemble point prediction and disagreement risk."""
from __future__ import annotations

import argparse
import hashlib
import json
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
import run_crossfit_robust as robust

SEEDS = previous.SEEDS
METHODS = ["mean_prediction", "disagreement", "disagreement_simple"]
CFG = previous.CFG


def inputs(prediction_matrix, simple):
    mean = prediction_matrix.mean(axis=1)
    disagreement = np.log1p(prediction_matrix.std(axis=1, ddof=0))
    return {
        "mean_prediction": mean[:, None],
        "disagreement": np.column_stack([mean, disagreement]),
        "disagreement_simple": np.column_stack([mean, disagreement, simple]),
    }


def fit_risk_models(train_inputs, target):
    fitted = {}
    for name in METHODS:
        median, scale = robust.fit_transformer(train_inputs[name])
        model = Ridge(alpha=10.0).fit(robust.transform(train_inputs[name], median, scale), target)
        fitted[name] = {"median": median, "scale": scale, "model": model,
                        "target_min": float(target.min()), "target_max": float(target.max())}
    return fitted


def predict_risk(fitted, eval_inputs):
    result = {}
    for name, bundle in fitted.items():
        value = bundle["model"].predict(robust.transform(
            eval_inputs[name], bundle["median"], bundle["scale"]))
        result[name] = np.clip(value, bundle["target_min"], bundle["target_max"])
    return result


def train_ensemble(x, y, train_ids, predict_ids, split_seed, device):
    predictions, epochs = [], {}
    for seed in SEEDS:
        prediction, epoch = robust.train_and_predict(
            x, y, train_ids, predict_ids, seed, split_seed, device)
        predictions.append(prediction)
        epochs[str(seed)] = epoch
    return np.column_stack(predictions), epochs


def risk_metrics(rows):
    target = rows.target_log_error.to_numpy()
    base_mse = float(np.mean((target - rows.constant.to_numpy()) ** 2))
    result = {}
    for name in ["constant"] + METHODS:
        estimate = rows[name].to_numpy()
        mse = float(np.mean((target - estimate) ** 2))
        rho = None if np.ptp(estimate) == 0 else float(spearmanr(target, estimate).statistic)
        result[name] = {"log_mse": mse,
                        "improvement": 0.0 if name == "constant" else 1.0 - mse / base_mse,
                        "spearman": rho,
                        "scale_mae": float(np.mean(np.abs(
                            np.expm1(target) - np.maximum(1.0, np.expm1(estimate)))))}
    return result


def diagnose(x, y, names, simple, device, out):
    outer = KFold(5, shuffle=True, random_state=2024)
    all_rows, epochs = [], []
    for outer_fold, (outer_train, outer_eval) in enumerate(outer.split(x)):
        oof = np.full((len(outer_train), len(SEEDS)), np.nan)
        inner = KFold(4, shuffle=True, random_state=12024 + outer_fold)
        for inner_fold, (train_pos, eval_pos) in enumerate(inner.split(outer_train)):
            matrix, selected = train_ensemble(
                x, y, outer_train[train_pos], outer_train[eval_pos],
                22024 + 100 * outer_fold + inner_fold, device)
            oof[eval_pos] = matrix
            epochs.append({"outer_fold": outer_fold, "role": f"inner_{inner_fold}",
                           "epochs": selected})
        assert np.isfinite(oof).all()
        evaluation, selected = train_ensemble(
            x, y, outer_train, outer_eval, 32024 + outer_fold, device)
        epochs.append({"outer_fold": outer_fold, "role": "outer", "epochs": selected})

        target = np.log1p(np.abs(y[outer_train] - oof.mean(axis=1)))
        fitted = fit_risk_models(inputs(oof, simple[outer_train]), target)
        estimates = predict_risk(fitted, inputs(evaluation, simple[outer_eval]))
        mean_prediction = evaluation.mean(axis=1)
        fold = pd.DataFrame({
            "cell_id": [names[i][:-4] for i in outer_eval], "outer_fold": outer_fold,
            "eol": y[outer_eval], "prediction": mean_prediction,
            "seed_std": evaluation.std(axis=1, ddof=0),
            "target_log_error": np.log1p(np.abs(y[outer_eval] - mean_prediction)),
            "constant": float(target.mean()),
        })
        for column, seed in enumerate(SEEDS):
            fold[f"prediction_{seed}"] = evaluation[:, column]
        for name in METHODS:
            fold[name] = estimates[name]
        all_rows.append(fold)
        print(f"ensemble outer fold {outer_fold + 1}/5", flush=True)
    rows = pd.concat(all_rows, ignore_index=True)
    assert len(rows) == 102 and rows.cell_id.nunique() == 102
    metrics = {"point": previous.base.point_metrics(rows.eol.to_numpy(), rows.prediction.to_numpy()),
               "risk": risk_metrics(rows), "selected_epochs": epochs}
    rows.to_csv(out / "predictions.csv", index=False)
    previous.save_json(out / "metrics.json", metrics)
    return rows, metrics


def decide(metrics, rows):
    candidates = {}
    for name in METHODS:
        value = metrics["risk"][name]
        fold_values = [risk_metrics(rows[rows.outer_fold == fold])[name]
                       for fold in sorted(rows.outer_fold.unique())]
        positive_folds = sum(v["improvement"] > 0 and v["spearman"] is not None
                             and v["spearman"] > 0 for v in fold_values)
        candidates[name] = {"improvement": value["improvement"],
                            "spearman": value["spearman"],
                            "positive_folds": positive_folds,
                            "passes": value["improvement"] >= 0.05 and positive_folds >= 3}
    passing = [name for name in METHODS if candidates[name]["passes"]]
    selected = None
    if passing:
        best = max(candidates[name]["improvement"] for name in passing)
        selected = next(name for name in passing if best - candidates[name]["improvement"] < 0.01)
    return {"candidates": candidates, "selected": selected}


def train_external_ensemble(x_dev, y_dev, x_external, device):
    all_dev = np.arange(len(x_dev))
    predictions, epochs = [], {}
    for seed in SEEDS:
        split = robust.split_for_training(all_dev, 42024)
        selection, _, _, epoch, _ = previous.train_predictor(
            x_dev, y_dev, split["selection_train"], split["validation"], seed, device)
        del selection
        model, mean, scale, _, _ = previous.train_predictor(
            x_dev, y_dev, all_dev, [], seed, device, fixed_epochs=epoch)
        prediction = previous.base.infer(model, x_external, device, CFG.batch_size)[0]
        predictions.append(prediction.astype(float) * scale + mean)
        epochs[str(seed)] = epoch
        del model
    return np.column_stack(predictions), epochs


def interval_metrics(y, lower, upper):
    result = previous.base.interval_metrics(y, lower, upper, 0.1)
    result.update({"miss_lower": int(np.sum(y < lower)), "miss_upper": int(np.sum(y > upper))})
    return result


def run_intervals(data_root, x_dev, y_dev, names_dev, simple_dev,
                  diagnostic_rows, selected, device, out):
    extra_names = (list(previous.base.split_recorder.MATR_val_files) +
                   list(previous.base.split_recorder.MATR_test_files))
    x_extra, y_extra, audit = previous.load_data(data_root, extra_names)
    previous.save_json(out / "external_data_audit.json", audit)
    simple_extra = robust.stable_features(x_extra)[0]
    ncal = len(previous.base.split_recorder.MATR_val_files)

    order = {name[:-4]: i for i, name in enumerate(names_dev)}
    rows = diagnostic_rows.assign(order=diagnostic_rows.cell_id.map(order)).sort_values("order")
    dev_matrix = rows[[f"prediction_{seed}" for seed in SEEDS]].to_numpy()
    target = rows.target_log_error.to_numpy()
    fitted = fit_risk_models(inputs(dev_matrix, simple_dev), target)

    external_matrix, epochs = train_external_ensemble(x_dev, y_dev, x_extra, device)
    prediction = external_matrix.mean(axis=1)
    predicted_log_risk = predict_risk(fitted, inputs(external_matrix, simple_extra))[selected]
    scale = np.maximum(1.0, np.expm1(predicted_log_risk))
    cal_y, cal_prediction = y_extra[:ncal], prediction[:ncal]
    test_y, test_prediction = y_extra[ncal:], prediction[ncal:]
    q_abs, rank = previous.base.conformal_upper_quantile(
        np.abs(cal_y - cal_prediction), 0.1)
    q_normalized, _ = previous.base.conformal_upper_quantile(
        np.abs(cal_y - cal_prediction) / scale[:ncal], 0.1)
    radii = {"split": np.full(len(test_y), q_abs),
             "normalized": q_normalized * scale[ncal:]}
    metrics = {"method": selected, "epochs": epochs, "rank": rank,
               "point": previous.base.point_metrics(test_y, test_prediction),
               "historical_test_previously_viewed": True}
    table = pd.DataFrame({"cell_id": [name[:-4] for name in extra_names[ncal:]],
                          "eol": test_y, "prediction": test_prediction,
                          "seed_std": external_matrix[ncal:].std(axis=1, ddof=0),
                          "predicted_scale": scale[ncal:]})
    for method, radius in radii.items():
        lower, upper = test_prediction - radius, test_prediction + radius
        metrics[method] = {**interval_metrics(test_y, lower, upper),
                           "q": q_abs if method == "split" else q_normalized}
        table[f"{method}_lower"], table[f"{method}_upper"] = lower, upper
    table.to_csv(out / "interval_predictions.csv", index=False)
    joblib.dump({"method": selected, "risk_model": fitted[selected]}, out / "risk_model.joblib")
    previous.save_json(out / "interval_metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = previous.HERE / "runs_ensemble_disagreement" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    config = {"ensemble_seeds": SEEDS, "outer_folds": 5, "inner_folds": 4,
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
    simple = robust.stable_features(x)[0]
    rows, metrics = diagnose(x, y, names, simple, device, out)
    decision = decide(metrics, rows)
    previous.save_json(out / "decision.json", decision)
    print(f"METRICS: {json.dumps(metrics['risk'])}", flush=True)
    print(f"DECISION: {json.dumps(decision)}", flush=True)
    if decision["selected"]:
        intervals = run_intervals(args.data_root, x, y, names, simple, rows,
                                  decision["selected"], device, out)
        print(f"INTERVALS: {json.dumps(intervals)}", flush=True)
    else:
        print("Gate not passed; calibration/test were not used.", flush=True)


if __name__ == "__main__":
    main()
