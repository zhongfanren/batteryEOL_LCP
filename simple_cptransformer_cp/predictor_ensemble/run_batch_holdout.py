"""Exploratory MATR T=100 point-prediction check with one entire batch held out."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from run_feature_blend import cell_features
from run_seed_ensemble import OLD_SEEDS, measures


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "error_predictability"
sys.path.insert(0, str(EXPERIMENTS))
import run as experiment  # noqa: E402
import run_crossfit_robust as robust  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This experiment expects the available GPU")
    names = list(experiment.base.split_recorder.MATR_train_files)
    x, y, audit = experiment.load_data(args.data_root, names)
    batch = np.array([name.split("c")[0].replace("MATR_", "") for name in names])
    groups = sorted(set(batch))
    if groups != ["b1", "b2", "b3", "b4"]:
        raise ValueError(f"Unexpected batches: {groups}")
    feature = np.array([
        cell_features(args.data_root / "MATR" / name)[0] for name in names
    ])[:, None]
    output = HERE / "runs_batch_holdout" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    experiment.save_json(output / "config.json", {
        "task": "exploratory leave-one-MATR-batch-out T=100 point prediction",
        "seeds": OLD_SEEDS,
        "predictor": vars(experiment.CFG),
        "feature": "Qdlin cycle 100 - cycle 10 log variance",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_validation_or_test_used": False,
        "device": device,
    })
    experiment.save_json(output / "data_audit.json", audit)
    print(f"RUN_DIR={output}", flush=True)
    records = []
    fold_metrics = {}
    for k, group in enumerate(groups):
        train_ids = np.flatnonzero(batch != group)
        eval_ids = np.flatnonzero(batch == group)
        model = LinearRegression().fit(feature[train_ids], np.log(y[train_ids]))
        variance = np.clip(
            np.exp(model.predict(feature[eval_ids])),
            float(y[train_ids].min()), float(y[train_ids].max())
        )
        matrix = []
        epochs = {}
        for seed in OLD_SEEDS:
            prediction, epoch = robust.train_and_predict(
                x, y, train_ids, eval_ids, seed, 52024 + k, device
            )
            matrix.append(prediction)
            epochs[str(seed)] = int(epoch)
            print(f"holdout {group}, seed {seed}: epoch {epoch}", flush=True)
        matrix = np.column_stack(matrix)
        cp = matrix.mean(axis=1)
        blend = .5 * (cp + variance)
        for j, index in enumerate(eval_ids):
            records.append({
                "cell_id": names[index].removesuffix(".pkl"),
                "held_out_batch": group,
                "eol": float(y[index]),
                "cptransformer_mean_3": float(cp[j]),
                "variance_one_feature": float(variance[j]),
                "blend_3_50pct": float(blend[j]),
                **{f"seed_{seed}": float(matrix[j, s]) for s, seed in enumerate(OLD_SEEDS)},
            })
        fold_metrics[group] = {
            "train_n": len(train_ids), "eval_n": len(eval_ids),
            "selected_epochs": epochs,
            "dummy_train_mean": measures(y[eval_ids], np.full(len(eval_ids), y[train_ids].mean())),
            "cptransformer_mean_3": measures(y[eval_ids], cp),
            "variance_one_feature": measures(y[eval_ids], variance),
            "blend_3_50pct": measures(y[eval_ids], blend),
        }
    rows = pd.DataFrame(records)
    rows.to_csv(output / "predictions.csv", index=False)
    actual = rows.eol.to_numpy()
    result = {
        "folds": fold_metrics,
        "overall": {name: measures(actual, rows[name].to_numpy()) for name in
                    ("cptransformer_mean_3", "variance_one_feature", "blend_3_50pct")},
    }
    experiment.save_json(output / "metrics.json", result)
    print("RESULTS=" + json.dumps({k: {m: round(v[m]["rmse"], 2) for m in
        ("cptransformer_mean_3", "variance_one_feature", "blend_3_50pct")}
        for k, v in fold_metrics.items()}), flush=True)
    print("OVERALL=" + json.dumps(result["overall"]), flush=True)


if __name__ == "__main__":
    main()
