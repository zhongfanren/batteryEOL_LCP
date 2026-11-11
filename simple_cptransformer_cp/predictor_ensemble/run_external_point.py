"""Fit five CPTransformers and a cycle-10/100 variance model on MATR train.

Official validation and test are evaluated as exploratory checks because this
project has viewed the test set in earlier experiments. Neither is used for
model selection, feature regression, or ensemble weighting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from run_feature_blend import cell_features
from run_seed_ensemble import ALL_SEEDS, measures


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "error_predictability"
sys.path.insert(0, str(EXPERIMENTS))
import run as experiment  # noqa: E402
import run_crossfit_robust as robust  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This experiment expects the available GPU")
    names = {
        "train": list(experiment.base.split_recorder.MATR_train_files),
        "validation": list(experiment.base.split_recorder.MATR_val_files),
        "test": list(experiment.base.split_recorder.MATR_test_files),
    }
    sets = {name: set(files) for name, files in names.items()}
    if len(sets["train"] | sets["validation"] | sets["test"]) != 169:
        raise ValueError("MATR official cell partitions are not disjoint")
    config = {
        "task": "T=100 full MATR-train point predictor; exploratory official validation/test",
        "seeds": ALL_SEEDS,
        "predictor": vars(experiment.CFG),
        "train_n": 102,
        "validation_n": 34,
        "test_n": 33,
        "epoch_selection_split_seed": 42024,
        "variance_feature": "log10 variance of Qdlin cycle-100 minus cycle-10",
        "variance_target": "log EOL",
        "variance_prediction_clipping": "training EOL min/max",
        "blend": "0.5 * CPTransformer 5-seed mean + 0.5 * variance model",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": device,
        "test_status": "previously viewed; exploratory only",
    }
    if args.resume:
        output = args.resume.resolve()
        old = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(config)):
            raise ValueError("Resume configuration mismatch")
    else:
        output = HERE / "runs_external" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        experiment.save_json(output / "config.json", config)
    print(f"RUN_DIR={output}", flush=True)

    train_x, train_y, train_audit = experiment.load_data(args.data_root, names["train"])
    extra_names = names["validation"] + names["test"]
    extra_x, extra_y, extra_audit = experiment.load_data(args.data_root, extra_names)
    experiment.save_json(output / "data_audit.json", {
        "train": train_audit, "external": extra_audit
    })
    train_feature = np.stack([
        cell_features(args.data_root / "MATR" / name)[0]
        for name in names["train"]
    ])[:, None]
    external_feature = np.stack([
        cell_features(args.data_root / "MATR" / name)[0]
        for name in extra_names
    ])[:, None]
    variance_model = LinearRegression().fit(train_feature, np.log(train_y))
    variance_prediction = np.clip(
        np.exp(variance_model.predict(external_feature)),
        float(train_y.min()), float(train_y.max())
    )
    joblib.dump({
        "model": variance_model,
        "feature": "log10 variance of Qdlin at cycle 100 minus cycle 10",
        "eol_clip": (float(train_y.min()), float(train_y.max())),
    }, output / "variance_model.joblib")

    matrix = np.full((len(extra_names), len(ALL_SEEDS)), np.nan)
    chosen_epochs = {}
    all_train = np.arange(len(train_y))
    for column, seed in enumerate(ALL_SEEDS):
        pred_path = output / f"seed_{seed}_predictions.npy"
        model_path = output / f"seed_{seed}_model.pt"
        if pred_path.exists() and model_path.exists():
            matrix[:, column] = np.load(pred_path)
            bundle = torch.load(model_path, map_location="cpu", weights_only=False)
            chosen_epochs[str(seed)] = int(bundle["selected_epoch"])
            continue
        split = robust.split_for_training(all_train, 42024)
        selection, _, _, epoch, _ = experiment.train_predictor(
            train_x, train_y, split["selection_train"], split["validation"], seed, device
        )
        del selection
        model, mean, scale, _, _ = experiment.train_predictor(
            train_x, train_y, all_train, [], seed, device, fixed_epochs=epoch
        )
        prediction = experiment.base.infer(
            model, extra_x, device, experiment.CFG.batch_size
        )[0].astype(float) * scale + mean
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"Nonfinite predictions for seed {seed}")
        torch.save({
            "state_dict": model.state_dict(), "label_mean": mean,
            "label_scale": scale, "selected_epoch": epoch,
            "predictor_config": vars(experiment.CFG),
        }, model_path)
        np.save(pred_path, prediction)
        matrix[:, column] = prediction
        chosen_epochs[str(seed)] = int(epoch)
        del model
        print(f"seed {seed}: selected epoch {epoch}", flush=True)

    cp_mean = matrix.mean(axis=1)
    predictions = pd.DataFrame({
        "cell_id": [name.removesuffix(".pkl") for name in extra_names],
        "split": ["validation"] * len(names["validation"]) + ["test"] * len(names["test"]),
        "eol": extra_y,
        "log10_delta_q_variance": external_feature[:, 0],
        "cptransformer_mean_5": cp_mean,
        "variance_one_feature": variance_prediction,
        "blend_50pct": .5 * (cp_mean + variance_prediction),
    })
    for column, seed in enumerate(ALL_SEEDS):
        predictions[f"seed_{seed}"] = matrix[:, column]
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics = {"selected_epochs": chosen_epochs}
    for split in ("validation", "test"):
        block = predictions[predictions.split == split]
        actual = block.eol.to_numpy()
        metrics[split] = {
            method: measures(actual, block[method].to_numpy())
            for method in ("cptransformer_mean_5", "variance_one_feature", "blend_50pct")
        }
        for seed in ALL_SEEDS:
            metrics[split][f"seed_{seed}"] = measures(actual, block[f"seed_{seed}"].to_numpy())
    experiment.save_json(output / "metrics.json", metrics)
    print("RESULTS=" + json.dumps({k: metrics[k] for k in ("validation", "test")}), flush=True)


if __name__ == "__main__":
    main()
