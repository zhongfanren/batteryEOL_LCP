"""Small, exploratory point-predictor check on BatteryML's MATR1/MATR2 cell splits.

The cell IDs and label rule follow BatteryML. Input pickles are the available
BatteryLife-processed version, so these numbers are not an exact BatteryML
reproduction. The historical test cells have already been viewed in this project.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "predictor_ensemble"))
sys.path.insert(0, str(HERE.parent / "error_predictability"))
import run_t100 as base  # noqa: E402
from run_generic_dq import generic_feature  # noqa: E402
from run_seed_ensemble import OLD_SEEDS, measures  # noqa: E402
import run_crossfit_robust as robust  # noqa: E402


# Exact IDs from BatteryML's MATRPrimary/SecondaryTestTrainTestSplitter.
TRAIN = """
b1c1 b1c3 b1c5 b1c7 b1c11 b1c15 b1c17 b1c19 b1c21 b1c24
b1c26 b1c28 b1c30 b1c32 b1c34 b1c36 b1c38 b1c40 b1c42 b1c44
b2c0 b2c2 b2c4 b2c6 b2c11 b2c13 b2c17 b2c19 b2c21 b2c23
b2c25 b2c27 b2c29 b2c31 b2c33 b2c35 b2c37 b2c39 b2c41 b2c43 b2c45
""".split()
PRIMARY = """
b1c0 b1c2 b1c4 b1c6 b1c9 b1c14 b1c16 b1c18 b1c20 b1c23
b1c25 b1c27 b1c29 b1c31 b1c33 b1c35 b1c37 b1c39 b1c41 b1c43 b1c45
b2c3 b2c5 b2c10 b2c12 b2c14 b2c18 b2c20 b2c22 b2c24 b2c26
b2c28 b2c30 b2c32 b2c34 b2c36 b2c38 b2c40 b2c42 b2c44 b2c46 b2c47
""".split()
SECONDARY = """
b3c0 b3c1 b3c3 b3c4 b3c5 b3c6 b3c7 b3c8 b3c9 b3c10
b3c11 b3c12 b3c13 b3c14 b3c15 b3c16 b3c17 b3c18 b3c19 b3c20
b3c21 b3c22 b3c24 b3c25 b3c26 b3c27 b3c28 b3c29 b3c30 b3c31
b3c33 b3c34 b3c35 b3c36 b3c38 b3c39 b3c40 b3c41 b3c44 b3c45
""".split()


def batteryml_eol(cell: dict) -> float:
    """BatteryML RULLabelAnnotator default rule, including its 1-based offset."""
    label = 1
    found = False
    threshold = 0.8 * float(cell["nominal_capacity_in_Ah"])
    for cycle in cell["cycle_data"]:
        label += 1
        if max(cycle["discharge_capacity_in_Ah"]) <= threshold:
            found = True
            break
    if not found:
        label += 1
    if label <= 100:
        raise ValueError("BatteryML label is not greater than 100")
    return float(label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    parser.add_argument("--feature-only", action="store_true")
    args = parser.parse_args()
    ids = TRAIN + PRIMARY + SECONDARY
    if (len(TRAIN), len(PRIMARY), len(SECONDARY)) != (41, 42, 40) or len(set(ids)) != 123:
        raise ValueError("BatteryML split IDs have incorrect counts or overlap")
    labels = json.loads((args.data_root / "Life labels" / "MATR_labels.json").read_text(encoding="utf-8"))
    x, y, feature, old_y = [], [], [], []
    for index, cell_id in enumerate(ids):
        file = f"MATR_{cell_id}.pkl"
        path = args.data_root / "MATR" / file
        with path.open("rb") as stream:
            cell = pickle.load(stream)
        y.append(batteryml_eol(cell))
        old_y.append(float(labels[file]))
        feature.append(generic_feature(path))
        if not args.feature_only:
            x.append(base.load_cell(path, y[-1]))
        if (index + 1) % 25 == 0:
            print(f"loaded {index + 1}/{len(ids)} cells", flush=True)
    y = np.asarray(y)
    old_y = np.asarray(old_y)
    feature = np.asarray(feature)[:, None]
    if not np.isfinite(feature).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite input or label")
    split_index = np.array(["train"] * 41 + ["MATR1"] * 42 + ["MATR2"] * 40)
    fit = np.arange(41)
    variance_model = LinearRegression().fit(feature[fit], np.log(y[fit]))
    variance = np.clip(np.exp(variance_model.predict(feature)), y[fit].min(), y[fit].max())
    # BatteryML's log-label transformation makes its transformed-space dummy
    # correspond to the geometric mean after the inverse transformation.
    dummy = np.full(len(y), float(np.exp(np.log(y[fit]).mean())))
    print("log-space dummy RMSE MATR1/MATR2:",
          *[round(measures(y[split_index == s], dummy[split_index == s])["rmse"], 2)
            for s in ("MATR1", "MATR2")], flush=True)

    predictions = {"generic_variance": variance}
    epochs = {}
    if not args.feature_only:
        x = np.stack(x)
        if x.shape != (123, 100, 3, 300):
            raise ValueError(f"Unexpected input shape: {x.shape}")
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for the CPTransformer check")
        seed_matrix = []
        for seed in OLD_SEEDS:
            pred, epoch = robust.train_and_predict(
                x, y, fit, np.arange(41, 123), seed, 42024, "cuda"
            )
            seed_matrix.append(pred)
            epochs[str(seed)] = int(epoch)
            print(f"seed {seed}: selected epoch {epoch}", flush=True)
        cp = np.full(len(y), np.nan)
        cp[41:] = np.column_stack(seed_matrix).mean(axis=1)
        blend = np.full(len(y), np.nan)
        blend[41:] = 0.5 * (cp[41:] + variance[41:])
        predictions.update(cptransformer_mean_3=cp, blend_3_50pct=blend)

    output = HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    rows = pd.DataFrame({
        "cell_id": [f"MATR_{cell_id}" for cell_id in ids],
        "split": split_index,
        "batteryml_rule_eol": y,
        "batterylife_eol": old_y,
        "generic_feature": feature[:, 0],
        "dummy_log_mean": dummy,
        **predictions,
    })
    rows.to_csv(output / "predictions.csv", index=False)
    metrics = {
        "data_source": "BatteryLife-processed MATR pickles; not an exact BatteryML data reproduction",
        "split_source": "https://github.com/microsoft/BatteryML/blob/main/batteryml/train_test_split/MATR_split.py",
        "label_source": "https://github.com/microsoft/BatteryML/blob/main/batteryml/label/rul.py",
        "old_vs_recomputed_label": {
            "median_absolute_difference": float(np.median(np.abs(y - old_y))),
            "maximum_absolute_difference": float(np.max(np.abs(y - old_y))),
        },
        "predictor": "existing T=100 CPTransformer, three fixed seeds, and generic variance feature",
        "selected_epochs": epochs,
        "feature_coefficient": float(variance_model.coef_[0]),
        "results": {},
        "status": "exploratory: these historical test cells have been viewed before",
    }
    for split in ("MATR1", "MATR2"):
        mask = split_index == split
        metrics["results"][split] = {
            "n": int(mask.sum()),
            "dummy_log_mean": measures(y[mask], dummy[mask]),
            **{name: measures(y[mask], pred[mask]) for name, pred in predictions.items()},
        }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"RUN_DIR={output}", flush=True)
    print(json.dumps(metrics["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
