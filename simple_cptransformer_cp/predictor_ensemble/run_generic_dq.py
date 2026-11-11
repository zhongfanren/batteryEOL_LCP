"""Cross-dataset cycle-10/100 discharge-curve variance from raw common fields."""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from run_seed_ensemble import measures


HERE = Path(__file__).resolve().parent
GRID = np.linspace(2.05, 3.35, 256)


def discharge_curve(cycle: dict, nominal_capacity: float) -> np.ndarray:
    voltage = np.asarray(cycle["voltage_in_V"], dtype=float)
    current = np.asarray(cycle["current_in_A"], dtype=float)
    capacity = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=float)
    keep = (
        np.isfinite(voltage) & np.isfinite(current) & np.isfinite(capacity)
        & (current < -0.01 * nominal_capacity)
    )
    if keep.sum() < 30:
        raise ValueError("Too few discharge measurements")
    voltage, capacity = voltage[keep], capacity[keep] / nominal_capacity
    ordered = np.argsort(voltage)
    voltage, capacity = voltage[ordered], capacity[ordered]
    voltage, unique = np.unique(voltage, return_index=True)
    capacity = capacity[unique]
    if voltage[0] > GRID[0] + .05 or voltage[-1] < GRID[-1] - .05:
        raise ValueError("Discharge curve misses shared voltage grid")
    return np.interp(GRID, voltage, capacity)


def generic_feature(path: Path) -> float:
    with path.open("rb") as stream:
        cell = pickle.load(stream)
    if len(cell["cycle_data"]) < 100:
        raise ValueError(f"Less than 100 cycles: {path.name}")
    nominal = float(cell["nominal_capacity_in_Ah"])
    early = discharge_curve(cell["cycle_data"][9], nominal)
    late = discharge_curve(cell["cycle_data"][99], nominal)
    value = float(np.log10(max(np.var(late - early), 1e-12)))
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite feature: {path.name}")
    return value


def evaluate_dataset(data_root: Path, name: str, files: dict) -> tuple[dict, pd.DataFrame, LinearRegression, tuple[float, float]]:
    labels = json.loads((data_root / "Life labels" / f"{name}_labels.json").read_text())
    groups = {}
    for split, split_files in files.items():
        x = np.array([generic_feature(data_root / name / file) for file in split_files])[:, None]
        y = np.array([float(labels[file]) for file in split_files])
        if not np.isfinite(y).all() or np.any(y <= 100):
            raise ValueError(f"Invalid labels in {name}/{split}")
        groups[split] = (x, y)
    x_train, y_train = groups["train"]
    oof = np.full(len(y_train), np.nan)
    dummy = np.full(len(y_train), np.nan)
    coefficients = []
    for fold, (fit_ids, eval_ids) in enumerate(
        KFold(5, shuffle=True, random_state=2024).split(y_train)
    ):
        model = LinearRegression().fit(x_train[fit_ids], np.log(y_train[fit_ids]))
        oof[eval_ids] = np.clip(
            np.exp(model.predict(x_train[eval_ids])),
            float(y_train[fit_ids].min()), float(y_train[fit_ids].max())
        )
        dummy[eval_ids] = y_train[fit_ids].mean()
        coefficients.append(float(model.coef_[0]))
    result = {
        "train_5fold": {
            "n": len(y_train),
            "dummy": measures(y_train, dummy),
            "variance_feature": measures(y_train, oof),
            "coefficients": coefficients,
        }
    }
    records = [pd.DataFrame({
        "dataset": name, "split": "train_5fold",
        "cell_id": [file.removesuffix(".pkl") for file in files["train"]],
        "eol": y_train,
        "feature": x_train[:, 0],
        "prediction": oof,
    })]
    full = LinearRegression().fit(x_train, np.log(y_train))
    for split in ("validation", "test"):
        x, y = groups[split]
        prediction = np.clip(
            np.exp(full.predict(x)),
            float(y_train.min()), float(y_train.max())
        )
        result[split] = {
            "n": len(y),
            "dummy": measures(y, np.full(len(y), y_train.mean())),
            "variance_feature": measures(y, prediction),
        }
        records.append(pd.DataFrame({
            "dataset": name, "split": split,
            "cell_id": [file.removesuffix(".pkl") for file in files[split]],
            "eol": y, "feature": x[:, 0], "prediction": prediction,
        }))
    result["full_train_coefficient"] = float(full.coef_[0])
    return result, pd.concat(records, ignore_index=True), full, (
        float(y_train.min()), float(y_train.max())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    args = parser.parse_args()
    from sys import path
    path.insert(0, str(HERE.parent.parent / "third_party" / "BatteryLife"))
    from data_provider.data_split_recorder import split_recorder

    output = HERE / "runs_generic_dq" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    all_metrics, all_rows = {}, []
    for name in ("MATR", "HUST"):
        files = {
            "train": list(getattr(split_recorder, f"{name}_train_files")),
            "validation": list(getattr(split_recorder, f"{name}_val_files")),
            "test": list(getattr(split_recorder, f"{name}_test_files")),
        }
        metrics, rows, fitted, eol_clip = evaluate_dataset(args.data_root, name, files)
        all_metrics[name] = metrics
        all_rows.append(rows)
        joblib.dump({
            "model": fitted,
            "dataset": name,
            "voltage_grid": GRID,
            "feature": "log10 variance of normalized Qd100(V)-Qd10(V)",
            "eol_clip": eol_clip,
        }, output / f"{name.lower()}_variance_model.joblib")
        print(f"{name}: {json.dumps(metrics)}", flush=True)
    pd.concat(all_rows, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps({
        "voltage_grid": [float(GRID[0]), float(GRID[-1]), len(GRID)],
        "feature": "log10 variance of normalized Qd100(V)-Qd10(V)",
        "test_status": "official tests viewed in prior work; exploratory",
        "metrics": all_metrics,
    }, indent=2), encoding="utf-8")
    print(f"RUN_DIR={output}", flush=True)


if __name__ == "__main__":
    main()
