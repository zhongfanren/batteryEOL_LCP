"""Development-only five-fold MATR comparison for shortlisted BatteryLife models."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import KFold

from run_matr_models import HERE, prepare, train_one, train_fixed_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    parser.add_argument("--models", nargs="+", default=["CPMLP", "CPLSTM", "CPBiGRU", "PatchTST", "DLinear"])
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.evaluate_official = False
    out = HERE / "nested_cv_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    data = prepare(args.data_root, HERE / "cache" / "matr_t100.npz")
    y, ids = data["train"][1:]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=2024).split(y))
    summary = {}
    for name in args.models:
        oof = np.full(len(y), np.nan)
        fold_results = []
        for fold, (fit, check) in enumerate(folds):
            fold_dir = out / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)
            print(f"START {name} fold {fold}", flush=True)
            inner = fit.copy()
            np.random.default_rng(2024 + fold).shuffle(inner)
            inner_select, inner_fit = inner[:8], inner[8:]
            selected = train_one(name, data, inner_fit, inner_select, args, fold_dir / "inner")
            prediction = train_fixed_epoch(name, data, fit, check, args,
                                           selected["best_epoch"], fold_dir / "outer" / name)
            oof[check] = prediction
            fold_score = {"n": len(check),
                          "rmse": float(np.sqrt(mean_squared_error(y[check], prediction))),
                          "mae": float(mean_absolute_error(y[check], prediction)),
                          "mape": float(mean_absolute_percentage_error(y[check], prediction))}
            fold_results.append({"best_epoch": selected["best_epoch"], "development": fold_score})
            print(f"DONE {name} fold {fold}: {fold_score}", flush=True)
        if not np.isfinite(oof).all():
            raise ValueError(f"missing OOF predictions for {name}")
        score = {"rmse": float(np.sqrt(mean_squared_error(y, oof))),
                 "mae": float(mean_absolute_error(y, oof)),
                 "mape": float(mean_absolute_percentage_error(y, oof))}
        summary[name] = {"n": len(y), "oof": score,
                         "fold_best_epochs": [r["best_epoch"] for r in fold_results],
                         "fold_metrics": [r["development"] for r in fold_results]}
        with (out / f"{name}_oof.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("cell_id", "eol", "prediction"))
            writer.writerows(zip(ids, y, oof))
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "summary": {k: v["oof"] for k, v in summary.items()}}), flush=True)


if __name__ == "__main__":
    main()
