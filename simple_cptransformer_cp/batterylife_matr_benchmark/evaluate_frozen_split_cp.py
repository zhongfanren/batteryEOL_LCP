"""Evaluate ordinary split CP using the eleven existing frozen MATR models.

The model-selection cells are within the original MATR training split. The
official validation cells supply calibration residuals, and the official test
cells are evaluated once per cell at T=100. These test results are exploratory
because the official test labels were previously inspected in this project.
"""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs" / "20261014_222508_198380"
OUT_ROOT = HERE / "frozen_split_cp"
MODELS = ("DLinear", "MLP", "CPMLP", "PatchTST", "iTransformer",
          "CPTransformer", "CNN", "CPGRU", "CPBiGRU", "CPLSTM", "CPBiLSTM")
ENSEMBLES = {"mean_all_11": MODELS,
             "mean_paper_cp_top3": ("CPMLP", "CPTransformer", "CPGRU")}
ALPHA = 0.1


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    split = json.loads((SOURCE / "development_split.json").read_text(encoding="utf-8"))
    fit = set(split["fit"])
    selection = set(split["selection"])
    if len(fit) != 90 or len(selection) != 12 or fit & selection:
        raise RuntimeError("Unexpected model fit/selection split")

    reference_ids = {}
    reference_labels = {}
    scores = []
    interval_rows = []
    predictions_by_model = {}
    manifest = {"source_run": str(SOURCE), "files": {}}
    rank = math.ceil((34 + 1) * (1 - ALPHA))
    for name in MODELS:
        model_dir = SOURCE / name
        paths = {"checkpoint": model_dir / "development_model.pt",
                 "calibration": model_dir / "val_predictions.csv",
                 "test": model_dir / "test_predictions.csv"}
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"Missing frozen model artifacts for {name}")
        manifest["files"][name] = {key: {"path": str(path), "sha256": sha256(path)}
                                   for key, path in paths.items()}
        cal = pd.read_csv(paths["calibration"])
        test = pd.read_csv(paths["test"])
        for phase, frame, count in (("calibration", cal, 34), ("test", test, 33)):
            if list(frame.columns) != ["cell_id", "eol", "prediction"]:
                raise RuntimeError(f"Unexpected {name} {phase} columns")
            if len(frame) != count or frame.cell_id.nunique() != count:
                raise RuntimeError(f"Unexpected {name} {phase} cell count")
            ids = frame.cell_id.tolist()
            labels = frame.eol.to_numpy(dtype=float)
            predictions = frame.prediction.to_numpy(dtype=float)
            if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
                raise RuntimeError(f"Nonfinite {name} {phase} values")
            if set(ids) & (fit | selection):
                raise RuntimeError(f"Training/selection leakage into {name} {phase}")
            if phase not in reference_ids:
                reference_ids[phase] = ids
                reference_labels[phase] = labels
            elif ids != reference_ids[phase] or not np.array_equal(labels, reference_labels[phase]):
                raise RuntimeError(f"Models disagree on {phase} cells or labels")
        if set(cal.cell_id) & set(test.cell_id):
            raise RuntimeError(f"Calibration/test overlap for {name}")
        predictions_by_model[name] = (cal.prediction.to_numpy(dtype=float),
                                      test.prediction.to_numpy(dtype=float))

        residuals = np.abs(cal.eol.to_numpy(dtype=float) - cal.prediction.to_numpy(dtype=float))
        q = float(np.partition(residuals, rank - 1)[rank - 1])
        y = test.eol.to_numpy(dtype=float)
        pred = test.prediction.to_numpy(dtype=float)
        lo, hi = pred - q, pred + q
        covered = (lo <= y) & (y <= hi)
        score = hi - lo + (2 / ALPHA) * np.maximum(lo - y, 0) + (2 / ALPHA) * np.maximum(y - hi, 0)
        scores.append({"model": name, "calibration_cells": len(cal), "test_cells": len(test),
                       "alpha": ALPHA, "conformal_rank": rank, "q_cycles": q,
                       "test_point_mape": float(np.mean(np.abs(y - pred) / y)),
                       "test_point_rmse_cycles": float(np.sqrt(np.mean((y - pred) ** 2))),
                       "test_coverage": float(covered.mean()),
                       "test_covered_cells": int(covered.sum()),
                       "test_mean_width_cycles": float(np.mean(hi - lo)),
                       "test_mean_interval_score": float(score.mean()),
                       "test_below_lower": int((y < lo).sum()),
                       "test_above_upper": int((y > hi).sum())})
        interval_rows.extend({"model": name, "cell_id": cell, "eol": float(actual),
                              "prediction": float(point), "lower": float(lower),
                              "upper": float(upper), "covered": bool(hit),
                              "interval_score": float(iscore)}
                             for cell, actual, point, lower, upper, hit, iscore in
                             zip(test.cell_id, y, pred, lo, hi, covered, score))

    manifest["ensembles"] = {name: list(members) for name, members in ENSEMBLES.items()}
    for name, members in ENSEMBLES.items():
        cal_pred = np.mean([predictions_by_model[member][0] for member in members], axis=0)
        pred = np.mean([predictions_by_model[member][1] for member in members], axis=0)
        cal_y = reference_labels["calibration"]
        y = reference_labels["test"]
        residuals = np.abs(cal_y - cal_pred)
        q = float(np.partition(residuals, rank - 1)[rank - 1])
        lo, hi = pred - q, pred + q
        covered = (lo <= y) & (y <= hi)
        score = hi - lo + (2 / ALPHA) * np.maximum(lo - y, 0) + (2 / ALPHA) * np.maximum(y - hi, 0)
        scores.append({"model": name, "ensemble_members": list(members),
                       "calibration_cells": len(cal_y), "test_cells": len(y),
                       "alpha": ALPHA, "conformal_rank": rank, "q_cycles": q,
                       "test_point_mape": float(np.mean(np.abs(y - pred) / y)),
                       "test_point_rmse_cycles": float(np.sqrt(np.mean((y - pred) ** 2))),
                       "test_coverage": float(covered.mean()),
                       "test_covered_cells": int(covered.sum()),
                       "test_mean_width_cycles": float(np.mean(hi - lo)),
                       "test_mean_interval_score": float(score.mean()),
                       "test_below_lower": int((y < lo).sum()),
                       "test_above_upper": int((y > hi).sum())})
        interval_rows.extend({"model": name, "cell_id": cell, "eol": float(actual),
                              "prediction": float(point), "lower": float(lower),
                              "upper": float(upper), "covered": bool(hit),
                              "interval_score": float(iscore)}
                             for cell, actual, point, lower, upper, hit, iscore in
                             zip(reference_ids["test"], y, pred, lo, hi, covered, score))

    out = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    result = {"protocol": "frozen MATR T=100, 90 fit / 12 epoch selection / 34 calibration / 33 test",
              "test_exploratory": True, "alpha": ALPHA, "models": scores}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(interval_rows).to_csv(out / "test_intervals.csv", index=False)
    print(out)
    for row in scores:
        print(f"{row['model']:15} MAPE={row['test_point_mape']:.4f} "
              f"coverage={row['test_coverage']:.3f} width={row['test_mean_width_cycles']:.1f}")


if __name__ == "__main__":
    main()
