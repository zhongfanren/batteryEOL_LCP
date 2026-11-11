"""Verify and fingerprint one trained MATR-held-out predictor for CP/LCP reuse."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from audit_mix_large import split_lists


FILES = ("best_model.pt", "config.json", "metrics.json",
         "source_split_cells.json", "matr_all_t100_predictions.csv")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def freeze(run_dir):
    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    source = json.loads((run_dir / "source_split_cells.json").read_text(encoding="utf-8"))
    original = split_lists()
    matr_splits = {phase: original[f"MATR_{phase}_files"] for phase in ("train", "val", "test")}
    matr_names = set().union(*(set(names) for names in matr_splits.values()))
    assert len(matr_names) == 169
    assert config["exclude_matr"] is True and config["subset_present"] is False
    assert metrics["matr_target_t100"]["n_cells"] == 169
    for phase in ("train", "val", "test"):
        expected = [name for name in original[f"MIX_large_{phase}_files"] if name not in matr_names]
        assert source[phase] == expected, f"Source {phase} split differs from non-MATR protocol"
    with (run_dir / "matr_all_t100_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 169 and {row["cell_file"] for row in rows} == matr_names
    split_by_name = {name: phase for phase, names in matr_splits.items() for name in names}
    for row in rows:
        assert int(row["prefix_cycles"]) == 100
        assert row["original_matr_split"] == split_by_name[row["cell_file"]]
        assert all(float(row[key]) == float(row[key]) for key in ("eol", "prediction"))
    manifest = {
        "state": "frozen_for_cp_lcp", "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir), "predictor": config["model"], "seed": config["seed"],
        "no_matr_in_training_validation_scaler_or_selection": True,
        "target_prediction_protocol": "one T=100 prediction per MATR cell",
        "matr_cell_counts": {phase: len(names) for phase, names in matr_splits.items()},
        "initial_calibration": "original MATR train+validation (136 cells)",
        "initial_test": "original MATR test (33 cells; exploratory because viewed previously)",
        "changing_matr_calibration_test_split_requires_predictor_retraining": False,
        "sha256": {name: digest(run_dir / name) for name in FILES},
    }
    output = run_dir / "frozen_predictor.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(freeze(parser.parse_args().run_dir))
