"""One-cell-per-example mixed Li-ion split-conformal pilot with CPMLP.

Run --prepare first to freeze a cell-level split before fitting. The CPMLP
training path is the same single-GPU adapter as run_single_gpu.py.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from audit_mix_large import split_lists
from run_single_gpu import (HERE, UPSTREAM, evaluate, official_config, seed_all,
                            upstream_loader)


SOURCES = ("UL_PUR", "RWTH", "HUST", "MATR", "Stanford", "Tongji",
           "ISU_ILCC", "CALCE", "HNEI", "SNL", "MICH", "MICH_EXP", "XJTU")
PHASE_SIZES = {"train": 450, "dev": 60, "calibration": 180, "test": 133}
DEFAULT_DIR = HERE / "mixed_liion_pilot"
PRIOR_COUNTS = HERE / "runs_single_gpu" / "20261021_114809_194841" / "cell_prefix_counts.json"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(out, counts_file, split_seed):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "split_manifest.json"
    if path.exists():
        raise FileExistsError(f"Refusing to replace frozen split: {path}")
    prefix_counts = json.loads(counts_file.read_text(encoding="utf-8"))
    cells = sorted({name for phase in prefix_counts.values()
                    for name, count in phase.items() if count == 100})
    names_by_source = split_lists()
    source_of = {}
    for source in SOURCES:
        for original_phase in ("train", "val", "test"):
            for name in names_by_source[f"{source}_{original_phase}_files"]:
                if name in source_of:
                    raise RuntimeError(f"Duplicate source mapping: {name}")
                source_of[name] = source
    if len(cells) != sum(PHASE_SIZES.values()) or any(n not in source_of for n in cells):
        raise RuntimeError(f"Expected {sum(PHASE_SIZES.values())} mapped eligible cells, got {len(cells)}")
    random.Random(split_seed).shuffle(cells)
    phases, start = {}, 0
    for phase, size in PHASE_SIZES.items():
        phases[phase] = cells[start:start + size]
        start += size
    manifest = {"description": "Unstratified cell-level shuffle of labeled MIX_large Li-ion cells with 100 usable cycles",
                "split_seed": split_seed, "train_seed": 2021, "model": "CPMLP",
                "alpha": 0.1, "prediction_cycle": 100,
                "source_counts_file": str(counts_file.resolve()),
                "source_counts_sha256": hashlib.sha256(counts_file.read_bytes()).hexdigest(),
                "source_of": {n: source_of[n] for n in sorted(cells)},
                "phases": phases, "phase_sizes": PHASE_SIZES,
                "interpretation": "Exploratory. Marginal split-CP guarantee requires exchangeable calibration and future test cells from the same prespecified mixture; source-conditional or shifted-domain coverage is not guaranteed."}
    write_json(path, manifest)
    print(f"FROZEN_SPLIT={path}", flush=True)
    print(json.dumps({p: {s: sum(source_of[n] == s for n in names)
                          for s in SOURCES if any(source_of[n] == s for n in names)}
                      for p, names in phases.items()}), flush=True)


def metrics_for(rows, lower_col, upper_col):
    covered = (rows.eol >= rows[lower_col]) & (rows.eol <= rows[upper_col])
    return {"n_cells": len(rows), "coverage": float(covered.mean()),
            "mean_width_cycles": float((rows[upper_col] - rows[lower_col]).mean()),
            "median_width_cycles": float((rows[upper_col] - rows[lower_col]).median())}


def run(out, root, epochs, train_seed):
    manifest_path = out / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phases = manifest["phases"]
    if set(phases) != set(PHASE_SIZES) or any(len(phases[p]) != PHASE_SIZES[p] for p in phases):
        raise RuntimeError("Frozen split sizes changed")
    all_names = sum((phases[p] for p in PHASE_SIZES), [])
    if len(all_names) != len(set(all_names)):
        raise RuntimeError("A cell appears in more than one split")
    if train_seed != manifest["train_seed"]:
        raise RuntimeError("Training seed differs from frozen manifest")
    if (out / "config.json").exists():
        raise FileExistsError("Pilot already started; refusing to overwrite")
    os.chdir(UPSTREAM)  # Upstream loader opens its life_classes.json relative to cwd.
    torch.set_num_threads(4)
    seed_all(train_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = official_config(root, "CPMLP")
    loader_module = upstream_loader()
    recorder = loader_module.split_recorder
    recorder.MIX_large_train_files = phases["train"]
    recorder.MIX_large_val_files = phases["dev"]
    recorder.MIX_large_test_files = phases["test"]
    Dataset = loader_module.Dataset_original
    print("Loading training cells and 1..100 prefixes", flush=True)
    train = Dataset(cfg, flag="train")
    print("Loading development cells and 1..100 prefixes", flush=True)
    dev = Dataset(cfg, flag="val", label_scaler=train.label_scaler,
                  life_class_scaler=train.life_class_scaler)
    if len(train) != 100 * len(phases["train"]) or len(dev) != 100 * len(phases["dev"]):
        raise RuntimeError(f"Unexpected train/dev prefixes: {len(train)}, {len(dev)}")
    source_path = UPSTREAM / "data_provider" / "data_loader.py"
    config = {"model": "CPMLP", "train_seed": train_seed, "epochs_cap": epochs,
              "model_config": vars(cfg), "data_root": str(root), "device": device,
              "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "upstream_loader_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "upstream_model_sha256": hashlib.sha256((UPSTREAM / "models" / "CPMLP.py").read_bytes()).hexdigest(),
              "training_prefixes": len(train), "dev_prefixes": len(dev),
              "adapter": "Same original loader and CPMLP class as run_single_gpu.py; one GPU with original effective batch via gradient accumulation; 0 data workers"}
    write_json(out / "config.json", config)
    print(f"RUN_DIR={out} device={device}", flush=True)
    model = importlib.import_module("models.CPMLP").Model(cfg).to(device)
    collate = loader_module.my_collate_fn_baseline
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, num_workers=0, collate_fn=collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    best_mape, stale, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for i, (x, mask, labels, *_rest) in enumerate(train_loader, 1):
            x, mask, labels = x.float().to(device), mask.float().to(device), labels.float().to(device)
            pred = model(x, mask)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = torch.mean((pred - labels) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            (loss / cfg.accumulation_steps).backward()
            if i % cfg.accumulation_steps == 0 or i == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        dev_metrics, _ = evaluate(model, dev, collate, device, cfg.batch_size)
        row = {"epoch": epoch, "train_scaled_mse": float(np.mean(losses)), **dev_metrics}
        history.append(row)
        write_json(out / "history.json", history)
        print(f"epoch={epoch} train_mse={row['train_scaled_mse']:.5f} dev_mape={dev_metrics['mape_fraction']:.5f}", flush=True)
        if dev_metrics["mape_fraction"] < best_mape:
            best_mape, best_epoch, stale = dev_metrics["mape_fraction"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "label_mean": float(train.label_scaler.mean_[0]),
                        "label_std": float(np.sqrt(train.label_scaler.var_[0]))}, out / "best_model.pt")
        elif epoch > cfg.least_epochs:
            stale += 1
        if stale >= cfg.patience:
            break
    checkpoint = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    del train, dev, train_loader
    output = {}
    for phase in ("calibration", "test"):
        recorder.MIX_large_val_files = phases[phase]
        print(f"Loading frozen-model {phase} cells at T=100", flush=True)
        dataset = Dataset(cfg, flag="val", label_scaler=checkpoint_scaler(checkpoint),
                          life_class_scaler=None, eval_cycle_max=100, eval_cycle_min=100)
        if len(dataset) != len(phases[phase]) or any(v != 1 for v in dataset.cell_prefix_counts.values()):
            raise RuntimeError(f"Unexpected {phase} T=100 sample count")
        point_metrics, rows = evaluate(model, dataset, collate, device, cfg.batch_size)
        rows["prefix_cycles"] = 100
        rows["source"] = rows.cell_file.map(manifest["source_of"])
        if rows.source.isna().any():
            raise RuntimeError("Source mapping failed")
        output[phase] = (point_metrics, rows)
        rows.to_csv(out / f"{phase}_predictions.csv", index=False)
        del dataset
    cal = output["calibration"][1]
    test = output["test"][1]
    cal_scores = np.abs(cal.eol.to_numpy() - cal.prediction.to_numpy())
    rank = math.ceil((len(cal_scores) + 1) * (1 - manifest["alpha"]))
    q = float(np.sort(cal_scores)[rank - 1]) if rank <= len(cal_scores) else float("inf")
    test["split_cp_lower"] = np.maximum(0, test.prediction - q)
    test["split_cp_upper"] = test.prediction + q
    test["split_cp_covered"] = ((test.eol >= test.split_cp_lower) & (test.eol <= test.split_cp_upper))
    # A fixed, unsupervised prediction-space kernel is an exploratory LCP-style
    # comparator. It has no finite-sample conformal coverage claim.
    bandwidth = float(np.std(cal.prediction.to_numpy(), ddof=1))
    if bandwidth <= 0:
        raise RuntimeError("Cannot compute local bandwidth")
    for idx in test.index:
        pred = float(test.at[idx, "prediction"])
        weights = np.exp(-0.5 * ((cal.prediction.to_numpy() - pred) / bandwidth) ** 2)
        order = np.argsort(cal_scores)
        cumulative = np.cumsum(weights[order])
        local_rank = np.searchsorted(cumulative, (1 - manifest["alpha"]) * (weights.sum() + 1.0))
        local_q = float(cal_scores[order[local_rank]]) if local_rank < len(order) else float("inf")
        test.at[idx, "local_q"] = local_q
    test["local_lower"] = np.maximum(0, test.prediction - test.local_q)
    test["local_upper"] = test.prediction + test.local_q
    test["local_covered"] = ((test.eol >= test.local_lower) & (test.eol <= test.local_upper))
    test.to_csv(out / "test_intervals.csv", index=False)
    by_source = {}
    for source, group in test.groupby("source"):
        by_source[source] = {"split_cp": metrics_for(group, "split_cp_lower", "split_cp_upper"),
                             "local": metrics_for(group, "local_lower", "local_upper"),
                             "point_mape_fraction": float(np.mean(np.abs(group.eol - group.prediction) / group.eol))}
    result = {"best_epoch": best_epoch, "calibration": output["calibration"][0],
              "test_point": output["test"][0], "split_cp": {
                  "alpha": manifest["alpha"], "calibration_n": len(cal), "rank": rank,
                  "radius_cycles": q, **metrics_for(test, "split_cp_lower", "split_cp_upper")},
              "local_exploratory": {"kernel": "Gaussian on frozen point prediction",
                                    "bandwidth_cycles": bandwidth,
                                    "formal_coverage_guarantee": False,
                                    **metrics_for(test, "local_lower", "local_upper")},
              "by_source": by_source, "test_exploratory": True,
              "coverage_assumption": "Split-CP marginal coverage requires calibration and future test cells exchangeable from this same mixture, conditional on the frozen predictor. Cell disjointness alone does not establish this; no per-source or shifted-domain guarantee."}
    write_json(out / "results.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def checkpoint_scaler(checkpoint):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.array([checkpoint["label_mean"]])
    scaler.var_ = np.array([checkpoint["label_std"] ** 2])
    scaler.scale_ = np.array([checkpoint["label_std"]])
    scaler.n_features_in_ = 1
    scaler.n_samples_seen_ = 1
    return scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--counts-file", type=Path, default=PRIOR_COUNTS)
    parser.add_argument("--split-seed", type=int, default=20261021)
    parser.add_argument("--train-seed", type=int, default=2021)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\BatteryLife_v4"))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        prepare(out, args.counts_file.resolve(), args.split_seed)
    else:
        run(out, args.data_root.resolve(), args.epochs, args.train_seed)


if __name__ == "__main__":
    main()
