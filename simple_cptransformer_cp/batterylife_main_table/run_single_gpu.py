"""Single-GPU BatteryLife model runs with the original MIX_large prefix loader.

The upstream loader is adapted in memory for unavailable imports, an unlabeled
cell return bug, and a full-split NaN check that exhausts RAM. Its prefix and
curve construction remain upstream. The model class remains unchanged.
"""

import argparse
import ast
import hashlib
import importlib
import json
import os
import pickle
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from audit_mix_large import audit


HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parents[1] / "third_party" / "BatteryLife"
sys.path.insert(0, str(UPSTREAM))
MODEL_NAMES = ("DLinear", "MLP", "CPMLP", "PatchTST", "iTransformer",
               "CPTransformer", "CNN", "CPGRU", "CPBiGRU", "CPLSTM", "CPBiLSTM")
SELECTED_LI_ION = {
    "CPMLP": dict(batch_size=16, d_model=32, d_ff=256, e_layers=12,
                  d_layers=7, dropout=0, learning_rate=5e-5),
    "CPTransformer": dict(batch_size=128, d_model=256, d_ff=64, e_layers=1,
                          d_layers=12, dropout=0, learning_rate=5e-5),
    "CPGRU": dict(batch_size=8, d_model=32, d_ff=32, e_layers=9,
                  d_layers=2, dropout=0.05, learning_rate=5e-4),
    "CPLSTM": dict(batch_size=8, d_model=128, d_ff=256, e_layers=10,
                   d_layers=2, dropout=0.05, learning_rate=5e-5),
}
MICRO_BATCH_OVERRIDES = {"CPTransformer": 64, "CNN": 16}
CACHE_DIR = HERE / "processed_dataset_cache"
CACHE_FIELDS = ("total_charge_discharge_curves", "total_curve_attn_masks",
                "total_labels", "unique_labels", "class_labels", "total_dataset_ids",
                "total_cj_aug_charge_discharge_curves", "total_seen_unseen_IDs",
                "KDE_samples", "weights", "raw_labels", "label_scaler",
                "life_class_scaler", "scaled_life_classes", "files",
                "cell_prefix_counts", "flag", "dataset")


def upstream_loader():
    source = UPSTREAM / "data_provider" / "data_loader.py"
    loader_source = source.read_text(encoding="utf-8")
    # Each cell's curve array is referenced by up to 100 prefix samples. The
    # original np.isnan(list_of_prefix_arrays) materializes all of them into
    # one array (tens of GB on MIX_large). Check each distinct array once.
    original_nan_check = "if np.any(np.isnan(self.total_charge_discharge_curves)):"
    distinct_nan_check = ("if any(np.isnan(curves).any() for curves in "
                          "{id(curves): curves for curves in self.total_charge_discharge_curves}.values()):")
    if loader_source.count(original_nan_check) != 1:
        raise RuntimeError("Upstream full-split NaN check changed; inspect adapter")
    loader_source = loader_source.replace(original_nan_check, distinct_nan_check)
    tree = ast.parse(loader_source, filename=str(source))
    skipped = []
    kept = []
    for node in tree.body:
        module = node.module if isinstance(node, ast.ImportFrom) else None
        imported = [a.name for a in node.names] if isinstance(node, ast.Import) else []
        if module in {"batteryml.data.battery_data", "denseweight"} or imported == ["accelerate"]:
            skipped.append(module or imported[0])
        else:
            kept.append(node)
    assert set(skipped) == {"batteryml.data.battery_data", "accelerate", "denseweight"}
    tree.body = kept
    module = ModuleType("battery_life_loader_single_gpu")
    module.__file__ = str(source)
    exec(compile(tree, str(source), "exec"), module.__dict__)
    dataset_class = module.Dataset_original
    original_read_cell_df = dataset_class.read_cell_df

    def six_value_cell_df(self, filename):
        result = original_read_cell_df(self, filename)
        # Upstream returns five Nones for an unlabeled cell although its caller
        # always unpacks six values. v4 has six such MICH_EXP cells.
        if len(result) == 5 and result[0] is None:
            return (*result, None)
        return result

    dataset_class.read_cell_df = six_value_cell_df
    original = dataset_class.read_samples_from_one_cell

    def recording_read(self, filename):
        result = original(self, filename)
        if not hasattr(self, "cell_prefix_counts"):
            self.cell_prefix_counts = {}
        self.cell_prefix_counts[filename] = 0 if result[0] is None else len(result[0])
        return result

    dataset_class.read_samples_from_one_cell = recording_read
    return module


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def official_config(root, model_name):
    # Read the supplied launcher, then apply published v4 Li-ion rows where
    # available. One-GPU accumulation preserves its two-GPU effective batch.
    cfg = SimpleNamespace(root_path=str(root), dataset="MIX_large", seq_len=1,
        early_cycle_threshold=100, charge_discharge_length=300,
        weighted_loss=False, task_name="classification", pred_len=5,
        label_len=50, enc_in=3, dec_in=1, c_out=1, output_num=1,
        class_num=8, moving_avg=25, factor=3, activation="relu",
        embed="timeF", freq="h", output_attention=False,
        d_model=16, d_ff=32, e_layers=2, d_layers=1, lstm_layers=1,
        n_heads=4, dropout=0.1, patch_len=10, stride=10,
        patch_len2=10, stride2=10, batch_size=32, learning_rate=1e-4,
        accumulation_steps=1, train_epochs=100, least_epochs=5, patience=5)
    script = UPSTREAM / "train_eval_scripts" / f"{model_name}.sh"
    if not script.is_file():
        raise FileNotFoundError(script)
    numeric = {"batch_size", "learning_rate", "accumulation_steps", "d_model",
               "d_ff", "e_layers", "d_layers", "lstm_layers", "n_heads",
               "dropout", "patch_len", "stride", "patch_len2", "stride2",
               "moving_avg", "patience", "train_epochs"}
    for line in script.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([a-z_0-9]+)=([^ #]+)", line)
        if match and match.group(1) in numeric:
            key, raw = match.groups()
            setattr(cfg, key, float(raw) if key in {"learning_rate", "dropout"} else int(raw))
    for key, value in SELECTED_LI_ION.get(model_name, {}).items():
        setattr(cfg, key, value)
    per_gpu_batch, per_gpu_steps = cfg.batch_size, cfg.accumulation_steps
    cfg.batch_size = MICRO_BATCH_OVERRIDES.get(model_name, per_gpu_batch)
    effective_batch = 2 * per_gpu_batch * per_gpu_steps
    if effective_batch % cfg.batch_size:
        raise RuntimeError(f"Cannot preserve effective batch for {model_name}")
    cfg.accumulation_steps = effective_batch // cfg.batch_size
    return cfg


def load_or_build_dataset(dataset_class, cfg, flag, recorder, label_scaler=None,
                          life_class_scaler=None):
    """Cache only the original loader's completed data arrays and metadata."""
    listed_files = list(getattr(recorder, f"MIX_large_{flag}_files"))
    identity = {"format": 1, "root": str(Path(cfg.root_path).resolve()),
                "dataset": cfg.dataset, "flag": flag, "files": listed_files,
                "loader_sha256": hashlib.sha256((UPSTREAM / "data_provider" / "data_loader.py").read_bytes()).hexdigest(),
                "split_sha256": hashlib.sha256((UPSTREAM / "data_provider" / "data_split_recorder.py").read_bytes()).hexdigest(),
                "life_classes_sha256": hashlib.sha256((UPSTREAM / "data_provider" / "life_classes.json").read_bytes()).hexdigest(),
                "early_cycle_threshold": cfg.early_cycle_threshold,
                "charge_discharge_length": cfg.charge_discharge_length,
                "seq_len": cfg.seq_len,
                "label_scaler_mean": (float(label_scaler.mean_[0]) if label_scaler is not None else None),
                "label_scaler_var": (float(label_scaler.var_[0]) if label_scaler is not None else None)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = CACHE_DIR / f"mix_large_{flag}_{key}.pkl"
    if path.is_file():
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload["identity"] != identity:
            raise RuntimeError(f"Preprocessing cache identity mismatch: {path}")
        result = dataset_class.__new__(dataset_class)
        result.__dict__.update(payload["fields"])
        if result.files != listed_files or len(result) != len(result.total_charge_discharge_curves):
            raise RuntimeError(f"Preprocessing cache validation failed: {path}")
        print(f"loaded preprocessing cache {path}", flush=True)
        return result, path
    result = dataset_class(cfg, flag=flag, label_scaler=label_scaler,
                           life_class_scaler=life_class_scaler)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"identity": identity,
               "fields": {name: getattr(result, name) for name in CACHE_FIELDS}}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    print(f"saved preprocessing cache {path}", flush=True)
    return result, path


def evaluate(model, dataset, collate, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate)
    model.eval()
    y_true, y_pred = [], []
    mean = float(dataset.label_scaler.mean_[0])
    std = float(np.sqrt(dataset.label_scaler.var_[0]))
    with torch.inference_mode():
        for x, mask, labels, *_ in loader:
            pred = model(x.float().to(device), mask.float().to(device))
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.reshape(-1)
            y_pred.extend((pred.cpu().numpy() * std + mean).tolist())
            y_true.extend((labels.numpy().reshape(-1) * std + mean).tolist())
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    error = np.abs(y_true - y_pred)
    relative = error / y_true
    metrics = {"prefix_samples": len(y_true), "rmse_cycles": float(np.sqrt(np.mean(error**2))),
               "mae_cycles": float(np.mean(error)), "mape_fraction": float(np.mean(relative)),
               "accuracy_15_fraction": float(np.mean(relative <= 0.15))}
    names = [name for name in dataset.files for _ in range(dataset.cell_prefix_counts[name])]
    prefixes = [i for name in dataset.files for i in range(1, dataset.cell_prefix_counts[name] + 1)]
    assert len(names) == len(y_true)
    rows = pd.DataFrame({"cell_file": names, "prefix_cycles": prefixes,
                         "eol": y_true, "prediction": y_pred})
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\BatteryLife_v4"))
    parser.add_argument("--model", choices=MODEL_NAMES, default="DLinear")
    parser.add_argument("--subset-present", action="store_true",
                        help="Smoke test only: use cells already present; never call this a main-table result.")
    parser.add_argument("--exclude-matr", action="store_true",
                        help="Train/validate on non-MATR MIX_large cells; predict MATR only after freezing.")
    parser.add_argument("--smoke-loader", action="store_true", help="Load one cell and run two model predictions.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.exclude_matr and (args.subset_present or args.smoke_loader):
        parser.error("--exclude-matr cannot be combined with smoke or partial-data modes")
    args.data_root = args.data_root.resolve()
    # The upstream loader opens data_provider/life_classes.json relative to cwd.
    os.chdir(UPSTREAM)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(args.seed)
    cfg = official_config(args.data_root, args.model)
    source = UPSTREAM / "data_provider" / "data_loader.py"
    loader_module = upstream_loader()
    recorder = loader_module.split_recorder
    if args.smoke_loader:
        recorder.MIX_large_train_files = list(recorder.HUST_train_files[:1])
    elif args.subset_present:
        for phase in ("train", "val", "test"):
            key = f"MIX_large_{phase}_files"
            setattr(recorder, key, [n for n in getattr(recorder, key)
                    if (args.data_root / ({"UL-PUR": "UL_PUR", "ISU-ILCC": "ISU_ILCC",
                                           "Tongji1": "Tongji", "Tongji2": "Tongji",
                                           "Tongji3": "Tongji", "MICH": "MICH"}.get(n.split("_")[0], n.split("_")[0])) / n).is_file()])
    elif args.exclude_matr:
        matr_names = set(recorder.MATR_train_files + recorder.MATR_val_files + recorder.MATR_test_files)
        if len(matr_names) != 169:
            raise RuntimeError(f"Expected 169 unique MATR cells, found {len(matr_names)}")
        for phase in ("train", "val", "test"):
            key = f"MIX_large_{phase}_files"
            filtered = [name for name in getattr(recorder, key) if name not in matr_names]
            if len(filtered) != len(getattr(recorder, key)) - len(getattr(recorder, f"MATR_{phase}_files")):
                raise RuntimeError(f"MATR exclusion count mismatch in {phase}")
            setattr(recorder, key, filtered)
        if any(matr_names.intersection(getattr(recorder, f"MIX_large_{phase}_files"))
               for phase in ("train", "val", "test")):
            raise RuntimeError("MATR remains in a source-domain split")
        print("MATR held out from training, validation, scaler fitting and source test", flush=True)
        inventory = audit(args.data_root)
        if any(inventory["phases"][phase]["files_present"] != inventory["phases"][phase]["expected"]
               for phase in ("train", "val", "test")):
            raise RuntimeError("Incomplete source data for MATR-held-out training")
    else:
        inventory = audit(args.data_root)
        # v4 contains six MICH_EXP files without EOL labels. The original
        # read_samples_from_one_cell skips these after loading the cell file.
        expected_labeled = {"train": 510, "val": 165, "test": 162}
        for phase in ("train", "val", "test"):
            row = inventory["phases"][phase]
            if row["files_present"] != row["expected"] or row["labels_present"] != expected_labeled[phase]:
                raise RuntimeError(f"Incomplete MIX_large {phase}: {row}")
    dataset_class = loader_module.Dataset_original
    print("loading train prefixes", flush=True)
    if args.smoke_loader or args.subset_present:
        train = dataset_class(cfg, flag="train")
        train_cache = None
    else:
        train, train_cache = load_or_build_dataset(dataset_class, cfg, "train", recorder)
    train_included = sum(n > 0 for n in train.cell_prefix_counts.values())
    print(f"train cells={train_included}/{len(train.cell_prefix_counts)} prefixes={len(train)}", flush=True)
    model = importlib.import_module(f"models.{args.model}").Model(cfg).to(device)
    if args.smoke_loader:
        batch = next(iter(DataLoader(train, batch_size=cfg.batch_size,
                                     collate_fn=loader_module.my_collate_fn_baseline)))
        pred = model(batch[0].float().to(device), batch[1].float().to(device))
        if isinstance(pred, tuple):
            pred = pred[0]
        loss = torch.mean((pred.reshape(-1) - batch[2].float().reshape(-1).to(device)) ** 2)
        loss.backward()
        peak_mb = round(torch.cuda.max_memory_allocated() / 1024**2) if device == "cuda" else 0
        print(f"smoke model output shape={tuple(pred.shape)} device={device} "
              f"batch={len(batch[0])} peak_cuda_mb={peak_mb}", flush=True)
        return
    print("loading validation prefixes", flush=True)
    if args.subset_present:
        val = dataset_class(cfg, flag="val", label_scaler=train.label_scaler,
                            life_class_scaler=train.life_class_scaler)
        val_cache = None
    else:
        val, val_cache = load_or_build_dataset(dataset_class, cfg, "val", recorder,
                                               train.label_scaler, train.life_class_scaler)
    val_included = sum(n > 0 for n in val.cell_prefix_counts.values())
    print(f"validation cells={val_included}/{len(val.cell_prefix_counts)} prefixes={len(val)}", flush=True)
    output_family = "runs_no_matr" if args.exclude_matr else "runs_single_gpu"
    out = args.out or HERE / output_family / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    config = {"model": args.model, "seed": args.seed, "data_root": str(args.data_root.resolve()),
              "subset_present": args.subset_present, "exclude_matr": args.exclude_matr,
              "epochs": args.epochs, "model_config": vars(cfg),
              "device": device, "train_cells_included": train_included,
              "validation_cells_included": val_included,
              "train_cells_listed": len(train.cell_prefix_counts),
              "validation_cells_listed": len(val.cell_prefix_counts),
              "train_prefixes": len(train), "validation_prefixes": len(val),
              "preprocessing_cache": {"train": str(train_cache) if train_cache else None,
                                      "validation": str(val_cache) if val_cache else None},
              "data_loader_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256((UPSTREAM / "models" / f"{args.model}.py").read_bytes()).hexdigest(),
              "adapter": ("3 unused imports omitted; six-value unlabeled-cell return fix; "
                          "per-distinct-cell NaN check; single GPU; original two-GPU effective "
                          "batch preserved by gradient accumulation; 0 workers")}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.exclude_matr:
        source_splits = {phase: list(getattr(recorder, f"MIX_large_{phase}_files"))
                         for phase in ("train", "val", "test")}
        (out / "source_split_cells.json").write_text(json.dumps(source_splits, indent=2), encoding="utf-8")
    print(f"RUN_DIR={out}", flush=True)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=0, collate_fn=loader_module.my_collate_fn_baseline)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    best_mape, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for i, (x, mask, labels, *_rest) in enumerate(train_loader, start=1):
            x, mask, labels = x.float().to(device), mask.float().to(device), labels.float().to(device)
            pred = model(x, mask)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = torch.mean((pred - labels)**2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            (loss / cfg.accumulation_steps).backward()
            if i % cfg.accumulation_steps == 0 or i == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        metrics, _ = evaluate(model, val, loader_module.my_collate_fn_baseline, device, cfg.batch_size)
        row = {"epoch": epoch, "train_scaled_mse": float(np.mean(losses)), **metrics}
        history.append(row)
        print(f"epoch={epoch} train_mse={row['train_scaled_mse']:.5f} val_mape={metrics['mape_fraction']:.5f}", flush=True)
        if metrics["mape_fraction"] < best_mape:
            best_mape = metrics["mape_fraction"]
            best_epoch = epoch
            stale = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "label_mean": float(train.label_scaler.mean_[0]),
                        "label_std": float(np.sqrt(train.label_scaler.var_[0]))}, out / "best_model.pt")
        elif epoch > cfg.least_epochs:
            stale += 1
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= cfg.patience:
            break
    checkpoint = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    val_metrics, val_rows = evaluate(model, val, loader_module.my_collate_fn_baseline, device, cfg.batch_size)
    val_rows.to_csv(out / "validation_predictions.csv", index=False)
    print("loading test prefixes", flush=True)
    if args.subset_present:
        test = dataset_class(cfg, flag="test", label_scaler=train.label_scaler,
                             life_class_scaler=train.life_class_scaler)
        test_cache = None
    else:
        test, test_cache = load_or_build_dataset(dataset_class, cfg, "test", recorder,
                                                 train.label_scaler, train.life_class_scaler)
    config["preprocessing_cache"]["test"] = str(test_cache) if test_cache else None
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "cell_prefix_counts.json").write_text(json.dumps({
        "train": train.cell_prefix_counts, "validation": val.cell_prefix_counts,
        "test": test.cell_prefix_counts}, indent=2), encoding="utf-8")
    test_metrics, test_rows = evaluate(model, test, loader_module.my_collate_fn_baseline, device, cfg.batch_size)
    test_rows.to_csv(out / "test_predictions.csv", index=False)
    target_result = None
    if args.exclude_matr:
        # All 169 MATR cells are evaluated only after checkpoint selection on
        # non-MATR validation. Exactly one prefix (cycle 100) is made per cell.
        matr_splits = {phase: list(getattr(recorder, f"MATR_{phase}_files"))
                       for phase in ("train", "val", "test")}
        recorder.MATR_val_files = sum((matr_splits[phase] for phase in ("train", "val", "test")), [])
        matr_cfg = SimpleNamespace(**vars(cfg))
        matr_cfg.dataset = "MATR"
        print("loading frozen-model MATR T=100 predictions", flush=True)
        matr = dataset_class(matr_cfg, flag="val", label_scaler=train.label_scaler,
                             life_class_scaler=train.life_class_scaler,
                             eval_cycle_max=100, eval_cycle_min=100)
        if len(matr) != len(recorder.MATR_val_files) or any(
                count != 1 for count in matr.cell_prefix_counts.values()):
            raise RuntimeError("MATR fixed T=100 must yield one prefix for every cell")
        matr_metrics, matr_rows = evaluate(model, matr, loader_module.my_collate_fn_baseline,
                                          device, cfg.batch_size)
        matr_rows["prefix_cycles"] = 100
        split_by_name = {name: phase for phase, names in matr_splits.items() for name in names}
        matr_rows["original_matr_split"] = matr_rows["cell_file"].map(split_by_name)
        if matr_rows["original_matr_split"].isna().any():
            raise RuntimeError("MATR split mapping failed")
        target_path = out / "matr_all_t100_predictions.csv"
        matr_rows.to_csv(target_path, index=False)
        target_result = {"n_cells": len(matr_rows), "T": 100, "all_cell_metrics": matr_metrics,
                         "predictions": str(target_path), "test_exploratory": True,
                         "selection_uses_matr": False}
    result = {"best_epoch": best_epoch, "validation": val_metrics, "test": test_metrics,
              "test_exploratory": True,
              "full_official_split": not args.subset_present and not args.exclude_matr,
              "reproduction_type": ("MATR-held-out source-trained" if args.exclude_matr else
                                    "single-GPU approximate" if not args.subset_present else
                                    "partial-data pipeline trial")}
    if target_result is not None:
        result["matr_target_t100"] = target_result
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.exclude_matr:
        from freeze_no_matr import freeze
        print(f"FROZEN_MANIFEST={freeze(out)}", flush=True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
