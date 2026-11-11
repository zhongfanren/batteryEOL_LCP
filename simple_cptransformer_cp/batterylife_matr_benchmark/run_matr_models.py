"""MATR-only fixed-T=100 comparison using BatteryLife model classes.

The original BatteryLife loader enumerates many prefixes per cell and its
training entry point needs optional distributed dependencies.  This runner
uses the already audited fixed-T preprocessing adapter in run_t100.py, the
official MATR cell lists/labels, and unmodified upstream model classes.
Official validation/test are never used for epoch or model selection here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
UPSTREAM = PROJECT.parent / "third_party" / "BatteryLife"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(UPSTREAM))
from run_t100 import load_cell  # noqa: E402
from data_provider.data_split_recorder import split_recorder  # noqa: E402

ALL_MODELS = (
    "DLinear", "MLP", "CPMLP", "PatchTST", "Autoformer", "iTransformer",
    "Transformer", "CPTransformer", "CNN", "MICN", "CPGRU", "CPBiGRU",
    "CPLSTM", "CPBiLSTM", "GRU", "BiGRU", "LSTM", "BiLSTM",
)
FEASIBLE_MODELS = (
    "DLinear", "MLP", "CPMLP", "PatchTST", "iTransformer",
    "CPTransformer", "CNN", "CPGRU", "CPBiGRU", "CPLSTM", "CPBiLSTM",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=list(FEASIBLE_MODELS))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate-official", action="store_true",
                        help="After development-only selection, save official val/test predictions (exploratory).")
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare(data_root, cache_path):
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return {split: (cached[f"{split}_x"], cached[f"{split}_y"], cached[f"{split}_id"].tolist())
                for split in ("train", "val", "test")}
    labels = json.loads((data_root / "Life labels" / "MATR_labels.json").read_text(encoding="utf-8"))
    splits = {"train": split_recorder.MATR_train_files,
              "val": split_recorder.MATR_val_files,
              "test": split_recorder.MATR_test_files}
    if len(set(sum((list(v) for v in splits.values()), []))) != sum(map(len, splits.values())):
        raise ValueError("MATR official splits overlap")
    data, audit, arrays = {}, {}, {}
    for split, names in splits.items():
        xs, ys, ids, excluded = [], [], [], []
        for name in names:
            try:
                if name not in labels:
                    raise ValueError("missing life label")
                y = float(labels[name])
                x = load_cell(data_root / "MATR" / name, y)
                xs.append(x)
                ys.append(y)
                ids.append(name.removesuffix(".pkl"))
            except (FileNotFoundError, ValueError, IndexError) as exc:
                excluded.append({"cell": name, "reason": str(exc)})
        if not xs:
            raise ValueError(f"{split} has no usable cells")
        data[split] = (np.stack(xs), np.asarray(ys, np.float32), ids)
        arrays.update({f"{split}_x": data[split][0], f"{split}_y": data[split][1],
                       f"{split}_id": np.asarray(ids)})
        audit[split] = {"requested": len(names), "included": ids, "excluded": excluded}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    (cache_path.parent / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return data


def config_for(name):
    # Start from the upstream model's launch script.  For the four CP models
    # published Li-ion rows, use those rows as the model-size/learning-rate
    # reference instead of template scripts that may target CALB.
    common = dict(task_name="classification", seq_len=1, pred_len=1, label_len=50,
                  charge_discharge_length=300, early_cycle_threshold=100,
                  enc_in=3, dec_in=1, c_out=1, output_num=1, num_class=8,
                  d_model=64, d_ff=128, e_layers=2, d_layers=2, lstm_layers=2,
                  n_heads=4, factor=3, dropout=0.0, activation="relu",
                  moving_avg=25, embed="timeF", freq="h", patch_len=300,
                  stride=300, patch_len2=10, stride2=10)
    script = UPSTREAM / "train_eval_scripts" / f"{name}.sh"
    if script.exists():
        allowed = {"d_model", "d_ff", "e_layers", "d_layers", "lstm_layers",
                   "n_heads", "dropout", "patch_len", "stride", "moving_avg"}
        for line in script.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([a-z_]+)=([^ #]+)", line)
            if not match:
                continue
            key, value = match.groups()
            if key in allowed:
                common[key] = float(value) if key == "dropout" else int(value)
            elif key == "learning_rate":
                common["lr"] = float(value)
    special = {
        "CPMLP": dict(d_model=32, d_ff=256, e_layers=12, d_layers=7, lr=5e-5),
        "CPTransformer": dict(d_model=256, d_ff=64, e_layers=1, d_layers=12, lr=5e-5),
        "CPGRU": dict(d_model=32, d_ff=32, e_layers=9, d_layers=2, dropout=.05, lr=5e-4),
        "CPLSTM": dict(d_model=128, d_ff=256, e_layers=10, d_layers=2, dropout=.05, lr=5e-5),
        "DLinear": dict(lr=5e-4),
    }
    common.update(special.get(name, {}))
    lr = common.pop("lr", 5e-5)
    return SimpleNamespace(**common), lr


def score(y, pred):
    return {"n": len(y), "rmse": float(math.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "mape": float(mean_absolute_percentage_error(y, pred))}


def predict(model, x, scaler, device, batch_size):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size):
            xb = xb.to(device)
            mask = torch.ones((len(xb), 100), device=device)
            output = model(xb, mask)
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output.reshape(-1).cpu().numpy())
    return scaler.inverse_transform(np.concatenate(outputs)[:, None]).ravel()


def train_one(name, data, fit_indices, check_indices, args, out):
    seed_all(args.seed)
    config, lr = config_for(name)
    model = importlib.import_module(f"models.{name}").Model(config).to(args.device)
    x, y, ids = data["train"]
    scaler = StandardScaler().fit(y[fit_indices, None])
    scaled_y = scaler.transform(y[fit_indices, None]).astype(np.float32).ravel()
    loader = DataLoader(TensorDataset(torch.from_numpy(x[fit_indices]), torch.from_numpy(scaled_y)),
                        batch_size=args.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mape, best_epoch, best_state, stale, history = float("inf"), 0, None, 0, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb, torch.ones((len(xb), 100), device=args.device))
            if isinstance(output, tuple):
                output = output[0]
            loss = torch.mean((output.reshape(-1) - yb) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        pred = predict(model, x[check_indices], scaler, args.device, args.batch_size)
        current = score(y[check_indices], pred)
        history.append({"epoch": epoch, "train_scaled_mse": float(np.mean(losses)), **current})
        if current["mape"] < best_mape:
            best_mape, best_epoch = current["mape"], epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch >= 5 and stale >= args.patience:
            break
    model.load_state_dict(best_state)
    pred = predict(model, x[check_indices], scaler, args.device, args.batch_size)
    model_dir = out / name
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "development_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("cell_id", "eol", "prediction"))
        writer.writerows(zip(np.asarray(ids)[check_indices], y[check_indices], pred))
    torch.save({"state_dict": best_state, "model_config": vars(config),
                "label_mean": float(scaler.mean_[0]), "label_scale": float(scaler.scale_[0]),
                "fit_cell_ids": np.asarray(ids)[fit_indices].tolist(), "best_epoch": best_epoch},
               model_dir / "development_model.pt")
    (model_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    result = {"model": name, "status": "ok", "development": score(y[check_indices], pred),
              "best_epoch": best_epoch, "elapsed_seconds": round(time.time() - started, 1),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "learning_rate": lr, "config": vars(config)}
    if args.evaluate_official:
        result["official"] = {}
        for split in ("val", "test"):
            xs, ys, cell_ids = data[split]
            prediction = predict(model, xs, scaler, args.device, args.batch_size)
            result["official"][split] = score(ys, prediction)
            with (model_dir / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("cell_id", "eol", "prediction"))
                writer.writerows(zip(cell_ids, ys, prediction))
    return result


def train_fixed_epoch(name, data, fit_indices, check_indices, args, epochs, out):
    """Fit using only fit_indices; check_indices are never used during training."""
    seed_all(args.seed)
    config, lr = config_for(name)
    model = importlib.import_module(f"models.{name}").Model(config).to(args.device)
    x, y, ids = data["train"]
    scaler = StandardScaler().fit(y[fit_indices, None])
    scaled_y = scaler.transform(y[fit_indices, None]).astype(np.float32).ravel()
    loader = DataLoader(TensorDataset(torch.from_numpy(x[fit_indices]), torch.from_numpy(scaled_y)),
                        batch_size=args.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb, torch.ones((len(xb), 100), device=args.device))
            if isinstance(output, tuple):
                output = output[0]
            loss = torch.mean((output.reshape(-1) - yb) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
    prediction = predict(model, x[check_indices], scaler, args.device, args.batch_size)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": vars(config),
                "label_mean": float(scaler.mean_[0]), "label_scale": float(scaler.scale_[0]),
                "fit_cell_ids": np.asarray(ids)[fit_indices].tolist(), "epochs": epochs},
               out / "model.pt")
    return prediction


def main():
    args = arguments()
    unknown = sorted(set(args.models) - set(ALL_MODELS) - {"Dummy"})
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = args.out or HERE / "runs" / stamp
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update({"data_root": str(args.data_root), "out": str(out),
                   "T": 100, "selection_set": "12 cells held from official MATR train",
                   "official_status": "exploratory; previously viewed"})
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    cache_path = HERE / "cache" / "matr_t100.npz"
    data = prepare(args.data_root, cache_path)
    x, y, ids = data["train"]
    indices = list(range(len(ids)))
    random.Random(args.split_seed).shuffle(indices)
    check_indices = np.asarray(indices[:12])
    fit_indices = np.asarray(indices[12:])
    (out / "development_split.json").write_text(json.dumps({
        "fit": np.asarray(ids)[fit_indices].tolist(),
        "selection": np.asarray(ids)[check_indices].tolist()}, indent=2), encoding="utf-8")
    results = []
    dummy_pred = np.full(len(check_indices), np.mean(y[fit_indices]))
    results.append({"model": "Dummy", "status": "ok", "development": score(y[check_indices], dummy_pred)})
    for name in args.models:
        if name == "Dummy":
            continue
        print(f"START {name}", flush=True)
        try:
            result = train_one(name, data, fit_indices, check_indices, args, out)
        except Exception as exc:
            result = {"model": name, "status": "failed", "error": repr(exc),
                      "traceback": traceback.format_exc()[-4000:]}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(result)
        (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"DONE {name}: {result.get('development', result.get('error'))}", flush=True)
    print(json.dumps({"out": str(out), "summary": [
        {"model": item["model"], "status": item["status"],
         "development": item.get("development"), "official": item.get("official"),
         "error": item.get("error")}
        for item in results]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
