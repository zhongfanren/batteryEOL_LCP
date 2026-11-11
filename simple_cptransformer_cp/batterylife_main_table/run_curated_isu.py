"""Train two predictors on the fixed ISU-ILCC candidate, then evaluate CP on CPU.

--queue runs serially with persistent logs/state; --evaluate-only reuses frozen
predictions.  --smoke checks the actual loader and CUDA forward/backward path.
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from curated_cp_evaluation import evaluate_frozen, write_json
from run_mixed_liion_pilot import checkpoint_scaler
from run_single_gpu import (CACHE_FIELDS, HERE, MODEL_NAMES, UPSTREAM, evaluate,
                            official_config, seed_all, upstream_loader)


CANDIDATE = HERE / "protocol_candidates" / "isu_ilcc_t100_candidate.json"
DEFAULT_OUT = HERE / "curated_isu_pilot"
MODELS = ("CPMLP", "CPLSTM")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "dataset_manifest.json"
    plan_path = out / "experiment_plan.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if digest(manifest_path) != digest(CANDIDATE):
            raise RuntimeError("Candidate differs from this frozen experiment; use a new output directory")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["manifest_sha256"] != digest(manifest_path):
            raise RuntimeError("Frozen experiment manifest changed")
        return manifest, plan
    manifest = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    phases = manifest["split"]["phases"]
    all_names = [name for phase in phases.values() for name in phase]
    if len(all_names) != len(set(all_names)) or len(all_names) != 239:
        raise RuntimeError("Invalid candidate cell partition")
    if any(not (Path(manifest["data_root"]) / "ISU_ILCC" / name).is_file() for name in all_names):
        raise FileNotFoundError("Candidate has missing cell files")
    manifest_path.write_bytes(CANDIDATE.read_bytes())
    plan = {"created_at": timestamp(), "models": list(MODELS), "train_seed": 2021,
            "epochs_cap": 100, "patience": 8, "least_epochs": 5,
            "manifest_sha256": digest(manifest_path),
            "question": "Can feature-local calibration allocate width to harder cells while retaining useful coverage, on a chemically consistent multi-protocol population?",
            "training": "Original BatteryLife 1..100 prefixes; train cells only; epoch selected on dev T=100 MAPE",
            "evaluation": "One T=100 observation per dev/calibration/test cell; predictor frozen before calibration",
            "methods": ["split_cp", "normalized_cp", "base_lcp_protocol", "rlcp_prediction", "rlcp_protocol"],
            "local_features": ["prediction", "charge rate", "discharge rate"],
            "local_bandwidth": 1.5, "local_random_seed": 20261027,
            "difficulty": "dev-fitted kernel-smoothed absolute error; no calibration/test residual used to fit difficulty",
            "test_exploratory": True, "selection": "No selection by test scores; report both models and all preset methods",
            "caveat": "Same underlying data viewed in earlier experiments; no new-source or per-protocol coverage claim"}
    write_json(plan_path, plan)
    print(f"FROZEN_PLAN={plan_path}", flush=True)
    return manifest, plan


def load_dataset(module, cfg, phases, phase, out, scaler=None, class_scaler=None, smoke=False):
    flag = "train" if phase == "train" else "val"
    names = phases[phase][:1] if smoke else phases[phase]
    setattr(module.split_recorder, f"MIX_large_{flag}_files", names)
    t100 = phase != "train"
    identity = {"files": names, "phase": phase, "t100": t100,
                "root": cfg.root_path, "format": 1,
                "loader_sha256": digest(UPSTREAM / "data_provider" / "data_loader.py"),
                "adapter_sha256": digest(HERE / "run_single_gpu.py"),
                "life_classes_sha256": digest(UPSTREAM / "data_provider" / "life_classes.json"),
                "label_file_sha256": digest(Path(cfg.root_path) / "Life labels" / "ISU-ILCC_labels.json"),
                "length": cfg.charge_discharge_length, "threshold": cfg.early_cycle_threshold,
                "mean": float(scaler.mean_[0]) if scaler is not None else None,
                "var": float(scaler.var_[0]) if scaler is not None else None}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache = out / "processed_cache" / f"{phase}_{key}.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            payload = pickle.load(handle)
        if payload["identity"] != identity:
            raise RuntimeError("Preprocessing cache identity changed")
        dataset = module.Dataset_original.__new__(module.Dataset_original)
        dataset.__dict__.update(payload["fields"])
        print(f"CACHE_HIT phase={phase}", flush=True)
    else:
        print(f"PREPROCESS phase={phase} cells={len(names)}", flush=True)
        kwargs = {"eval_cycle_min": 100, "eval_cycle_max": 100} if t100 else {}
        dataset = module.Dataset_original(cfg, flag=flag, label_scaler=scaler,
                                          life_class_scaler=class_scaler, **kwargs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"identity": identity, "fields": {k: getattr(dataset, k) for k in CACHE_FIELDS}},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache)
    multiplier = 1 if t100 else 100
    if dataset.files != names or len(dataset) != multiplier * len(names):
        raise RuntimeError(f"Unexpected dataset membership/count: {phase}")
    if any(dataset.cell_prefix_counts.get(name) != multiplier for name in names):
        raise RuntimeError(f"Unexpected prefix count: {phase}")
    return dataset


def worker(args):
    out = args.out.resolve()
    manifest, plan = prepare(out)
    model_out = out / args.model
    model_out.mkdir(exist_ok=True)
    if args.evaluate_only:
        checkpoint = torch.load(model_out / "best_model.pt", map_location="cpu", weights_only=False)
        evaluate_frozen(model_out, manifest, checkpoint)
        return
    if (model_out / "config.json").exists() and not args.smoke:
        raise FileExistsError(f"Training already started: {model_out}; use --evaluate-only for saved predictions")
    os.chdir(UPSTREAM)
    torch.set_num_threads(4)
    seed_all(plan["train_seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this queue expects the local GPU")
    device = "cuda"
    cfg = official_config(Path(manifest["data_root"]), args.model)
    cfg.patience, cfg.least_epochs = plan["patience"], plan["least_epochs"]
    module = upstream_loader()

    def progress(stage, **fields):
        write_json(model_out / "progress.json", {"state": stage, "updated_at": timestamp(),
                   "pid": os.getpid(), "model": args.model, **fields})
        print(f"STAGE={stage}", flush=True)

    progress("loading_train")
    train = load_dataset(module, cfg, manifest["split"]["phases"], "train", out, smoke=args.smoke)
    model = importlib.import_module(f"models.{args.model}").Model(cfg).to(device)
    collate = module.my_collate_fn_baseline
    if args.smoke:
        batch = collate([train[0], train[-1]])
        model.train()
        pred = model(batch[0].float().to(device), batch[1].float().to(device))
        loss = torch.mean((pred.reshape(-1) - batch[2].float().to(device).reshape(-1))**2)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Smoke produced nonfinite gradient")
        dev_smoke = load_dataset(module, cfg, manifest["split"]["phases"], "dev", out,
                                 train.label_scaler, train.life_class_scaler, smoke=True)
        model.eval()
        dev_batch = collate([dev_smoke[0]])
        with torch.inference_mode():
            dev_pred = model(dev_batch[0].float().to(device), dev_batch[1].float().to(device))
        if not torch.isfinite(dev_pred).all() or dev_batch[1].sum().item() != 100:
            raise RuntimeError("T=100 development loader smoke failed")
        print(f"SMOKE_PASS model={args.model} device={device} shape={tuple(pred.shape)} loss={loss.item():.6f}", flush=True)
        return
    progress("loading_dev")
    dev = load_dataset(module, cfg, manifest["split"]["phases"], "dev", out,
                       train.label_scaler, train.life_class_scaler)
    config = {"model": args.model, "device": device, "plan": plan, "model_config": vars(cfg),
              "loader_sha256": digest(UPSTREAM / "data_provider" / "data_loader.py"),
              "model_sha256": digest(UPSTREAM / "models" / f"{args.model}.py"),
              "runner_sha256": digest(__file__),
              "cp_evaluator_sha256": digest(HERE / "curated_cp_evaluation.py"),
              "train_prefixes": len(train), "dev_t100_cells": len(dev)}
    write_json(model_out / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                        drop_last=False, collate_fn=collate)
    best_mape, stale, history, best_epoch = float("inf"), 0, [], None
    progress("training")
    for epoch in range(1, plan["epochs_cap"] + 1):
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, n_samples = 0.0, 0
        # Scale by the actual number of examples in each accumulation window,
        # including the final partial window.
        window_samples = 0
        for batch_index, (x, mask, labels, *_rest) in enumerate(loader, 1):
            prediction = model(x.float().to(device), mask.float().to(device))
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            prediction = prediction.reshape(-1)
            target = labels.float().to(device).reshape(-1)
            loss_sum = torch.sum((prediction - target)**2)
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("Training loss became nonfinite")
            loss_sum.backward()
            n = len(target)
            window_samples += n
            total_loss += loss_sum.item()
            n_samples += n
            if batch_index % cfg.accumulation_steps == 0 or batch_index == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_samples)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window_samples = 0
            if batch_index % 100 == 0:
                write_json(model_out / "progress.json", {"state": "training", "updated_at": timestamp(),
                           "pid": os.getpid(), "model": args.model, "epoch": epoch,
                           "batch": batch_index, "batches": len(loader), "best_epoch": best_epoch})
                print(f"{args.model} epoch={epoch:03d} batch={batch_index}/{len(loader)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        dev_metrics, _ = evaluate(model, dev, collate, device, cfg.batch_size)
        value = dev_metrics["mape_fraction"]
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite development MAPE")
        if value < best_mape:
            best_mape, best_epoch, stale = value, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "label_mean": float(train.label_scaler.mean_[0]),
                        "label_median": float(np.median(train.unique_labels)),
                        "label_std": float(np.sqrt(train.label_scaler.var_[0]))}, model_out / "best_model.pt")
        elif epoch > cfg.least_epochs:
            stale += 1
        row = {"epoch": epoch, "train_scaled_mse": total_loss / n_samples,
               "seconds": time.monotonic() - started, "best_epoch": best_epoch,
               "best_dev_mape": best_mape, **dev_metrics}
        history.append(row)
        write_json(model_out / "history.json", history)
        progress("training", **row)
        print(f"{args.model} epoch={epoch:03d} train_mse={row['train_scaled_mse']:.5f} dev_T100_MAPE={value:.2%} best={best_mape:.2%} seconds={row['seconds']:.1f} stale={stale}/{cfg.patience}", flush=True)
        if stale >= cfg.patience:
            break
    checkpoint = torch.load(model_out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    write_json(model_out / "frozen_predictor.json", {"frozen_at": timestamp(), "model": args.model,
               "best_epoch": best_epoch, "checkpoint_sha256": digest(model_out / "best_model.pt"),
               "manifest_sha256": plan["manifest_sha256"], "trained_on": "train; epoch selected using dev only",
               "scaler_fit": "train only"})
    _, dev_rows = evaluate(model, dev, collate, device, cfg.batch_size)
    dev_rows["prefix_cycles"] = 100
    dev_rows.to_csv(model_out / "dev_predictions.csv", index=False)
    del train, dev, loader
    gc.collect()
    for phase in ("calibration", "test"):
        progress(f"predicting_{phase}")
        dataset = load_dataset(module, cfg, manifest["split"]["phases"], phase, out,
                               checkpoint_scaler(checkpoint))
        _, rows = evaluate(model, dataset, collate, device, cfg.batch_size)
        rows["prefix_cycles"] = 100
        rows.to_csv(model_out / f"{phase}_predictions.csv", index=False)
        del dataset
        gc.collect()
    progress("evaluating_intervals")
    evaluate_frozen(model_out, manifest, checkpoint)
    progress("complete", best_epoch=best_epoch)


def queue(args):
    out = args.out.resolve()
    _, plan = prepare(out)
    state_path = out / "queue_state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if previous.get("state") == "running":
        raise RuntimeError("Queue is already marked running; inspect its PID before restarting")
    state = {"state": "running", "started_at": timestamp(), "pid": os.getpid(),
             "models": previous.get("models", {}), "model_order": plan["models"]}
    write_json(state_path, state)
    for model in plan["models"]:
        model_out = out / model
        if (model_out / "results.json").exists():
            state["models"][model] = {"state": "complete", "output": str(model_out)}
            continue
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--model", model, "--out", str(out)]
        log_path = out / f"{model}.log"
        print(f"\nSTART {model}\nLOG={log_path}\n", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=str(HERE.parents[1]), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                     env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
            state["models"][model] = {"state": "running", "pid": child.pid,
                                      "started_at": timestamp(), "log": str(log_path), "output": str(model_out)}
            write_json(state_path, state)
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = child.wait()
        state["models"][model].update(state="complete" if code == 0 else "failed",
                                       exit_code=code, finished_at=timestamp())
        write_json(state_path, state)
    state["state"] = "complete" if all(x["state"] == "complete" for x in state["models"].values()) else "finished_with_failures"
    state["finished_at"] = timestamp()
    write_json(state_path, state)
    lines = ["# ISU-ILCC frozen-split model comparison", "",
             "Exploratory. Both models are preset; do not select a model or localizer by these test scores.", "",
             "| Model | Method | Test MAPE | Coverage | Mean width | Infinite |", "|---|---|---:|---:|---:|---:|"]
    for model in plan["models"]:
        path = out / model / "results.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        for name, metric in result["methods"].items():
            lines.append(f"| {model} | {name} | {result['point']['test']['mape_fraction']:.2%} | {metric['coverage']:.2%} | {metric['mean_width_cycles']} | {metric['infinite_intervals']} |")
    (out / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nQUEUE {state['state']}\nREPORT={out / 'COMPARISON.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", choices=MODEL_NAMES, default="CPMLP")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.prepare:
            prepare(args.out.resolve())
        elif args.queue:
            queue(args)
        else:
            worker(args)
    except Exception:
        if not args.queue:
            target = args.out.resolve() / args.model
            target.mkdir(parents=True, exist_ok=True)
            write_json(target / "progress.json", {"state": "failed", "updated_at": timestamp(),
                       "pid": os.getpid(), "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
