"""Bounded train-only, nested-epoch cross-fitting and automatic CP evaluation.

Five outer CELL folds within the frozen train97; inner validation chooses
epochs and is then included in outer refitting. Original dev/cal/test never
enter auxiliary training. OOF residuals train a risk model, not a CP quantile.
"""
import argparse
import importlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from curated_cp_evaluation import write_json
from run_isu_disagreement_cp import ROOT, read, digest, stamp
from run_single_gpu import UPSTREAM, seed_all


HERE = Path(__file__).resolve().parent
OUT = HERE / "isu_crossfit_risk_20261028"
SOURCE = ROOT / "runs/split_20261026/seed_2024"
MODELS = ("CPMLP", "CPLSTM")


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = read(plan_path)
        if plan["runner_sha256"] != digest(__file__):
            raise RuntimeError("Frozen code changed; use a new output directory")
        if any(digest(row["path"]) != row["sha256"] for row in plan["dependencies"]):
            raise RuntimeError("A frozen model/configuration dependency changed")
        return plan
    manifest = read(SOURCE / "dataset_manifest.json")
    train = manifest["split"]["phases"]["train"]
    assert len(train) == len(set(train)) == 97
    jobs = []
    for fold, (fit, held) in enumerate(KFold(5, shuffle=True, random_state=20261028).split(train)):
        inner_fit, inner_dev = train_test_split(fit, test_size=.2, random_state=20261028 + fold)
        for model in MODELS:
            names = lambda ix: [train[int(i)] for i in ix]
            jobs.append({"key": f"fold_{fold}_{model}", "fold": fold, "model": model,
                         "seed": 2024 + 101 * fold + (17 if model == "CPLSTM" else 0),
                         "outer_fit": names(fit), "outer_holdout": names(held),
                         "inner_fit": names(inner_fit), "inner_dev": names(inner_dev)})
    dependencies = [HERE / "run_single_gpu.py", HERE / "run_isu_disagreement_cp.py"]
    dependencies += [UPSTREAM / "models" / f"{model}.py" for model in MODELS]
    dependencies += [SOURCE / model / "config.json" for model in MODELS]
    parent = read(ROOT / "plan.json")
    dependencies += [ROOT / "analysis" / spec["key"] / "features.npz" for spec in parent["jobs"] if spec["split_seed"] == 20261026]
    plan = {"created_at": stamp(), "runner_sha256": digest(__file__),
            "evaluator_sha256": digest(HERE / "evaluate_isu_crossfit_risk.py"),
            "dependencies": [{"path": str(path), "sha256": digest(path)} for path in dependencies],
            "manifest_sha256": digest(SOURCE / "dataset_manifest.json"), "source": str(SOURCE),
            "hypotheses": ["H1: train-only OOF residual97 improve risk scaling over original dev20 residuals",
                           "H2: the learned scalar relative-risk distance improves same-score RLCP over lifetime-only distance"],
            "jobs": jobs, "epoch_cap": 30, "patience": 6, "least_epochs": 5,
            "outer_refit": "from scratch on all outer-fit cells for inner-selected epochs; no outer-holdout selection",
            "preprocessing": "original cached per-cell curves, per-stage label scaling on fitting cells only",
            "prefixes": "training1..100 per cell; OOF one observation at T100 per cell",
            "queue_seconds_cap": 5400, "job_seconds_cap": 1200,
            "risk_model": {"ridge_alpha": 10., "tail_quantile": .9, "tail_L1_alpha": .05,
                           "log_relative_floor": .02, "prediction_clip": [-4., 2.],
                           "features": "standardized log prediction, its square, and standardized log1p(pair gap/prediction)"},
            "cp": {"alpha": .1, "holdout_resplits": 200, "rlcp_bandwidths": [3., 6.],
                   "scores": ["relative", "oof_pair_tail"], "geometries": ["life", "learned_tail_risk"],
                   "group_cutpoints": "dev covariate pair-gap tertiles, no errors"},
            "caveats": ["historical tests exposed; exploratory, overlapping resplits not independent",
                        "OOF fits77 vs main97; conditional error transport not assumed proven",
                        "OOF pair uses one seed per architecture, so matched frozen pair is used at inference; six-model gap remains separate baseline",
                        "outer folds are by cell to match original cell-random target; no unseen-protocol claim",
                        "no supervised models from historical-cal/test risk audits are reused"]}
    write_json(plan_path, plan)
    (out / "dataset_manifest.json").write_bytes((SOURCE / "dataset_manifest.json").read_bytes())
    write_json(out / "state.json", {"state": "prepared", "updated_at": stamp(), "jobs": {}})
    return plan


def prepare_bank(out):
    path = out / "train_bank.npz"
    if path.exists():
        identity = read(out / "bank_identity.json")
        assert digest(path) == identity["bank_sha256"]
        assert digest(identity["labels"]) == identity["labels_sha256"]
        with np.load(path) as bank:
            return {k: bank[k] for k in bank.files}
    manifest = read(out / "dataset_manifest.json")
    names = manifest["split"]["phases"]["train"]
    cache_dir = ROOT / "shared_cache/split_20261026/processed_cache"
    cache = next(cache_dir.glob("train_*.pkl"))
    with cache.open("rb") as handle:
        payload = pickle.load(handle)
    fields = payload["fields"]
    assert payload["identity"]["files"] == fields["files"] == names
    counts = [fields["cell_prefix_counts"][n] for n in names]
    assert counts == [100] * 97
    indices = np.cumsum(counts) - 1
    curves = np.stack([fields["total_charge_discharge_curves"][i] for i in indices]).astype(np.float32)
    labels_path = Path(manifest["data_root"]) / "Life labels/ISU-ILCC_labels.json"
    labels = read(labels_path)
    y = np.asarray([labels[n] for n in names], dtype=float)
    np.testing.assert_allclose(y, fields["unique_labels"], rtol=0, atol=.002)
    assert curves.shape == (97, 100, 3, 300) and np.isfinite(curves).all()
    # Match the existing upstream prefix masks; never derive eligibility anew.
    for i in (0, 19, 99, 100, 9699):
        np.testing.assert_array_equal(fields["total_curve_attn_masks"][i], np.arange(100) < (i % 100 + 1))
        np.testing.assert_array_equal(curves[i // 100], np.asarray(fields["total_charge_discharge_curves"][i], dtype=np.float32))
    np.savez_compressed(path, names=np.asarray(names), curves=curves, labels=y)
    write_json(out / "bank_identity.json", {"cache": str(cache), "cache_sha256": digest(cache),
               "labels": str(labels_path), "labels_sha256": digest(labels_path), "bank_sha256": digest(path),
               "cells": 97, "curve_and_mask_parity": True})
    return {"names": np.asarray(names), "curves": curves, "labels": y}


class PrefixDataset(Dataset):
    def __init__(self, bank, names, mean, std):
        lookup = {n: i for i, n in enumerate(bank["names"])}
        self.indices = np.asarray([lookup[n] for n in names])
        self.curves = torch.from_numpy(bank["curves"])
        self.labels = torch.from_numpy(((bank["labels"] - mean) / std).astype(np.float32))

    def __len__(self):
        return len(self.indices) * 100

    def __getitem__(self, index):
        cell, prefix = self.indices[index // 100], index % 100 + 1
        return self.curves[cell], (torch.arange(100) < prefix).float(), self.labels[cell]


def collate(samples):
    x, mask, y = [torch.stack(v) for v in zip(*samples)]
    # Equivalent to upstream collate's exact zeroing of masked cycles.
    x.masked_fill_(~mask.bool()[:, :, None, None], 0.)
    return x, mask, y


def label_scale(bank, names):
    lookup = {n: i for i, n in enumerate(bank["names"])}
    values = bank["labels"][[lookup[n] for n in names]]
    return float(values.mean()), max(float(values.std()), 1e-6)


def predict(model, bank, names, mean, std, cfg):
    lookup = {n: i for i, n in enumerate(bank["names"])}
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(names), cfg.batch_size):
            batch_names = names[start:start + cfg.batch_size]
            x = torch.from_numpy(bank["curves"][[lookup[n] for n in batch_names]]).cuda()
            mask = torch.ones((len(x), 100), device="cuda")
            pred = model(x, mask).reshape(-1).cpu().numpy() * std + mean
            result.extend(pred.tolist())
    assert np.isfinite(result).all()
    return np.asarray(result)


def train_stage(bank, names, validation, cfg, spec, folder, plan, epochs=None, smoke=False):
    seed_all(spec["seed"])
    mean, std = label_scale(bank, names)
    model = importlib.import_module(f"models.{spec['model']}").Model(cfg).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    dataset = PrefixDataset(bank, names, mean, std)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    phase = "outer_refit" if epochs is not None else "inner_selection"
    best, best_epoch, stale, history = np.inf, None, 0, []
    lookup = {n: i for i, n in enumerate(bank["names"])}
    cap = epochs if epochs is not None else plan["epoch_cap"]
    for epoch in range(1, cap + 1):
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count, window = 0., 0, 0
        for batch_index, (x, mask, y) in enumerate(loader, 1):
            prediction = model(x.cuda(), mask.cuda()).reshape(-1)
            loss = torch.sum((prediction - y.cuda())**2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite auxiliary training loss")
            loss.backward()
            total += loss.item(); count += len(y); window += len(y)
            if batch_index % cfg.accumulation_steps == 0 or batch_index == len(loader) or smoke:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); window = 0
            if smoke:
                break
        row = {"epoch": epoch, "train_mse": total / count, "phase": phase}
        if validation:
            predictions = predict(model, bank, validation, mean, std, cfg)
            truth = bank["labels"][[lookup[n] for n in validation]]
            value = float(np.mean(np.abs(truth - predictions) / truth))
            if value < best:
                best, best_epoch, stale = value, epoch, 0
            elif epoch > plan["least_epochs"]:
                stale += 1
            row.update(inner_dev_mape=value, best_epoch=best_epoch, best_inner_mape=best)
        row["seconds"] = time.monotonic() - started
        history.append(row)
        write_json(folder / f"{phase}_history.json", history)
        write_json(folder / "progress.json", {"state": phase, "updated_at": stamp(), **row})
        print(f"{spec['key']} {phase} epoch={epoch}/{cap} mse={row['train_mse']:.4f} innerMAPE={row.get('inner_dev_mape', float('nan')):.3f} elapsed={row['seconds']:.1f}s", flush=True)
        if smoke or (validation and stale >= plan["patience"]):
            break
    return model, mean, std, best_epoch if validation else cap


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, value):
        self.console.write(value); self.console.flush(); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self):
        self.console.flush(); self.log.flush()


def worker(out, index, smoke=False):
    plan = prepare(out)
    spec = plan["jobs"][index]
    folder = out / ("smoke" if smoke else "folds") / spec["key"]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        print(f"SKIP COMPLETE {spec['key']}", flush=True)
        return
    with (folder / "console.log").open("a", encoding="utf-8") as log:
        previous = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        try:
            bank = prepare_bank(out)
            assert set(spec["inner_fit"]) | set(spec["inner_dev"]) == set(spec["outer_fit"])
            assert not set(spec["outer_holdout"]) & set(spec["outer_fit"])
            original = read(out / "dataset_manifest.json")["split"]["phases"]
            assert set(spec["outer_fit"] + spec["outer_holdout"]) == set(original["train"])
            cfg = SimpleNamespace(**read(SOURCE / spec["model"] / "config.json")["model_config"])
            os.chdir(UPSTREAM)
            torch.set_num_threads(4)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            model, _, _, selected_epoch = train_stage(bank, spec["inner_fit"], spec["inner_dev"], cfg, spec, folder, plan, smoke=smoke)
            del model
            torch.cuda.empty_cache()
            model, mean, std, _ = train_stage(bank, spec["outer_fit"], [], cfg, spec, folder, plan, epochs=selected_epoch, smoke=smoke)
            forecast = predict(model, bank, spec["outer_holdout"], mean, std, cfg)
            lookup = {n: i for i, n in enumerate(bank["names"])}
            truth = bank["labels"][[lookup[n] for n in spec["outer_holdout"]]]
            checkpoint = folder / "auxiliary_model.pt"
            torch.save({"state_dict": model.state_dict(), "label_mean": mean, "label_std": std,
                        "epoch": selected_epoch, "trained_on": spec["outer_fit"]}, checkpoint)
            pd.DataFrame({"cell_file": spec["outer_holdout"], "eol": truth, "prediction": forecast,
                          "fold": spec["fold"], "model": spec["model"]}).to_csv(folder / "oof_predictions.csv", index=False)
            write_json(folder / "complete.json", {"completed_at": stamp(), "job": spec, "selected_epoch": selected_epoch,
                       "checkpoint_sha256": digest(checkpoint), "smoke_only": smoke,
                       "label_scale_fit": spec["outer_fit"], "epoch_selection": "inner_dev only; outer_holdout labels not consulted"})
            print(f"{'SMOKE_PASS' if smoke else 'COMPLETE'} {spec['key']} selected_epoch={selected_epoch}", flush=True)
        finally:
            sys.stdout, sys.stderr = previous


def assemble(out, plan):
    tables = []
    for spec in plan["jobs"]:
        folder = out / "folds" / spec["key"]
        completion = read(folder / "complete.json")
        assert not completion["smoke_only"]
        assert completion["job"] == spec
        assert digest(folder / "auxiliary_model.pt") == completion["checkpoint_sha256"]
        rows = pd.read_csv(folder / "oof_predictions.csv")
        assert rows.cell_file.tolist() == spec["outer_holdout"]
        tables.append(rows)
    oof = pd.concat(tables, ignore_index=True)
    for model in MODELS:
        rows = oof[oof.model == model]
        assert len(rows) == rows.cell_file.nunique() == 97
    oof.to_csv(out / "oof_predictions.csv", index=False)


def queue(out, resume=False):
    plan = prepare(out)
    state = read(out / "state.json")
    if state["state"] in ("running", "evaluating"):
        raise RuntimeError("Queue already marked active; inspect PID before any restart")
    if state["state"] != "prepared" and not resume:
        raise RuntimeError("Use --resume to resume a stopped queue")
    prepare_bank(out)
    state.update(state="running", pid=os.getpid(), started_at=stamp())
    write_json(out / "state.json", state)
    started = time.monotonic()
    for index, spec in enumerate(plan["jobs"]):
        folder = out / "folds" / spec["key"]
        if (folder / "complete.json").exists():
            state["jobs"][spec["key"]] = {"state": "complete", "reused": True}
            continue
        remaining = plan["queue_seconds_cap"] - (time.monotonic() - started)
        if remaining <= 0:
            state.update(state="stopped_time_limit", updated_at=stamp())
            write_json(out / "state.json", state)
            return
        print(f"START {index+1}/{len(plan['jobs'])} {spec['key']} remaining_cap={remaining/60:.1f}min", flush=True)
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(index), "--out", str(out)],
                                 cwd=str(HERE.parents[1]), env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        state["jobs"][spec["key"]] = {"state": "running", "pid": child.pid, "started_at": stamp(), "log": str(folder / "console.log")}
        write_json(out / "state.json", state)
        try:
            code = child.wait(timeout=min(remaining, plan["job_seconds_cap"]))
        except subprocess.TimeoutExpired:
            child.kill(); child.wait()
            state["jobs"][spec["key"]].update(state="timed_out", updated_at=stamp())
            state.update(state="stopped_time_limit", updated_at=stamp())
            write_json(out / "state.json", state)
            return
        state["jobs"][spec["key"]].update(state="complete" if code == 0 else "failed", exit_code=code, finished_at=stamp())
        write_json(out / "state.json", state)
        if code != 0:
            state.update(state="failed", updated_at=stamp())
            write_json(out / "state.json", state)
            return
    assemble(out, plan)
    state.update(state="evaluating", updated_at=stamp())
    write_json(out / "state.json", state)
    code = subprocess.call([sys.executable, "-u", str(HERE / "evaluate_isu_crossfit_risk.py"), "--out", str(out)], cwd=str(HERE.parents[1]))
    state.update(state="complete" if code == 0 else "evaluation_failed", finished_at=stamp(), evaluation_exit_code=code)
    write_json(out / "state.json", state)
    print(f"QUEUE {state['state']} REPORT={out / 'BRIEF.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.worker is not None:
        worker(out, args.worker, args.smoke)
    elif args.queue:
        queue(out, args.resume)
    else:
        plan = prepare(out)
        prepare_bank(out)
        print(f"PREPARED {len(plan['jobs'])} jobs: {out / 'plan.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
