"""Bounded ISU stability batch: 3 cell splits x 3 initializations x 2 models.

Reuses the two completed seed-2021 predictors on the original split.  All
methods/hyperparameters are fixed in a plan before any new model is fitted.
"""

import argparse
import importlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

import run_curated_isu as training
from curated_cp_evaluation import correlation, point_metrics, weighted_radius, write_json
from run_mixed_liion_pilot import checkpoint_scaler
from run_single_gpu import HERE, UPSTREAM, seed_all, upstream_loader


BASE = HERE / "curated_isu_pilot"
OUT = HERE / "isu_stability_20261028"
SPLIT_SEEDS = (20261026, 20261028, 20261029)
TRAIN_SEEDS = (2021, 42, 2024)
MODELS = ("CPMLP", "CPLSTM")
METHODS = ("split_cp", "relative_cp", "prediction_scale_cp", "protocol_scale_cp",
           "curve_scale_cp", "normalized_rlcp_prediction_h1.5", "normalized_rlcp_curve_h1.5",
           "normalized_rlcp_prediction_h3", "normalized_rlcp_curve_h3")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.json"
    if path.exists():
        plan = read(path)
        if plan["runner_sha256"] != training.digest(__file__):
            raise RuntimeError("Batch code changed; choose a new output directory")
        return plan
    candidate = read(BASE / "dataset_manifest.json")
    names = sorted(row["cell_file"] for row in candidate["cells"])
    for split_seed in SPLIT_SEEDS:
        shuffled = names.copy()
        random.Random(split_seed).shuffle(shuffled)
        phases, start = {}, 0
        for phase, size in candidate["split"]["phase_sizes"].items():
            phases[phase] = shuffled[start:start + size]
            start += size
        if split_seed == SPLIT_SEEDS[0]:
            assert phases == candidate["split"]["phases"]
        new_candidate = {**candidate, "status": "exploratory repeated-split batch",
                         "split": {**candidate["split"], "seed": split_seed, "phases": phases}}
        write_json(out / f"candidate_{split_seed}.json", new_candidate)
    jobs = []
    for split_seed in SPLIT_SEEDS:
        for train_seed in TRAIN_SEEDS:
            for model in MODELS:
                reused = split_seed == SPLIT_SEEDS[0] and train_seed == 2021
                source = BASE if reused else out / "runs" / f"split_{split_seed}" / f"seed_{train_seed}"
                key = f"s{split_seed}_r{train_seed}_{model}"
                jobs.append({"key": key, "split_seed": split_seed, "train_seed": train_seed,
                             "model": model, "reuse_predictor": reused, "source": str(source),
                             "candidate": str(out / f"candidate_{split_seed}.json")})
    plan = {"created_at": now(), "runner_sha256": training.digest(__file__),
            "training_runner_sha256": training.digest(HERE / "run_curated_isu.py"),
            "base_manifest_sha256": training.digest(BASE / "dataset_manifest.json"),
            "hypotheses": ["H1: apparent scale-CP improvements persist across predictor initialization and cell partitions",
                           "H2: early-curve representations add information beyond predicted-lifetime scale, including local calibration of normalized residuals"],
            "jobs": jobs, "methods": list(METHODS), "alpha": .1,
            "pca_components": 3, "pca_fit": "train embeddings only; whitened; no cal/test fitting",
            "scale_fit": "Gaussian regression on 20 development absolute residuals; h=1.5, positive floor",
            "rlcp": "Hore-Barber Gaussian random centers; h=1.5 and 3 fixed sensitivity settings; report all, no test-based choice",
            "local_scores": "absolute residual divided by fixed prediction-only development scale",
            "holdout_resplits": 200, "holdout_resplit_seed": 20261028,
            "resplit_scope": "shuffle only the 122 cal+test cells; predictor, dev, scales, PCA and risk cutoffs remain frozen",
            "primary_readout": "paired width and interval-score differences; coverage and infinite-interval counts; normalized residual risk signal; all jobs reported",
            "decision_rule": "Treat width/score improvements as useful only alongside coverage. Report signs across all completed split/seed pairs; do not rank/select using test. Consistent small effects remain exploratory, not a significance claim.",
            "max_queue_seconds": 10800, "max_job_seconds": 1800,
            "exploratory": "Historical test exposure, overlapping repetitions, and small samples preclude a pristine confirmatory claim. Repetitions are not independent datasets."}
    write_json(path, plan)
    write_json(out / "state.json", {"state": "prepared", "jobs": {}, "updated_at": now()})
    return plan


def install_shared_cache(out, spec):
    original = training.load_dataset
    cache_root = out / "shared_cache" / f"split_{spec['split_seed']}"
    def shared(*args, **kwargs):
        positional = list(args)
        positional[4] = cache_root
        # Dataset augmentation consumes RNG even though this task uses only
        # unaugmented curves. Make cache hits/misses neutral to model init RNG.
        states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        try:
            return original(*positional, **kwargs)
        finally:
            random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
            if states[3] is not None:
                torch.cuda.set_rng_state_all(states[3])
    training.load_dataset = shared


def export_features(source, model_name, cache_root, output):
    """Export one T=100 embedding per cell from the frozen predictor."""
    manifest = read(source / "dataset_manifest.json")
    model_dir = source / model_name
    frozen = read(model_dir / "frozen_predictor.json")
    assert training.digest(model_dir / "best_model.pt") == frozen["checkpoint_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    feature_file = output / "features.npz"
    if feature_file.exists():
        identity = read(output / "feature_identity.json")
        if identity["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
            raise RuntimeError("Feature checkpoint identity mismatch")
        return manifest, {k: v for k, v in np.load(feature_file).items()}
    os.chdir(UPSTREAM)
    torch.set_num_threads(4)
    seed_all(20261028)
    cfg = SimpleNamespace(**read(model_dir / "config.json")["model_config"])
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cuda", weights_only=False)
    model = importlib.import_module(f"models.{model_name}").Model(cfg).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"])
    module = upstream_loader()
    original_loader = training.load_dataset
    # Caller does not install the cache wrapper for reused fits.
    payload = {}
    for phase in ("train", "dev", "calibration", "test"):
        if phase == "train":
            # Prefer an existing completed training cache. This also avoids
            # rereading all raw cycling files for reused original predictors.
            candidates = list((cache_root / "processed_cache").glob("train_*.pkl"))
            dataset = None
            for candidate in candidates:
                with candidate.open("rb") as handle:
                    cached = pickle.load(handle)
                if cached["identity"]["files"] == manifest["split"]["phases"]["train"]:
                    dataset = module.Dataset_original.__new__(module.Dataset_original)
                    dataset.__dict__.update(cached["fields"])
                    break
            if dataset is None:
                dataset = original_loader(module, cfg, manifest["split"]["phases"], phase, cache_root)
            indices = np.cumsum([dataset.cell_prefix_counts[n] for n in dataset.files]) - 1
            selected = Subset(dataset, indices.tolist())
        else:
            dataset = original_loader(module, cfg, manifest["split"]["phases"], phase,
                                      cache_root, checkpoint_scaler(checkpoint))
            selected = dataset
        loader = DataLoader(selected, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                            collate_fn=module.my_collate_fn_baseline)
        predictions, embeddings = [], []
        with torch.inference_mode():
            for x, mask, *_ in loader:
                pred, embedding = model(x.float().cuda(), mask.float().cuda(), return_embedding=True)
                predictions.extend((pred.reshape(-1).cpu().numpy() * checkpoint["label_std"] + checkpoint["label_mean"]).tolist())
                embeddings.extend(embedding.cpu().numpy())
        names = manifest["split"]["phases"][phase]
        assert len(names) == len(predictions) == len(embeddings)
        payload[f"{phase}_names"] = np.asarray(names)
        payload[f"{phase}_embedding"] = np.asarray(embeddings)
        payload[f"{phase}_prediction"] = np.maximum(0, predictions)
        if phase != "train":
            previous = pd.read_csv(model_dir / f"{phase}_predictions.csv").set_index("cell_file").loc[names]
            np.testing.assert_allclose(predictions, previous.prediction, atol=.03, rtol=1e-5)
            # Preserve exact frozen values; inference is only exporting the
            # representation, not silently replacing previously saved forecasts.
            payload[f"{phase}_prediction"] = np.maximum(0, previous.prediction.to_numpy())
            payload[f"{phase}_eol"] = previous.eol.to_numpy()
        print(f"FEATURES phase={phase} n={len(names)} dim={len(embeddings[0])}", flush=True)
        del dataset, selected, loader
    np.savez_compressed(feature_file, **payload)
    write_json(output / "feature_identity.json", {"checkpoint_sha256": frozen["checkpoint_sha256"],
               "source_manifest_sha256": training.digest(source / "dataset_manifest.json"),
               "source": str(source), "model": model_name})
    return manifest, payload


def kernel_scale(dev_z, dev_errors, query, leave_out=False, min_floor=1.):
    logw = -np.sum((query[:, None, :] - dev_z[None, :, :])**2, axis=2) / 4.5
    if leave_out:
        np.fill_diagonal(logw, -np.inf)
    weights = np.exp(logw - logw.max(axis=1, keepdims=True))
    return np.maximum(max(min_floor, .1 * np.median(dev_errors)), weights @ dev_errors / weights.sum(axis=1))


def randomized_radius(cal_z, test_z, scores, bandwidth, seed):
    rng = np.random.default_rng(seed)
    radii, ess = [], []
    order = np.argsort(scores)
    for z in test_z:
        center = z + rng.normal(0, bandwidth, len(z))
        logs = -np.r_[np.sum((cal_z - center)**2, axis=1), np.sum((z - center)**2)] / (2 * bandwidth**2)
        weights = np.exp(logs - logs.max())
        threshold = .9 * weights.sum()
        i = np.searchsorted(np.cumsum(weights[:-1][order]), threshold, side="left")
        radii.append(scores[order[i]] if i < len(scores) else float("inf"))
        ess.append(weights[:-1].sum()**2 / np.sum(weights[:-1]**2))
    return np.asarray(radii), np.asarray(ess)


def metrics(y, prediction, radii, groups):
    lower, upper = np.maximum(0, prediction - radii), prediction + radii
    width = upper - lower
    covered = (lower <= y) & (y <= upper)
    interval_score = width + 20 * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
    result = {"n": len(y), "coverage": covered.mean(), "mean_width": width.mean(),
              "interval_score": interval_score.mean(), "infinite": int((~np.isfinite(radii)).sum())}
    for group in (0, 1, 2):
        keep = groups == group
        result[f"group_{group}_n"] = int(keep.sum())
        result[f"group_{group}_coverage"] = covered[keep].mean() if keep.any() else np.nan
    return result


def analyze_features(manifest, data, source, spec, output, repetitions):
    model_result = read(source / spec["model"] / "results.json")
    scales = model_result["scales"]
    pca = PCA(n_components=3, whiten=True, svd_solver="full")
    pca.fit(data["train_embedding"])
    metadata = {r["cell_file"]: r["metadata"] for r in manifest["cells"]}
    def build_z(phase):
        pred = data[f"{phase}_prediction"]
        rates = np.asarray([[metadata[n][k][0]["rate_in_C"] for k in ("charge_protocol", "discharge_protocol")]
                            for n in data[f"{phase}_names"]])
        level = ((pred - scales["prediction_mean"]) / scales["prediction_std"])[:, None]
        protocol = np.column_stack((level, (rates - scales["rate_mean"]) / scales["rate_std"]))
        curve = np.column_stack((level, pca.transform(data[f"{phase}_embedding"])))
        return level, protocol, curve
    dev_z = build_z("dev")
    cal_z, test_z = build_z("calibration"), build_z("test")
    pool_z = tuple(np.concatenate((a, b)) for a, b in zip(cal_z, test_z))
    dev_error = np.abs(data["dev_eol"] - data["dev_prediction"])
    pool_y = np.r_[data["calibration_eol"], data["test_eol"]]
    pool_pred = np.r_[data["calibration_prediction"], data["test_prediction"]]
    pool_error = np.abs(pool_y - pool_pred)
    pool_names = np.r_[data["calibration_names"], data["test_names"]]
    scale_pred = kernel_scale(dev_z[0], dev_error, pool_z[0])
    scale_protocol = kernel_scale(dev_z[1], dev_error, pool_z[1])
    scale_curve = kernel_scale(dev_z[2], dev_error, pool_z[2])
    # Risk groups fixed without cal/test labels. Estimate normalized-error risk
    # using leave-one-out development scales and development curve neighbours.
    dev_scale_loo = kernel_scale(dev_z[0], dev_error, dev_z[0], leave_out=True)
    dev_normalized_error = dev_error / dev_scale_loo
    risk = kernel_scale(dev_z[2], dev_normalized_error, pool_z[2], min_floor=1e-6)
    dev_risk_loo = kernel_scale(dev_z[2], dev_normalized_error, dev_z[2], leave_out=True, min_floor=1e-6)
    cutpoints = np.quantile(dev_risk_loo, [1/3, 2/3])
    groups = np.searchsorted(cutpoints, risk, side="right")
    description = {"model": spec["model"], "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
                   "normalized_risk_spearman_original_test": correlation(risk[61:], (pool_error / scale_pred)[61:]),
                   "absolute_risk_spearman_original_test": correlation(scale_curve[61:], pool_error[61:]),
                   "risk_cutpoints_from_dev": cutpoints.tolist(),
                   "test_exploratory": True, "method_order": list(METHODS)}
    write_json(output / "feature_diagnostics.json", description)
    rng = np.random.default_rng(20261028)
    all_metrics, original_rows = [], []
    for repetition in range(repetitions + 1):
        permutation = np.arange(122) if repetition == 0 else rng.permutation(122)
        ci, ti = permutation[:61], permutation[61:]
        scores = pool_error[ci]
        radii = {"split_cp": np.full(61, weighted_radius(scores, np.ones(61), 1))}
        for name, scale in (("relative_cp", np.maximum(1, pool_pred)),
                            ("prediction_scale_cp", scale_pred), ("protocol_scale_cp", scale_protocol),
                            ("curve_scale_cp", scale_curve)):
            radii[name] = weighted_radius(scores / scale[ci], np.ones(61), 1) * scale[ti]
        local_ess = {}
        for h in (1.5, 3.):
            for suffix, coordinates in (("prediction", pool_z[0]), ("curve", pool_z[2])):
                name = f"normalized_rlcp_{suffix}_h{h:g}"
                q, ess = randomized_radius(coordinates[ci], coordinates[ti], scores / scale_pred[ci], h, 20261028 + repetition)
                radii[name] = q * scale_pred[ti]
                local_ess[name] = ess
        for method, radius in radii.items():
            values = metrics(pool_y[ti], pool_pred[ti], radius, groups[ti])
            values.update({"method": method, "repetition": repetition,
                           "mean_ess": local_ess[method].mean() if method in local_ess else 61})
            all_metrics.append(values)
            if repetition == 0:
                for j, index in enumerate(ti):
                    original_rows.append({"cell_file": pool_names[index], "method": method,
                                          "prediction": pool_pred[index], "eol": pool_y[index],
                                          "radius": radius[j], "risk_group": groups[index],
                                          "normalized_risk": risk[index]})
        if repetition % 25 == 0:
            print(f"ANALYSIS repetition={repetition}/{repetitions}", flush=True)
            write_json(output / "progress.json", {"state": "holdout_resplits", "repetition": repetition,
                       "total": repetitions, "updated_at": now(), "pid": os.getpid()})
    pd.DataFrame(all_metrics).to_csv(output / "resplit_metrics.csv", index=False)
    pd.DataFrame(original_rows).to_csv(output / "original_test_intervals.csv", index=False)
    write_json(output / "complete.json", {"completed_at": now(), "job": spec, "resplits": repetitions,
               "runner_sha256": training.digest(__file__)})


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log
    def write(self, value):
        self.original.write(value); self.original.flush()
        self.log.write(value); self.log.flush()
        return len(value)
    def flush(self):
        self.original.flush(); self.log.flush()


def job(out, index, smoke=False):
    plan = prepare(out)
    spec = plan["jobs"][index]
    output = out / "analysis" / spec["key"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists() and not smoke:
        print(f"SKIP_COMPLETE {spec['key']}", flush=True)
        return
    with (output / "console.log").open("a", encoding="utf-8") as log:
        prior_stdout, prior_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        source = Path(spec["source"])
        try:
            if not spec["reuse_predictor"]:
                training.CANDIDATE = Path(spec["candidate"])
                install_shared_cache(out, spec)
                training.prepare(source)
                train_plan = read(source / "experiment_plan.json")
                train_plan["train_seed"] = spec["train_seed"]
                train_plan["batch_runner_sha256"] = plan["runner_sha256"]
                train_plan["cache_rng_neutral"] = True
                write_json(source / "experiment_plan.json", train_plan)
                if not (source / spec["model"] / "results.json").exists():
                    training.worker(SimpleNamespace(out=source, model=spec["model"], smoke=False, evaluate_only=False))
            cache_root = BASE if spec["reuse_predictor"] else out / "shared_cache" / f"split_{spec['split_seed']}"
            manifest, data = export_features(source, spec["model"], cache_root, output)
            analyze_features(manifest, data, source, spec, output, 2 if smoke else plan["holdout_resplits"])
        except Exception:
            write_json(output / "failure.json", {"state": "failed", "at": now(), "error": traceback.format_exc()})
            raise
        finally:
            sys.stdout, sys.stderr = prior_stdout, prior_stderr


def summarize(out, plan):
    rows = []
    for spec in plan["jobs"]:
        path = out / "analysis" / spec["key"] / "resplit_metrics.csv"
        if not (path.parent / "complete.json").exists():
            continue
        table = pd.read_csv(path)
        for (method, original), group in table.groupby(["method", table.repetition.eq(0)]):
            widths = group.mean_width.to_numpy()
            scores = group.interval_score.to_numpy()
            reference = table[(table.method == "prediction_scale_cp") & table.repetition.eq(0).eq(original)].set_index("repetition").loc[group.repetition]
            rows.append({"job": spec["key"], "model": spec["model"], "split_seed": spec["split_seed"],
                         "train_seed": spec["train_seed"], "method": method, "original_test": bool(original),
                         "evaluations": len(group), "mean_coverage": group.coverage.mean(),
                         "mean_width": np.mean(widths), "mean_interval_score": np.mean(scores),
                         "fraction_repeats_with_infinite": np.mean(group.infinite > 0),
                         "mean_infinite_cells": group.infinite.mean(),
                         "coverage_delta_pp_vs_prediction_scale": 100 * np.mean(group.coverage.to_numpy() - reference.coverage.to_numpy()),
                         "mean_paired_width_delta": np.mean(widths - reference.mean_width.to_numpy()),
                         "mean_paired_interval_score_delta": np.mean(scores - reference.interval_score.to_numpy())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    lines = ["# ISU-ILCC stability batch", "", "Exploratory; overlapping cell resplits are not independent replications.",
             "No model, feature or bandwidth is selected by these results. Infinite widths are retained.", "",
             "| Model | Method | Completed fits | Mean resplit coverage | Infinite-repeat fraction | Mean width across all fits/resplits |",
             "|---|---|---:|---:|---:|---:|"]
    if len(summary):
        repeated = summary[~summary.original_test]
        for (model, method), group in repeated.groupby(["model", "method"]):
            lines.append(f"| {model} | {method} | {len(group)} | {group.mean_coverage.mean():.2%} | {group.fraction_repeats_with_infinite.mean():.2%} | {group.mean_width.mean()} |")
    lines += ["", "Read summary.csv for each split/initialization. Original test and resplit results are separate.",
              "Each analysis/job directory contains raw repetition metrics, original cell radii, feature diagnostics and feature checkpoint identity.",
              "Interpret width and interval score with coverage. Larger or infinite intervals alone do not establish useful localization."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def queue(out):
    plan = prepare(out)
    existing = read(out / "state.json")
    if existing.get("state") == "running":
        raise RuntimeError("Queue already marked running; inspect its PID before restarting")
    state = {"state": "running", "started_at": now(), "pid": os.getpid(), "jobs": existing.get("jobs", {})}
    started = time.monotonic()
    write_json(out / "state.json", state)
    for index, spec in enumerate(plan["jobs"]):
        completion = out / "analysis" / spec["key"] / "complete.json"
        if completion.exists():
            state["jobs"][spec["key"]] = {"state": "complete", "reused_completed_job": True}
            continue
        remaining = plan["max_queue_seconds"] - (time.monotonic() - started)
        if remaining < 120:
            break
        print(f"\nJOB {index+1}/{len(plan['jobs'])}: {spec['key']} remaining_budget={remaining/60:.1f}min", flush=True)
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--out", str(out), "--job", str(index)],
                                 cwd=str(HERE.parents[1]), env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        state["jobs"][spec["key"]] = {"state": "running", "pid": child.pid, "started_at": now()}
        state["current_job"] = spec["key"]
        write_json(out / "state.json", state)
        try:
            code = child.wait(timeout=min(remaining - 30, plan["max_job_seconds"]))
            status = "complete" if code == 0 and completion.exists() else "failed"
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait(timeout=30)
            code, status = -1, "time_budget_exceeded"
        state["jobs"][spec["key"]].update(state=status, exit_code=code, finished_at=now())
        write_json(out / "state.json", state)
        summarize(out, plan)
    completed = sum(v["state"] == "complete" for v in state["jobs"].values())
    state.update(state="complete" if completed == len(plan["jobs"]) else "finished_partial",
                 finished_at=now(), completed_jobs=completed, planned_jobs=len(plan["jobs"]))
    write_json(out / "state.json", state)
    summarize(out, plan)
    print(f"\nBATCH {state['state']} {completed}/{len(plan['jobs'])}; REPORT={out / 'SUMMARY.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--job", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        prepare(out)
    elif args.queue:
        queue(out)
    else:
        job(out, args.job, args.smoke)


if __name__ == "__main__":
    main()
