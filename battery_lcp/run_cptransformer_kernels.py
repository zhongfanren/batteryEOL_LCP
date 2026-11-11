"""BatteryLife CPTransformer adapter + learned-kernel localized CP.

Design: see ../CPTRANSFORMER_KERNEL_DESIGN.md.  The calibration and test labels
are never used to train predictor representations or kernel parameters.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
sys.path.insert(0, str(HERE))
import run_rul_landmarks as base  # noqa: E402

from models.CPTransformer import MLPBlock  # noqa: E402
from layers.Embed import PositionalEmbedding  # noqa: E402
from layers.SelfAttention_Family import FullAttention, AttentionLayer  # noqa: E402
from layers.Transformer_EncDec import Encoder, EncoderLayer  # noqa: E402
from lcp.pipeline import _LCPModuleSimple  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--landmarks", type=int, nargs="+", default=[50, 100, 150, 200])
    p.add_argument("--start-cycle", type=int, default=50)
    p.add_argument("--cycle-step", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--oof-folds", type=int, default=3)
    p.add_argument("--oof-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--intra-layers", type=int, default=1)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--model-val-cells", type=int, default=12)
    p.add_argument("--kernel-steps", type=int, default=400)
    p.add_argument("--min-ess", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()


class TruncatedCellDataset(Dataset):
    def __init__(self, cells: dict[str, dict], names: list[str], max_cycle: int):
        self.cells, self.names, self.max_cycle = cells, names, max_cycle

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray, int]:
        name = self.names[index]
        cell = self.cells[name]
        return name, cell["x_scaled"][: self.max_cycle], cell["life"]


def collate(batch: list[tuple[str, np.ndarray, int]]) -> dict[str, object]:
    names = [row[0] for row in batch]
    lengths = np.asarray([len(row[1]) for row in batch], dtype=np.int64)
    lives = np.asarray([row[2] for row in batch], dtype=np.float32)
    width = batch[0][1].shape[1]
    padded = np.zeros((len(batch), int(lengths.max()), width), dtype=np.float32)
    for i, (_, sequence, _) in enumerate(batch):
        padded[i, : len(sequence)] = sequence
    return {"names": names, "x": torch.from_numpy(padded),
            "lengths": torch.from_numpy(lengths), "lives": torch.from_numpy(lives)}


def loader(cells: dict[str, dict], names: list[str], args: argparse.Namespace,
           shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TruncatedCellDataset(cells, names, max(args.landmarks)),
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate,
    )


class CPTransformerRUL(nn.Module):
    """Causal per-cycle adapter built from BatteryLife CPTransformer blocks."""

    def __init__(self, input_dim: int, args: argparse.Namespace):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, args.d_model)
        self.intra = nn.ModuleList([
            MLPBlock(args.d_model, args.d_ff, args.d_model, args.dropout)
            for _ in range(args.intra_layers)
        ])
        self.position = PositionalEmbedding(args.d_model)
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(True, factor=5, attention_dropout=args.dropout,
                                  output_attention=False),
                    args.d_model, args.n_heads,
                ),
                args.d_model, args.d_ff, dropout=args.dropout, activation="gelu",
            ) for _ in range(args.transformer_layers)
        ], norm_layer=nn.LayerNorm(args.d_model))
        self.dropout = nn.Dropout(args.dropout)
        self.head = nn.Linear(args.d_model, 1)  # deliberately unbounded

    @staticmethod
    def causal_padding_mask(lengths: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
        causal = causal[None, None].expand(len(lengths), 1, length, length)
        positions = torch.arange(length, device=device)[None, :]
        key_padding = positions >= lengths.to(device)[:, None]
        return causal | key_padding[:, None, None, :]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.input_projection(x)
        for block in self.intra:
            z = block(z)
        z = z + self.position(z)
        mask = self.causal_padding_mask(lengths, z.shape[1], z.device)
        z, _ = self.encoder(z, attn_mask=mask)
        z = self.dropout(z)
        prediction = self.head(z).squeeze(-1)
        return prediction, z


def balanced_loss(pred: torch.Tensor, lives: torch.Tensor, lengths: torch.Tensor,
                  scale: float, args: argparse.Namespace) -> torch.Tensor:
    losses = []
    for i in range(len(lives)):
        indices = torch.arange(args.start_cycle - 1, int(lengths[i]), args.cycle_step,
                               device=pred.device)
        cycles = indices.float() + 1.0
        target = torch.clamp(lives[i] - cycles, min=0.0) / scale
        losses.append(torch.mean(torch.abs(pred[i, indices] - target)))
    return torch.stack(losses).mean()


@torch.no_grad()
def cell_mae(model: CPTransformerRUL, data_loader: DataLoader, scale: float,
             args: argparse.Namespace, device: torch.device) -> float:
    model.eval()
    values = []
    for batch in data_loader:
        prediction, _ = model(batch["x"].to(device), batch["lengths"])
        for i in range(len(batch["lives"])):
            indices = torch.arange(args.start_cycle - 1, int(batch["lengths"][i]),
                                   args.cycle_step, device=device)
            truth = batch["lives"][i].item() - (indices.cpu().numpy() + 1)
            estimate = np.maximum(prediction[i, indices].cpu().numpy() * scale, 0.0)
            values.append(float(np.mean(np.abs(truth - estimate))))
    return float(np.mean(values))


def train_predictor(cells: dict[str, dict], train_names: list[str], val_names: list[str],
                    args: argparse.Namespace, device: torch.device, epochs: int,
                    early_stop: bool, verbose: bool) -> tuple[CPTransformerRUL, dict]:
    model = CPTransformerRUL(len(base.feature_names()), args).to(device)
    train_loader = loader(cells, train_names, args, True, args.seed)
    val_loader = loader(cells, val_names, args, False, 0) if val_names else None
    scale = float(max(cells[name]["life"] for name in train_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_state, best_val, wait, history = copy.deepcopy(model.state_dict()), math.inf, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            prediction, _ = model(batch["x"].to(device), batch["lengths"])
            loss = balanced_loss(prediction, batch["lives"].to(device), batch["lengths"], scale, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch["lives"])
            seen += len(batch["lives"])
        train_l1 = total / seen
        val = cell_mae(model, val_loader, scale, args, device) if val_loader else math.nan
        history.append({"epoch": epoch, "train_l1_scaled": train_l1, "val_cell_mae": val})
        if val_loader and val < best_val - 1e-5:
            best_val, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        elif val_loader:
            wait += 1
        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"Epoch {epoch:3d} | train {train_l1:.5f} | val MAE {val:.2f}", flush=True)
        if early_stop and wait >= args.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}", flush=True)
            break
    if val_loader:
        model.load_state_dict(best_state)
    return model, {"label_scale": scale, "best_val_cell_mae": best_val, "history": history}


@torch.no_grad()
def predict(model: CPTransformerRUL, cells: dict[str, dict], names: list[str],
            scale: float, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    rows = []
    model.eval()
    for batch in loader(cells, names, args, False, 0):
        prediction, embedding = model(batch["x"].to(device), batch["lengths"])
        for i, name in enumerate(batch["names"]):
            life, length = int(batch["lives"][i]), int(batch["lengths"][i])
            for t in args.landmarks:
                if t > length or t >= life:
                    continue
                pred = max(float(prediction[i, t - 1].cpu()) * scale, 0.0)
                row = {"cell_id": name, "t": t, "true_rul": float(life - t),
                       "pred_rul": pred, "residual": float(life - t - pred)}
                for j, value in enumerate(embedding[i, t - 1].cpu().numpy()):
                    row[f"embedding_{j}"] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def generate_oof(cells: dict[str, dict], train_names: list[str], args: argparse.Namespace,
                 device: torch.device) -> tuple[pd.DataFrame, list[dict]]:
    batches = np.asarray([name.split("_")[1].split("c")[0] for name in train_names])
    folds = StratifiedKFold(args.oof_folds, shuffle=True, random_state=args.seed)
    frames, logs = [], []
    names = np.asarray(train_names)
    for fold, (fit_idx, hold_idx) in enumerate(folds.split(names, batches), start=1):
        print(f"OOF fold {fold}/{args.oof_folds}: train={len(fit_idx)}, holdout={len(hold_idx)}", flush=True)
        # Fixed epochs: holdout labels are never used for checkpoint selection.
        model, info = train_predictor(cells, names[fit_idx].tolist(), [], args, device,
                                      args.oof_epochs, False, False)
        frame = predict(model, cells, names[hold_idx].tolist(), info["label_scale"], args, device)
        frame["oof_fold"] = fold
        frames.append(frame)
        logs.append({"fold": fold, "train_cells": len(fit_idx), "holdout_cells": len(hold_idx),
                     "epochs": args.oof_epochs, "label_scale": info["label_scale"]})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = pd.concat(frames, ignore_index=True)
    return result, logs


STATE_COLUMNS = ["capacity_retention", "capacity_slope_10", "capacity_variability_10",
                 "internal_resistance", "temperature_10"]


def spectral_features(cells: dict[str, dict], frame: pd.DataFrame, window: int = 50,
                      bands: int = 5) -> tuple[np.ndarray, list[str]]:
    columns = [(19, "capacity"), (30, "resistance"), (21, "temperature")]
    output, names = [], []
    for _, signal_name in columns:
        names.extend([f"{signal_name}_current", f"{signal_name}_slope"])
        names.extend([f"{signal_name}_fft_band_{b}" for b in range(bands)])
    for row in frame.itertuples():
        features = []
        raw = cells[row.cell_id]["x"][: int(row.t)]
        for column, _ in columns:
            signal = raw[-min(window, len(raw)):, column].astype(float)
            finite = np.isfinite(signal)
            if finite.sum() == 0:
                signal = np.zeros_like(signal)
            elif not finite.all():
                indices = np.arange(len(signal))
                signal[~finite] = np.interp(indices[~finite], indices[finite], signal[finite])
            time = np.arange(len(signal), dtype=float)
            slope, intercept = np.polyfit(time, signal, 1) if len(signal) >= 2 else (0.0, signal[-1])
            residual = signal - (slope * time + intercept)
            power = np.abs(np.fft.rfft(residual))[1:] ** 2
            chunks = np.array_split(power, bands)
            energies = [math.log1p(float(chunk.sum())) if len(chunk) else 0.0 for chunk in chunks]
            features.extend([float(signal[-1]), float(slope), *energies])
        output.append(features)
    return np.asarray(output, dtype=np.float32), names


class QuantileProfile(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 16), nn.GELU(), nn.Dropout(0.05), nn.Linear(16, 3))
        self.register_buffer("quantiles", torch.tensor([0.5, 0.75, 0.9]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = F.softplus(self.net(x))
        return torch.cumsum(raw, dim=1)  # enforce ordered positive quantiles


def fit_quantile_profile(x: np.ndarray, score: np.ndarray, seed: int,
                         steps: int = 500) -> tuple[QuantileProfile, np.ndarray]:
    torch.manual_seed(seed)
    model = QuantileProfile(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)[:, None]
    for _ in range(steps):
        pred = model(xt)
        error = yt - pred
        q = model.quantiles[None, :]
        loss = torch.maximum(q * error, (q - 1.0) * error).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        fitted = model(xt).numpy()
    return model, fitted


@dataclass
class MetricFit:
    kind: str
    phi_train: np.ndarray
    transform: object
    effective_h: float
    parameters: dict
    train_loss: float
    train_ess: float


def learn_metric(x: np.ndarray, scores: np.ndarray, kind: str, steps: int,
                 min_ess: float, seed: int) -> tuple[np.ndarray, dict, float, float, float]:
    torch.manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float32)
    target = torch.tensor(np.log1p(scores), dtype=torch.float32)
    n, d = x.shape
    if kind == "lowrank":
        rank = min(3, d)
        parameter = nn.Parameter(torch.randn(rank, d) * 0.05)
        variables = [parameter]
    else:
        parameter = nn.Parameter(torch.zeros(d))
        variables = [parameter]
    h_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.9))))
    variables.append(h_raw)
    optimizer = torch.optim.Adam(variables, lr=0.03, weight_decay=1e-4)
    eye = torch.eye(n, dtype=torch.bool)
    final = {}
    for _ in range(steps):
        if kind == "lowrank":
            phi = xt @ parameter.T
            regularizer = 1e-3 * (parameter ** 2).mean()
        else:
            weights = torch.softmax(parameter, dim=0) * d
            phi = xt * torch.sqrt(weights)[None, :]
            regularizer = 1e-3 * (parameter ** 2).mean()
        d2_raw = ((phi[:, None, :] - phi[None, :, :]) ** 2).sum(dim=2)
        positive = d2_raw[~eye]
        scale = torch.median(positive).detach().clamp_min(1e-6)
        h = F.softplus(h_raw) + 0.1
        kernel = torch.exp(-d2_raw / (2.0 * scale * h ** 2)).masked_fill(eye, 0.0)
        normalizer = kernel.sum(dim=1).clamp_min(1e-8)
        estimate = kernel @ target / normalizer
        ess = normalizer ** 2 / (kernel.pow(2).sum(dim=1).clamp_min(1e-8))
        loss_fit = F.smooth_l1_loss(estimate, target)
        loss_ess = torch.relu(torch.tensor(min_ess) - ess).pow(2).mean() * 0.01
        loss = loss_fit + loss_ess + regularizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final = {"loss": float(loss_fit.detach()), "ess": float(ess.mean().detach()),
                 "scale": float(scale), "h": float(h.detach())}
    if kind == "lowrank":
        matrix = parameter.detach().numpy()
        phi_np = x @ matrix.T
        params = {"projection": matrix.tolist(), "rank": matrix.shape[0]}
    else:
        weights = (torch.softmax(parameter.detach(), dim=0) * d).numpy()
        phi_np = x * np.sqrt(weights)[None, :]
        params = {"feature_weights": weights.tolist()}
    effective_h = final["h"] * math.sqrt(final["scale"])
    params.update({"normalized_h": final["h"], "distance_scale": final["scale"],
                   "effective_h": effective_h})
    return phi_np, params, effective_h, final["loss"], final["ess"]


def apply_metric(x: np.ndarray, params: dict, kind: str) -> np.ndarray:
    if kind == "lowrank":
        return x @ np.asarray(params["projection"], dtype=float).T
    return x * np.sqrt(np.asarray(params["feature_weights"], dtype=float))[None, :]


def learned_lcp(cal_scores: np.ndarray, cal_phi: np.ndarray, test_pred: np.ndarray,
                test_phi: np.ndarray, learned_h: float, split_threshold: float,
                alpha: float, min_ess: float) -> dict[str, np.ndarray]:
    order = np.argsort(cal_scores)
    scores, cal_ordered = cal_scores[order], cal_phi[order]
    d_cal = base.pairwise_euclidean(cal_ordered, cal_ordered)
    d_test = base.pairwise_euclidean(test_phi, cal_ordered)
    deltas, esses, hs, expanded, fallback = [], [], [], [], []
    for row in d_test:
        h = max(float(learned_h), 1e-6)
        grew = False
        for _ in range(12):
            weights = np.exp(-(row ** 2) / (2.0 * h ** 2))
            ess = float(weights.sum() ** 2 / max(float(np.sum(weights ** 2)), 1e-12))
            if ess >= min_ess:
                break
            h *= 1.5
            grew = True
        half_cal, half_test = d_cal ** 2 / 2.0, row.reshape(1, -1) ** 2 / 2.0
        module = _LCPModuleSimple(half_cal, scores, h=h ** 2, alpha=alpha)
        module._prepare()
        candidate = float(module._construct(half_test, half_test.T)[0])
        failed = not np.isfinite(candidate)
        if failed:
            candidate = split_threshold
        deltas.append(candidate); esses.append(ess); hs.append(h); expanded.append(grew); fallback.append(failed)
    delta = np.asarray(deltas)
    return {"lower": np.maximum(0.0, test_pred - delta), "upper": test_pred + delta,
            "delta": delta, "ess": np.asarray(esses), "bandwidth": np.asarray(hs),
            "expanded": np.asarray(expanded), "fallback": np.asarray(fallback)}


def prepare_kernel_inputs(kind: str, t: int, oof: pd.DataFrame, cal: pd.DataFrame,
                          test: pd.DataFrame, cells: dict[str, dict], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    state_scaler = RobustScaler().fit(oof[STATE_COLUMNS])
    state_oof = state_scaler.transform(oof[STATE_COLUMNS])
    state_cal = state_scaler.transform(cal[STATE_COLUMNS])
    state_test = state_scaler.transform(test[STATE_COLUMNS])
    metadata = {"base": "state", "input_dimension": state_oof.shape[1]}
    if kind == "diagonal_state":
        return state_oof, state_cal, state_test, metadata
    if kind == "spectral_band":
        raw_oof, names = spectral_features(cells, oof)
        raw_cal, _ = spectral_features(cells, cal)
        raw_test, _ = spectral_features(cells, test)
        scaler = StandardScaler().fit(raw_oof)
        metadata = {"base": "spectral", "features": names, "input_dimension": len(names)}
        return scaler.transform(raw_oof), scaler.transform(raw_cal), scaler.transform(raw_test), metadata
    embedding_columns = [c for c in oof.columns if c.startswith("embedding_")]
    embedding_scaler = StandardScaler().fit(oof[embedding_columns])
    emb_oof = embedding_scaler.transform(oof[embedding_columns])
    count = min(8, emb_oof.shape[0], emb_oof.shape[1])
    pca = PCA(count, whiten=True, random_state=seed).fit(emb_oof)
    joint_oof = np.column_stack([state_oof, pca.transform(emb_oof)])
    joint_cal = np.column_stack([state_cal, pca.transform(embedding_scaler.transform(cal[embedding_columns]))])
    joint_test = np.column_stack([state_test, pca.transform(embedding_scaler.transform(test[embedding_columns]))])
    metadata = {"base": "state_plus_embedding_pca", "pca_components": count,
                "pca_explained": float(pca.explained_variance_ratio_.sum()),
                "input_dimension": joint_oof.shape[1]}
    if kind == "lowrank_joint":
        return joint_oof, joint_cal, joint_test, metadata
    score = np.abs(oof["residual"].to_numpy())
    profile, profile_oof = fit_quantile_profile(joint_oof, score, seed + t)
    with torch.no_grad():
        profile_cal = profile(torch.tensor(joint_cal, dtype=torch.float32)).numpy()
        profile_test = profile(torch.tensor(joint_test, dtype=torch.float32)).numpy()
    profile_scaler = StandardScaler().fit(profile_oof)
    metadata.update({"base": "learned_quantile_profile", "quantiles": [0.5, 0.75, 0.9]})
    return profile_scaler.transform(profile_oof), profile_scaler.transform(profile_cal), profile_scaler.transform(profile_test), metadata


def main() -> None:
    args = parse_args()
    base.seed_everything(args.seed)
    device = base.resolve_device(args.device)
    run_dir = args.output_dir / f"cptransformer_learned_kernels_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Loading MATR ...", flush=True)
    cells = base.load_cells(args.data_root, args.start_cycle)
    splits = base.official_splits(cells, args)
    medians, input_scaler = base.fit_input_scaler(cells, splits["train"])
    cells = base.transform_cells(cells, medians, input_scaler)
    (run_dir / "split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Splits: " + str({k: len(v) for k, v in splits.items()}), flush=True)

    oof, oof_logs = generate_oof(cells, splits["train"], args, device)
    oof = base.add_state_features(oof, cells)
    oof.to_csv(run_dir / "oof_train_predictions.csv", index=False)
    (run_dir / "oof_training.json").write_text(json.dumps(oof_logs, indent=2), encoding="utf-8")

    print("Training final CPTransformerRUL ...", flush=True)
    model, training = train_predictor(cells, splits["train"], splits["model_val"], args,
                                      device, args.epochs, True, True)
    pd.DataFrame(training["history"]).to_csv(run_dir / "training_history.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "input_medians": medians,
                "input_scaler_mean": input_scaler.mean_, "input_scaler_scale": input_scaler.scale_,
                "label_scale": training["label_scale"], "config": config}, run_dir / "model.pt")

    frames, all_frames = {}, []
    for split, names in splits.items():
        frame = predict(model, cells, names, training["label_scale"], args, device)
        frame.insert(0, "split", split)
        frame = base.add_state_features(frame, cells)
        frames[split] = frame
        all_frames.append(frame)
    pd.concat(all_frames, ignore_index=True).to_csv(run_dir / "predictions_all.csv", index=False)

    kernel_names = ["diagonal_state", "lowrank_joint", "quantile_profile", "spectral_band"]
    point_results, split_results = {}, {}
    kernel_results = {name: {} for name in kernel_names}
    kernel_parameters, diagnostics = {}, []
    for t in args.landmarks:
        subsets = {name: frame[frame.t == t].reset_index(drop=True) for name, frame in frames.items()}
        oof_t = oof[oof.t == t].reset_index(drop=True)
        cal, test = subsets["calibration"], subsets["test"]
        y_cal, pred_cal = cal.true_rul.to_numpy(), cal.pred_rul.to_numpy()
        y_test, pred_test = test.true_rul.to_numpy(), test.pred_rul.to_numpy()
        cal_scores, oof_scores = np.abs(y_cal - pred_cal), np.abs(oof_t.residual.to_numpy())
        cp_low, cp_high, threshold = base.split_cp(cal_scores, pred_test, args.alpha)
        point_results[str(t)] = base.regression_metrics(y_test, pred_test)
        split_results[str(t)] = {**base.interval_metrics(y_test, cp_low, cp_high), "threshold": threshold}
        output = test[["cell_id", "t", "true_rul", "pred_rul", "residual"]].copy()
        output["split_cp_lower"], output["split_cp_upper"] = cp_low, cp_high
        kernel_parameters[str(t)] = {}
        for kind in kernel_names:
            x_oof, x_cal, x_test, metadata = prepare_kernel_inputs(kind, t, oof_t, cal, test, cells, args.seed)
            metric_kind = "lowrank" if kind == "lowrank_joint" else "diagonal"
            _, parameters, learned_h, train_loss, train_ess = learn_metric(
                x_oof, oof_scores, metric_kind, args.kernel_steps, args.min_ess, args.seed + t)
            cal_phi, test_phi = apply_metric(x_cal, parameters, metric_kind), apply_metric(x_test, parameters, metric_kind)
            interval = learned_lcp(cal_scores, cal_phi, pred_test, test_phi, learned_h,
                                   threshold, args.alpha, args.min_ess)
            metrics = base.interval_metrics(y_test, interval["lower"], interval["upper"])
            metrics.update({"ess_min": float(interval["ess"].min()),
                            "ess_median": float(np.median(interval["ess"])),
                            "ess_mean": float(interval["ess"].mean()),
                            "bandwidth_expansion_fraction": float(interval["expanded"].mean()),
                            "fallback_fraction": float(interval["fallback"].mean()),
                            "mean_width_change_vs_split": float(metrics["mean_width"] - split_results[str(t)]["mean_width"]),
                            "kernel_train_loss": train_loss, "kernel_train_ess": train_ess,
                            "representation": metadata})
            kernel_results[kind][str(t)] = metrics
            kernel_parameters[str(t)][kind] = {**parameters, "representation": metadata,
                                               "train_loss": train_loss, "train_ess": train_ess}
            output[f"{kind}_lower"] = interval["lower"]
            output[f"{kind}_upper"] = interval["upper"]
            output[f"{kind}_ess"] = interval["ess"]
            output[f"{kind}_fallback"] = interval["fallback"]
            for i, cell_id in enumerate(output.cell_id):
                diagnostics.append({"cell_id": cell_id, "t": t, "kernel": kind,
                                    "ess": interval["ess"][i], "bandwidth": interval["bandwidth"][i],
                                    "expanded": bool(interval["expanded"][i]),
                                    "fallback": bool(interval["fallback"][i])})
        output.to_csv(run_dir / f"landmark_{t:03d}.csv", index=False)
        widths = ", ".join(f"{k}={kernel_results[k][str(t)]['coverage']:.3f}/{kernel_results[k][str(t)]['mean_width']:.1f}" for k in kernel_names)
        print(f"t={t}: RMSE={point_results[str(t)]['rmse']:.2f}; Split={split_results[str(t)]['coverage']:.3f}/{split_results[str(t)]['mean_width']:.1f}; {widths}", flush=True)

    pd.DataFrame(diagnostics).to_csv(run_dir / "kernel_diagnostics.csv", index=False)
    (run_dir / "kernel_parameters.json").write_text(json.dumps(kernel_parameters, indent=2), encoding="utf-8")
    results = {"split_counts": {k: len(v) for k, v in splits.items()},
               "training": {k: v for k, v in training.items() if k != "history"},
               "point": point_results, "split_cp": split_results, "learned_kernels": kernel_results,
               "artifacts": str(run_dir.resolve())}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
