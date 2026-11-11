"""Leakage-free BatteryLife MATR landmark RUL experiment.

One Mamba model is trained on cell sequences with sparse per-cycle RUL labels.
Split CP, observable-state LCP, and PCA-embedding LCP are calibrated separately
at fixed landmarks.  Every split and conformal operation uses cells as the
independent units.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
OFFICIAL_MAMBA = WORKSPACE / "baselines" / "MambaLithium"
PYTHON_LCP = WORKSPACE / "python_lcp"
BATTERYLIFE_REPO = WORKSPACE / "third_party" / "BatteryLife"
sys.path.insert(0, str(OFFICIAL_MAMBA))
sys.path.insert(0, str(PYTHON_LCP))
sys.path.insert(0, str(BATTERYLIFE_REPO))

from mamba import Mamba, MambaConfig  # noqa: E402
from lcp.pipeline import _LCPModuleSimple  # noqa: E402


SIGNAL_KEYS = (
    "voltage_in_V",
    "current_in_A",
    "charge_capacity_in_Ah",
    "discharge_capacity_in_Ah",
    "temperature_in_C",
    "internal_resistance_in_ohm",
)
STATS = ("mean", "std", "min", "max", "last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--landmarks", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--start-cycle", type=int, default=50)
    parser.add_argument("--cycle-step", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--model-val-cells", type=int, default=12)
    parser.add_argument("--lcp-k", type=int, default=15)
    parser.add_argument("--min-ess", type=float, default=10.0)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def safe_numeric(values: object) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def summary(values: object) -> list[float]:
    values = safe_numeric(values)
    if not len(values):
        return [np.nan] * len(STATS)
    return [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
        float(values[-1]),
    ]


def extract_sequence(record: dict, usable_cycles: int) -> np.ndarray:
    cycles = record.get("cycle_data", [])
    n = min(len(cycles), usable_cycles)
    rows = []
    for index, cycle in enumerate(cycles[:n], start=1):
        row = [float(index)]
        for key in SIGNAL_KEYS:
            row.extend(summary(cycle.get(key)))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def feature_names() -> list[str]:
    names = ["cycle_number"]
    for key in SIGNAL_KEYS:
        names.extend(f"{key}_{stat}" for stat in STATS)
    return names


def load_cells(data_root: Path, start_cycle: int) -> dict[str, dict]:
    matr_dir = data_root / "MATR"
    label_path = data_root / "Life labels" / "MATR_labels.json"
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    cells = {}
    for filename, raw_life in labels.items():
        life = int(raw_life)
        path = matr_dir / filename
        if not path.exists() or life <= start_cycle:
            continue
        with path.open("rb") as handle:
            record = pickle.load(handle)
        sequence = extract_sequence(record, life)
        # RUL labels are defined only where an observation exists and before EOL.
        usable = min(len(sequence), life)
        if usable <= start_cycle:
            continue
        cells[filename] = {"x": sequence[:usable], "life": life}
    if not cells:
        raise RuntimeError("No usable MATR cells found")
    return cells


def official_splits(cells: dict[str, dict], args: argparse.Namespace) -> dict[str, list[str]]:
    from data_provider.data_split_recorder import split_recorder

    available = set(cells)
    official_train = [x for x in split_recorder.MATR_train_files if x in available]
    calibration = [x for x in split_recorder.MATR_val_files if x in available]
    test = [x for x in split_recorder.MATR_test_files if x in available]
    max_landmark = max(args.landmarks)
    val_candidates = [name for name in official_train if len(cells[name]["x"]) >= max_landmark]
    batches = [name.split("_")[1].split("c")[0] for name in val_candidates]
    train_candidates, model_val = train_test_split(
        val_candidates,
        test_size=args.model_val_cells,
        random_state=args.seed,
        shuffle=True,
        stratify=batches,
    )
    # Cells too short for every landmark remain valid training sequences.
    train = train_candidates + [name for name in official_train if name not in val_candidates]
    splits = {
        "train": sorted(train),
        "model_val": sorted(model_val),
        "calibration": sorted(calibration),
        "test": sorted(test),
    }
    flat = [x for values in splits.values() for x in values]
    if len(flat) != len(set(flat)):
        raise RuntimeError("Cell leakage between splits")
    for t in args.landmarks:
        for split in ("calibration", "test"):
            missing = [name for name in splits[split] if len(cells[name]["x"]) < t]
            if missing:
                raise RuntimeError(f"{split} has {len(missing)} cells unavailable at t={t}")
    return splits


def fit_input_scaler(cells: dict[str, dict], names: list[str]) -> tuple[np.ndarray, StandardScaler]:
    data = np.concatenate([cells[name]["x"] for name in names], axis=0)
    medians = np.nanmedian(data, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    clean = np.where(np.isnan(data), medians, data)
    return medians, StandardScaler().fit(clean)


def transform_cells(
    cells: dict[str, dict], medians: np.ndarray, scaler: StandardScaler
) -> dict[str, dict]:
    transformed = {}
    for name, cell in cells.items():
        clean = np.where(np.isnan(cell["x"]), medians, cell["x"])
        transformed[name] = {
            **cell,
            "x_scaled": scaler.transform(clean).astype(np.float32),
        }
    return transformed


class CellDataset(Dataset):
    def __init__(self, cells: dict[str, dict], names: list[str]):
        self.cells = cells
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray, int]:
        name = self.names[index]
        cell = self.cells[name]
        return name, cell["x_scaled"], cell["life"]


def collate_cells(batch: list[tuple[str, np.ndarray, int]]) -> dict[str, object]:
    names = [x[0] for x in batch]
    lengths = np.asarray([len(x[1]) for x in batch], dtype=np.int64)
    lives = np.asarray([x[2] for x in batch], dtype=np.float32)
    width = batch[0][1].shape[1]
    padded = np.zeros((len(batch), int(lengths.max()), width), dtype=np.float32)
    for i, (_, sequence, _) in enumerate(batch):
        padded[i, : len(sequence)] = sequence
    return {
        "names": names,
        "x": torch.from_numpy(padded),
        "lengths": torch.from_numpy(lengths),
        "lives": torch.from_numpy(lives),
    }


def make_loader(
    cells: dict[str, dict], names: list[str], batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        CellDataset(cells, names),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_cells,
    )


class MambaRUL(nn.Module):
    """MambaLithium-style Linear -> Mamba -> Linear -> Sigmoid model."""

    def __init__(self, input_dim: int, hidden: int, layers: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden)
        self.encoder = Mamba(MambaConfig(d_model=hidden, n_layers=layers))
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(self.input_projection(x))
        prediction = self.head(embedding).squeeze(-1)
        return prediction, embedding


def cell_balanced_loss(
    prediction: torch.Tensor,
    lives: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    start: int,
    step: int,
) -> torch.Tensor:
    losses = []
    for i in range(len(lives)):
        length = int(lengths[i])
        indices = torch.arange(start - 1, length, step, device=prediction.device)
        cycles = indices.to(torch.float32) + 1.0
        target = torch.clamp(lives[i] - cycles, min=0.0) / scale
        losses.append(torch.mean(torch.abs(prediction[i, indices] - target)))
    return torch.stack(losses).mean()


@torch.no_grad()
def validation_mae(
    model: MambaRUL,
    loader: DataLoader,
    device: torch.device,
    scale: float,
    start: int,
    step: int,
) -> float:
    model.eval()
    per_cell = []
    for batch in loader:
        x = batch["x"].to(device)
        pred, _ = model(x)
        lengths = batch["lengths"]
        lives = batch["lives"]
        for i in range(len(lives)):
            length = int(lengths[i])
            idx = torch.arange(start - 1, length, step, device=device)
            truth = lives[i].item() - (idx.cpu().numpy() + 1)
            estimate = pred[i, idx].cpu().numpy() * scale
            per_cell.append(float(np.mean(np.abs(truth - estimate))))
    return float(np.mean(per_cell))


def train_model(
    model: MambaRUL,
    cells: dict[str, dict],
    splits: dict[str, list[str]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MambaRUL, dict]:
    train_loader = make_loader(cells, splits["train"], args.batch_size, True, args.seed)
    val_loader = make_loader(cells, splits["model_val"], args.batch_size, False, 0)
    # Fixed train-only scale; unlike the original script, no test life is used.
    label_scale = float(max(cells[name]["life"] for name in splits["train"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            lives = batch["lives"].to(device)
            prediction, _ = model(x)
            loss = cell_balanced_loss(
                prediction, lives, batch["lengths"], label_scale,
                args.start_cycle, args.cycle_step,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(lives)
            seen += len(lives)
        val_mae = validation_mae(
            model, val_loader, device, label_scale, args.start_cycle, args.cycle_step
        )
        train_loss = running / seen
        history.append({"epoch": epoch, "train_l1_scaled": train_loss, "val_cell_mae": val_mae})
        if val_mae < best_val - 1e-5:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | train L1 {train_loss:.5f} | val cell MAE {val_mae:.2f}", flush=True)
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    return model, {"label_scale": label_scale, "best_val_cell_mae": best_val, "history": history}


@torch.no_grad()
def predict_landmarks(
    model: MambaRUL,
    cells: dict[str, dict],
    names: list[str],
    landmarks: list[int],
    label_scale: float,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    loader = make_loader(cells, names, batch_size, False, 0)
    rows = []
    model.eval()
    for batch in loader:
        prediction, embedding = model(batch["x"].to(device))
        for i, name in enumerate(batch["names"]):
            life = int(batch["lives"][i])
            length = int(batch["lengths"][i])
            for t in landmarks:
                if t > length or t >= life:
                    continue
                pred = float(prediction[i, t - 1].cpu()) * label_scale
                truth = float(life - t)
                row = {"cell_id": name, "t": t, "true_rul": truth, "pred_rul": pred,
                       "residual": truth - pred}
                for j, value in enumerate(embedding[i, t - 1].cpu().numpy()):
                    row[f"embedding_{j}"] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def observable_state(raw_x: np.ndarray, t: int) -> np.ndarray:
    """Five predeclared, observation-only state features at landmark t."""
    prefix = raw_x[:t]
    # 31 columns: cycle + six signals x five stats. Discharge capacity max is 19.
    capacity = prefix[:, 19].astype(float)
    initial = float(np.nanmedian(capacity[: min(5, len(capacity))]))
    current = float(capacity[-1])
    retention = current / initial if np.isfinite(initial) and abs(initial) > 1e-12 else np.nan
    recent = capacity[-min(10, len(capacity)):]
    time = np.arange(len(recent), dtype=float)
    finite = np.isfinite(recent)
    if finite.sum() >= 2:
        slope, intercept = np.polyfit(time[finite], recent[finite], 1)
        variability = float(np.std(recent[finite] - (slope * time[finite] + intercept)))
    else:
        slope, variability = np.nan, np.nan
    resistance = float(prefix[-1, 30])
    temperature = float(np.nanmean(prefix[-min(10, len(prefix)):, 21]))
    return np.asarray([retention, slope, variability, resistance, temperature], dtype=float)


def add_state_features(frame: pd.DataFrame, cells: dict[str, dict]) -> pd.DataFrame:
    state = np.vstack([observable_state(cells[row.cell_id]["x"], int(row.t)) for row in frame.itertuples()])
    for j, name in enumerate(("capacity_retention", "capacity_slope_10", "capacity_variability_10",
                              "internal_resistance", "temperature_10")):
        frame[name] = state[:, j]
    return frame


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    denominator = np.maximum(np.abs(y), 1.0)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.mean(np.abs(y - pred) / denominator)),
        "r2": float(r2_score(y, pred)),
        "mean_error": float(np.mean(pred - y)),
        "abs_residual_q90": float(np.quantile(np.abs(y - pred), 0.9)),
        "max_abs_residual": float(np.max(np.abs(y - pred))),
    }


def split_cp(cal_scores: np.ndarray, test_pred: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    rank = int(math.ceil((len(cal_scores) + 1) * (1 - alpha)))
    threshold = math.inf if rank > len(cal_scores) else float(np.sort(cal_scores)[rank - 1])
    return np.maximum(0.0, test_pred - threshold), test_pred + threshold, threshold


def interval_metrics(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(low) & np.isfinite(high)
    widths = (high - low)[finite]
    return {
        "coverage": float(np.mean((y >= low) & (y <= high))),
        "covered": int(np.sum((y >= low) & (y <= high))),
        "total": int(len(y)),
        "mean_width": float(widths.mean()) if len(widths) else math.inf,
        "median_width": float(np.median(widths)) if len(widths) else math.inf,
        "width_q90": float(np.quantile(widths, 0.9)) if len(widths) else math.inf,
        "infinite_fraction": float(np.mean(~finite)),
    }


def pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2), 0.0))


def ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy()


def residual_similarity_correlation(z: np.ndarray, residual: np.ndarray) -> float:
    distance = pairwise_euclidean(z, z)
    residual_gap = np.abs(np.abs(residual)[:, None] - np.abs(residual)[None, :])
    upper = np.triu_indices(len(z), 1)
    if len(upper[0]) < 2:
        return math.nan
    return float(np.corrcoef(ranks(distance[upper]), ranks(residual_gap[upper]))[0, 1])


@dataclass
class LCPResult:
    lower: np.ndarray
    upper: np.ndarray
    delta: np.ndarray
    ess: np.ndarray
    bandwidth: np.ndarray
    fallback: np.ndarray


def adaptive_lcp(
    cal_scores: np.ndarray,
    cal_z: np.ndarray,
    test_pred: np.ndarray,
    test_z: np.ndarray,
    split_threshold: float,
    alpha: float,
    k: int,
    min_ess: float,
) -> LCPResult:
    order = np.argsort(cal_scores)
    scores = cal_scores[order]
    cal_ordered = cal_z[order]
    d_cal = pairwise_euclidean(cal_ordered, cal_ordered)
    d_test = pairwise_euclidean(test_z, cal_ordered)
    delta, ess_values, bandwidths, fallback = [], [], [], []
    k_index = min(max(k, 1), len(cal_scores)) - 1
    for row in d_test:
        positive = np.sort(row[row > 0])
        h = float(np.sort(row)[k_index])
        if not np.isfinite(h) or h <= 0:
            h = float(np.median(positive)) if len(positive) else 1.0
        for _ in range(12):
            weights = np.exp(-(row ** 2) / (2.0 * h ** 2))
            ess = float(weights.sum() ** 2 / max(float(np.sum(weights ** 2)), 1e-12))
            if ess >= min_ess:
                break
            h *= 1.5
        # The LCP implementation expects distance H and exp(-H/module.h).
        half_sq_cal = d_cal ** 2 / 2.0
        half_sq_test = row.reshape(1, -1) ** 2 / 2.0
        module = _LCPModuleSimple(half_sq_cal, scores, h=h ** 2, alpha=alpha)
        module._prepare()
        candidate = float(module._construct(half_sq_test, half_sq_test.T)[0])
        use_fallback = not np.isfinite(candidate)
        if use_fallback:
            candidate = split_threshold
        delta.append(candidate)
        ess_values.append(ess)
        bandwidths.append(h)
        fallback.append(use_fallback)
    delta = np.asarray(delta)
    return LCPResult(
        lower=np.maximum(0.0, test_pred - delta),
        upper=test_pred + delta,
        delta=delta,
        ess=np.asarray(ess_values),
        bandwidth=np.asarray(bandwidths),
        fallback=np.asarray(fallback, dtype=bool),
    )


def prepare_representations(
    train: pd.DataFrame,
    model_val: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    kind: str,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if kind == "state":
        columns = ["capacity_retention", "capacity_slope_10", "capacity_variability_10",
                   "internal_resistance", "temperature_10"]
        train_z = train[columns].to_numpy(float)
        median = np.nanmedian(train_z, axis=0)
        keep = np.isfinite(median) & (np.mean(np.isfinite(train_z), axis=0) >= 0.5)
        median = median[keep]
        def clean(frame: pd.DataFrame) -> np.ndarray:
            z = frame[columns].to_numpy(float)[:, keep]
            return np.where(np.isfinite(z), z, median)
        scaler = RobustScaler().fit(clean(train))
        transform = lambda frame: scaler.transform(clean(frame))
        metadata = {"columns": list(np.asarray(columns)[keep]), "dimension": int(keep.sum())}
    else:
        columns = [x for x in train.columns if x.startswith("embedding_")]
        scaler = StandardScaler().fit(train[columns])
        train_scaled = scaler.transform(train[columns])
        count = min(pca_components, train_scaled.shape[0], train_scaled.shape[1])
        pca = PCA(n_components=count, whiten=True, random_state=0).fit(train_scaled)
        transform = lambda frame: pca.transform(scaler.transform(frame[columns]))
        metadata = {
            "dimension": count,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
        }
    train_rep = transform(train)
    val_rep = transform(model_val)
    cal_rep = transform(calibration)
    test_rep = transform(test)
    metadata["val_residual_similarity_spearman"] = residual_similarity_correlation(
        val_rep, model_val["residual"].to_numpy()
    )
    metadata["cal_test_nearest_distance"] = {
        "min": float(pairwise_euclidean(test_rep, cal_rep).min(axis=1).min()),
        "median": float(np.median(pairwise_euclidean(test_rep, cal_rep).min(axis=1))),
        "max": float(pairwise_euclidean(test_rep, cal_rep).min(axis=1).max()),
    }
    return cal_rep, test_rep, metadata


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    run_dir = args.output_dir / f"rul_landmark_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading MATR cells from {args.data_root} ...", flush=True)
    cells = load_cells(args.data_root, args.start_cycle)
    splits = official_splits(cells, args)
    (run_dir / "split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    medians, input_scaler = fit_input_scaler(cells, splits["train"])
    cells = transform_cells(cells, medians, input_scaler)
    print(f"Loaded {len(cells)} cells; splits: " + str({k: len(v) for k, v in splits.items()}), flush=True)

    model = MambaRUL(len(feature_names()), args.hidden, args.layers).to(device)
    model, training = train_model(model, cells, splits, args, device)
    pd.DataFrame(training["history"]).to_csv(run_dir / "training_history.csv", index=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_medians": medians,
        "input_scaler_mean": input_scaler.mean_,
        "input_scaler_scale": input_scaler.scale_,
        "label_scale": training["label_scale"],
        "config": vars(args),
    }, run_dir / "model.pt")

    frames = {}
    all_frames = []
    for split, names in splits.items():
        frame = predict_landmarks(
            model, cells, names, args.landmarks, training["label_scale"], device, args.batch_size
        )
        frame.insert(0, "split", split)
        frame = add_state_features(frame, cells)
        frames[split] = frame
        all_frames.append(frame)
    predictions = pd.concat(all_frames, ignore_index=True)
    predictions.to_csv(run_dir / "predictions_all.csv", index=False)

    point_results, split_results = {}, {}
    lcp_results = {"state_lcp": {}, "pca_lcp": {}}
    diagnostics = []
    landmark_outputs = []
    for t in args.landmarks:
        subsets = {name: frame[frame["t"] == t].reset_index(drop=True) for name, frame in frames.items()}
        cal, test = subsets["calibration"], subsets["test"]
        y_cal, pred_cal = cal["true_rul"].to_numpy(), cal["pred_rul"].to_numpy()
        y_test, pred_test = test["true_rul"].to_numpy(), test["pred_rul"].to_numpy()
        scores = np.abs(y_cal - pred_cal)
        cp_low, cp_high, threshold = split_cp(scores, pred_test, args.alpha)
        point_results[str(t)] = regression_metrics(y_test, pred_test)
        split_results[str(t)] = {**interval_metrics(y_test, cp_low, cp_high), "threshold": threshold}
        output = test[["cell_id", "t", "true_rul", "pred_rul", "residual"]].copy()
        output["split_cp_lower"], output["split_cp_upper"] = cp_low, cp_high

        for kind, result_name in (("state", "state_lcp"), ("pca", "pca_lcp")):
            cal_z, test_z, metadata = prepare_representations(
                subsets["train"], subsets["model_val"], cal, test, kind, args.pca_components
            )
            lcp = adaptive_lcp(
                scores, cal_z, pred_test, test_z, threshold, args.alpha,
                args.lcp_k, args.min_ess,
            )
            metrics = interval_metrics(y_test, lcp.lower, lcp.upper)
            metrics.update({
                "ess_min": float(lcp.ess.min()),
                "ess_median": float(np.median(lcp.ess)),
                "ess_mean": float(lcp.ess.mean()),
                "fallback_fraction": float(lcp.fallback.mean()),
                "mean_width_change_vs_split": float(metrics["mean_width"] - split_results[str(t)]["mean_width"]),
                "representation": metadata,
            })
            lcp_results[result_name][str(t)] = metrics
            output[f"{result_name}_lower"] = lcp.lower
            output[f"{result_name}_upper"] = lcp.upper
            output[f"{result_name}_delta"] = lcp.delta
            output[f"{result_name}_ess"] = lcp.ess
            output[f"{result_name}_bandwidth"] = lcp.bandwidth
            output[f"{result_name}_fallback"] = lcp.fallback
            for i, row in output.iterrows():
                diagnostics.append({
                    "cell_id": row["cell_id"], "t": t, "method": result_name,
                    "ess": lcp.ess[i], "bandwidth": lcp.bandwidth[i],
                    "fallback": bool(lcp.fallback[i]),
                })
        output.to_csv(run_dir / f"landmark_{t:03d}.csv", index=False)
        landmark_outputs.append(output)
        print(
            f"t={t}: point RMSE={point_results[str(t)]['rmse']:.2f}; "
            f"Split coverage={split_results[str(t)]['coverage']:.3f}, "
            f"width={split_results[str(t)]['mean_width']:.2f}; "
            f"State-LCP width={lcp_results['state_lcp'][str(t)]['mean_width']:.2f}; "
            f"PCA-LCP width={lcp_results['pca_lcp'][str(t)]['mean_width']:.2f}",
            flush=True,
        )

    pd.DataFrame(diagnostics).to_csv(run_dir / "neighborhood_diagnostics.csv", index=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics_point.json").write_text(json.dumps(point_results, indent=2), encoding="utf-8")
    (run_dir / "metrics_split_cp.json").write_text(json.dumps(split_results, indent=2), encoding="utf-8")
    (run_dir / "metrics_state_lcp.json").write_text(json.dumps(lcp_results["state_lcp"], indent=2), encoding="utf-8")
    (run_dir / "metrics_pca_lcp.json").write_text(json.dumps(lcp_results["pca_lcp"], indent=2), encoding="utf-8")
    summary = {
        "split_counts": {k: len(v) for k, v in splits.items()},
        "training": {k: v for k, v in training.items() if k != "history"},
        "point": point_results,
        "split_cp": split_results,
        **lcp_results,
        "artifacts": str(run_dir.resolve()),
    }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
