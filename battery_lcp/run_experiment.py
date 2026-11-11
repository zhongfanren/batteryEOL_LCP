"""BatteryLife MATR -> Mamba lifetime prediction -> Split CP and LCP.

Each battery contributes exactly one sample: its first ``early_cycles`` cycles
predict its total cycle life. Splits are disjoint at cell level.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
OFFICIAL_MAMBA = WORKSPACE / "baselines" / "MambaLithium"
PYTHON_LCP = WORKSPACE / "python_lcp"
BATTERYLIFE_REPO = WORKSPACE / "third_party" / "BatteryLife"
sys.path.insert(0, str(OFFICIAL_MAMBA))
sys.path.insert(0, str(PYTHON_LCP))

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
    parser.add_argument("--early-cycles", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--split-mode",
        choices=("official", "random"),
        default="official",
        help="Official BatteryLife split, or deterministic random split for smoke data.",
    )
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
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def summary(values: object) -> list[float]:
    array = safe_numeric(values)
    if len(array) == 0:
        return [np.nan] * len(STATS)
    return [
        float(np.mean(array)),
        float(np.std(array)),
        float(np.min(array)),
        float(np.max(array)),
        float(array[-1]),
    ]


def extract_cell_features(record: dict, early_cycles: int) -> np.ndarray:
    cycles = record.get("cycle_data", [])
    if len(cycles) < early_cycles:
        raise ValueError(f"only {len(cycles)} cycles; need {early_cycles}")

    result = []
    for index, cycle in enumerate(cycles[:early_cycles], start=1):
        row = [float(index)]
        for key in SIGNAL_KEYS:
            row.extend(summary(cycle.get(key)))
        result.append(row)
    return np.asarray(result, dtype=np.float32)


def feature_names() -> list[str]:
    names = ["cycle_number"]
    for key in SIGNAL_KEYS:
        names.extend([f"{key}_{stat}" for stat in STATS])
    return names


def load_cells(data_root: Path, early_cycles: int) -> dict[str, dict]:
    matr_dir = data_root / "MATR"
    label_path = data_root / "Life labels" / "MATR_labels.json"
    if not matr_dir.is_dir() or not label_path.exists():
        raise FileNotFoundError(
            f"Expected {matr_dir} and {label_path}. See README.md."
        )

    labels = json.loads(label_path.read_text(encoding="utf-8"))
    cells = {}
    skipped = {}
    for filename, life in labels.items():
        path = matr_dir / filename
        if not path.exists():
            skipped[filename] = "file missing"
            continue
        with path.open("rb") as handle:
            record = pickle.load(handle)
        try:
            features = extract_cell_features(record, early_cycles)
        except (ValueError, TypeError) as exc:
            skipped[filename] = str(exc)
            continue
        cells[filename] = {
            "x": features,
            "life": float(life),
        }

    if not cells:
        raise RuntimeError("No usable MATR cells were loaded")
    if skipped:
        print(f"Skipped {len(skipped)} cells; loaded {len(cells)}")
    return cells


def official_file_lists() -> tuple[list[str], list[str], list[str]]:
    sys.path.insert(0, str(BATTERYLIFE_REPO))
    from data_provider.data_split_recorder import split_recorder

    return (
        list(split_recorder.MATR_train_files),
        list(split_recorder.MATR_val_files),
        list(split_recorder.MATR_test_files),
    )


def split_cells(
    cells: dict[str, dict], mode: str, seed: int
) -> dict[str, list[str]]:
    available = set(cells)
    rng = np.random.default_rng(seed)

    if mode == "official":
        official_train, official_val, official_test = official_file_lists()
        train = [name for name in official_train if name in available]
        val_pool = [name for name in official_val if name in available]
        test = [name for name in official_test if name in available]
        rng.shuffle(val_pool)
        cut = len(val_pool) // 2
        val, cal = val_pool[:cut], val_pool[cut:]
    else:
        names = np.array(sorted(available))
        rng.shuffle(names)
        n = len(names)
        n_train = max(1, int(0.6 * n))
        n_val = max(1, int(0.15 * n))
        n_cal = max(1, int(0.15 * n))
        train = names[:n_train].tolist()
        val = names[n_train : n_train + n_val].tolist()
        cal = names[n_train + n_val : n_train + n_val + n_cal].tolist()
        test = names[n_train + n_val + n_cal :].tolist()

    splits = {"train": train, "val": val, "cal": cal, "test": test}
    if any(len(names) == 0 for names in splits.values()):
        counts = {key: len(value) for key, value in splits.items()}
        raise RuntimeError(f"At least one split is empty: {counts}")
    flat = [name for names in splits.values() for name in names]
    if len(flat) != len(set(flat)):
        raise RuntimeError("Cell leakage detected between splits")
    return splits


def fit_feature_preprocessor(
    cells: dict[str, dict], train_names: list[str]
) -> tuple[np.ndarray, StandardScaler]:
    train_cycles = np.concatenate([cells[name]["x"] for name in train_names])
    medians = np.nanmedian(train_cycles, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isnan(train_cycles), medians, train_cycles)
    scaler = StandardScaler().fit(imputed)
    return medians, scaler


def arrays_for_split(
    cells: dict[str, dict],
    names: list[str],
    medians: np.ndarray,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(
        [
            scaler.transform(np.where(np.isnan(cells[name]["x"]), medians,
                                      cells[name]["x"]))
            for name in names
        ]
    ).astype(np.float32)
    y = np.asarray([cells[name]["life"] for name in names], dtype=np.float32)
    return x, y


class BatteryDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class MambaLife(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden)
        self.encoder = Mamba(MambaConfig(d_model=hidden, n_layers=layers))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sequence = self.encoder(self.input_projection(x))
        embedding = sequence[:, -1]
        prediction = self.head(embedding).squeeze(-1)
        if return_embedding:
            return prediction, embedding
        return prediction


def make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        BatteryDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


@torch.no_grad()
def predict(
    model: MambaLife,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    label_mean: float,
    label_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(
        x, np.zeros(len(x), dtype=np.float32), batch_size, False, 0
    )
    predictions, embeddings = [], []
    model.eval()
    for features, _ in loader:
        scaled, embedding = model(features.to(device), return_embedding=True)
        log_life = scaled.cpu().numpy() * label_scale + label_mean
        predictions.append(np.maximum(np.expm1(log_life), 0.0))
        embeddings.append(embedding.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(embeddings)


def train_model(
    model: MambaLife,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MambaLife, dict]:
    train_log = np.log1p(y_train)
    label_mean = float(train_log.mean())
    label_scale = float(train_log.std())
    if label_scale == 0:
        label_scale = 1.0
    train_target = ((train_log - label_mean) / label_scale).astype(np.float32)
    val_target = ((np.log1p(y_val) - label_mean) / label_scale).astype(np.float32)

    train_loader = make_loader(
        x_train, train_target, args.batch_size, True, args.seed
    )
    val_loader = make_loader(x_val, val_target, args.batch_size, False, 0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    loss_fn = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(target)
        train_loss = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for features, target in val_loader:
                features, target = features.to(device), target.to(device)
                val_sum += float(loss_fn(model(features), target)) * len(target)
        val_loss = val_sum / len(val_loader.dataset)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | train {train_loss:.5f} | val {val_loss:.5f}")
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, {
        "label_mean": label_mean,
        "label_scale": label_scale,
        "best_val_loss": best_val,
        "history": history,
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.mean(np.abs(y - pred) / y)),
        "r2": float(r2_score(y, pred)),
    }


def split_conformal(
    cal_y: np.ndarray, cal_pred: np.ndarray, test_pred: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float]:
    scores = np.abs(cal_y - cal_pred)
    n = len(scores)
    rank = int(math.ceil((n + 1) * (1 - alpha)))
    threshold = math.inf if rank > n else float(np.sort(scores)[rank - 1])
    return (
        np.maximum(test_pred - threshold, 0.0),
        test_pred + threshold,
        threshold,
    )


def localized_conformal(
    cal_y: np.ndarray,
    cal_pred: np.ndarray,
    cal_z: np.ndarray,
    test_pred: np.ndarray,
    test_z: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scores = np.abs(cal_y - cal_pred)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    cal_ordered = cal_z[order]
    d_cal = np.sqrt(
        np.mean((cal_ordered[:, None, :] - cal_ordered[None, :, :]) ** 2, axis=2)
    )
    d_test = np.sqrt(
        np.mean((test_z[:, None, :] - cal_ordered[None, :, :]) ** 2, axis=2)
    )
    nonzero = d_cal[d_cal > 0]
    bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.0

    module = _LCPModuleSimple(d_cal, sorted_scores, h=bandwidth, alpha=alpha)
    module._prepare()
    delta = module._construct(Hnew=d_test, HnewT=d_test.T)
    return (
        np.maximum(test_pred - delta, 0.0),
        test_pred + delta,
        delta,
        bandwidth,
    )


def interval_metrics(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    finite = np.isfinite(lower) & np.isfinite(upper)
    return {
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mean_width": float(np.mean((upper - lower)[finite]))
        if finite.any() else math.inf,
        "median_width": float(np.median((upper - lower)[finite]))
        if finite.any() else math.inf,
        "infinite_fraction": float(np.mean(~finite)),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    cells = load_cells(args.data_root, args.early_cycles)
    splits = split_cells(cells, args.split_mode, args.seed)
    medians, scaler = fit_feature_preprocessor(cells, splits["train"])

    arrays = {}
    for split, names in splits.items():
        arrays[split] = arrays_for_split(cells, names, medians, scaler)

    model = MambaLife(
        input_dim=len(feature_names()), hidden=args.hidden, layers=args.layers
    ).to(device)
    model, training = train_model(
        model,
        *arrays["train"],
        *arrays["val"],
        args,
        device,
    )

    predictions = {}
    embeddings = {}
    for split in ("train", "val", "cal", "test"):
        predictions[split], embeddings[split] = predict(
            model,
            arrays[split][0],
            device,
            args.batch_size,
            training["label_mean"],
            training["label_scale"],
        )

    embedding_scaler = StandardScaler().fit(embeddings["train"])
    cal_z = embedding_scaler.transform(embeddings["cal"])
    test_z = embedding_scaler.transform(embeddings["test"])
    cal_y = arrays["cal"][1]
    test_y = arrays["test"][1]
    minimum_calibration = int(math.ceil(1.0 / args.alpha - 1.0))
    if len(cal_y) < minimum_calibration:
        print(
            f"WARNING: alpha={args.alpha} needs at least "
            f"{minimum_calibration} calibration cells for a finite split-CP "
            f"quantile, but only {len(cal_y)} are available."
        )

    cp_low, cp_high, cp_q = split_conformal(
        cal_y, predictions["cal"], predictions["test"], args.alpha
    )
    lcp_low, lcp_high, lcp_delta, bandwidth = localized_conformal(
        cal_y,
        predictions["cal"],
        cal_z,
        predictions["test"],
        test_z,
        args.alpha,
    )

    point = regression_metrics(test_y, predictions["test"])
    results = {
        "point_prediction": point,
        "split_cp": {
            **interval_metrics(test_y, cp_low, cp_high),
            "threshold": cp_q,
        },
        "lcp": {
            **interval_metrics(test_y, lcp_low, lcp_high),
            "bandwidth": bandwidth,
        },
        "split_counts": {key: len(value) for key, value in splits.items()},
        "split_cells": splits,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "feature_names": feature_names(),
    }

    run_dir = args.output_dir / f"matr_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    pd.DataFrame(training["history"]).to_csv(
        run_dir / "training_history.csv", index=False
    )

    output = pd.DataFrame(
        {
            "cell_id": splits["test"],
            "true_life": test_y,
            "pred_life": predictions["test"],
            "split_cp_lower": cp_low,
            "split_cp_upper": cp_high,
            "lcp_lower": lcp_low,
            "lcp_upper": lcp_high,
            "lcp_delta": lcp_delta,
        }
    )
    for index in range(test_z.shape[1]):
        output[f"embedding_{index}"] = test_z[:, index]
    output.to_csv(run_dir / "test_predictions.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_medians": medians,
            "feature_scaler_mean": scaler.mean_,
            "feature_scaler_scale": scaler.scale_,
            "embedding_scaler_mean": embedding_scaler.mean_,
            "embedding_scaler_scale": embedding_scaler.scale_,
            "label_mean": training["label_mean"],
            "label_scale": training["label_scale"],
        },
        run_dir / "model.pt",
    )

    print(json.dumps({
        "point_prediction": point,
        "split_cp": results["split_cp"],
        "lcp": results["lcp"],
        "split_counts": results["split_counts"],
        "artifacts": str(run_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
