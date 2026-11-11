"""Leakage-free windowed RUL baseline for the MambaLithium CaseA/CaseB data.

The original Mamba implementation is imported read-only from the adjacent
``MambaLithium`` checkout.  This script intentionally changes the evaluation
protocol:

* feature scaling is fitted on training cells only;
* each input contains only a fixed history ending at the prediction cycle;
* the target is log1p(RUL), so test-cell lifetime is never used as a scale;
* the test cell is used only after training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
OFFICIAL_DIR = HERE.parent / "MambaLithium"
sys.path.insert(0, str(OFFICIAL_DIR))
from mamba import Mamba, MambaConfig  # noqa: E402


CASE_SPLITS = {
    "A": {"train": ["91.csv", "100.csv"], "test": ["124.csv"]},
    "B": {
        "train": ["101.csv", "108.csv", "120.csv"],
        "test": ["116.csv"],
    },
}


@dataclass(frozen=True)
class RunConfig:
    case: str
    window: int
    stride: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    layers: int
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("A", "B"), default="A")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def read_cells(case: str, names: list[str]) -> dict[str, pd.DataFrame]:
    data_dir = OFFICIAL_DIR / "data" / f"Case{case}"
    cells = {}
    for name in names:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"Cycle", "RUL", "SOH"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        cells[path.stem] = frame
    return cells


def feature_columns(frame: pd.DataFrame) -> list[str]:
    # Match the official baseline: SOH is a target, not an input feature.
    return [column for column in frame.columns if column not in {"RUL", "SOH"}]


def fit_train_scaler(
    train_cells: dict[str, pd.DataFrame], columns: list[str]
) -> StandardScaler:
    train_rows = np.concatenate(
        [cell[columns].to_numpy(dtype=np.float64) for cell in train_cells.values()],
        axis=0,
    )
    scaler = StandardScaler()
    scaler.fit(train_rows)
    return scaler


def make_windows(
    cells: dict[str, pd.DataFrame],
    columns: list[str],
    scaler: StandardScaler,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    targets: list[float] = []
    records: list[dict[str, float | int | str]] = []

    for cell_id, frame in cells.items():
        features = scaler.transform(frame[columns].to_numpy(dtype=np.float64))
        rul = frame["RUL"].to_numpy(dtype=np.float64)
        cycles = frame["Cycle"].to_numpy()

        if len(frame) < window:
            continue

        # Each endpoint uses rows [end-window+1, end] and no future features.
        endpoints = list(range(window - 1, len(frame), stride))
        if endpoints[-1] != len(frame) - 1:
            endpoints.append(len(frame) - 1)

        for end in endpoints:
            true_rul = float(rul[end])
            windows.append(features[end - window + 1 : end + 1])
            targets.append(float(np.log1p(true_rul)))
            records.append(
                {
                    "cell_id": cell_id,
                    "row_index": int(end),
                    "cycle": float(cycles[end]),
                    "true_rul": true_rul,
                    "true_log_rul": float(np.log1p(true_rul)),
                }
            )

    if not windows:
        raise ValueError("No windows were created. Reduce --window.")

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.DataFrame.from_records(records),
    )


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class MambaRUL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.encoder = Mamba(MambaConfig(d_model=hidden_dim, n_layers=layers))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(self.input_projection(x))
        embedding = hidden[:, -1, :]
        prediction = self.head(embedding).squeeze(-1)
        if return_embedding:
            return prediction, embedding
        return prediction


def train(
    model: MambaRUL,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> list[float]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.L1Loss()
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_count = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(targets)
            total_count += len(targets)

        mean_loss = total_loss / total_count
        history.append(mean_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d} | train log-RUL MAE: {mean_loss:.6f}")

    return history


@torch.no_grad()
def predict(
    model: MambaRUL,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(WindowDataset(x, np.zeros(len(x), dtype=np.float32)),
                        batch_size=batch_size, shuffle=False)
    predictions = []
    embeddings = []
    model.eval()
    for features, _ in loader:
        pred, embedding = model(features.to(device), return_embedding=True)
        predictions.append(pred.cpu().numpy())
        embeddings.append(embedding.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(embeddings)


def metrics(true_rul: np.ndarray, pred_rul: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(true_rul, pred_rul)),
        "rmse": float(np.sqrt(mean_squared_error(true_rul, pred_rul))),
        "mae": float(mean_absolute_error(true_rul, pred_rul)),
        "r2": float(r2_score(true_rul, pred_rul)),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    split = CASE_SPLITS[args.case]

    train_cells = read_cells(args.case, split["train"])
    test_cells = read_cells(args.case, split["test"])
    columns = feature_columns(next(iter(train_cells.values())))
    scaler = fit_train_scaler(train_cells, columns)

    x_train, y_train, train_meta = make_windows(
        train_cells, columns, scaler, args.window, args.stride
    )
    x_test, _, test_meta = make_windows(
        test_cells, columns, scaler, args.window, args.stride
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        WindowDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = MambaRUL(
        input_dim=len(columns), hidden_dim=args.hidden, layers=args.layers
    ).to(device)
    history = train(
        model,
        loader,
        device,
        args.epochs,
        args.lr,
        args.wd,
    )

    pred_log_rul, embeddings = predict(
        model, x_test, device, args.batch_size
    )
    pred_rul = np.maximum(np.expm1(pred_log_rul), 0.0)
    true_rul = test_meta["true_rul"].to_numpy(dtype=np.float64)
    result_metrics = metrics(true_rul, pred_rul)

    run_name = f"case{args.case}_seed{args.seed}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = test_meta.copy()
    predictions["pred_log_rul"] = pred_log_rul
    predictions["pred_rul"] = pred_rul
    predictions["residual"] = true_rul - pred_rul
    predictions["abs_residual"] = np.abs(predictions["residual"])
    for index in range(embeddings.shape[1]):
        predictions[f"embedding_{index}"] = embeddings[:, index]
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    config = RunConfig(
        case=args.case,
        window=args.window,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.wd,
        hidden_dim=args.hidden,
        layers=args.layers,
        seed=args.seed,
        device=str(device),
    )
    metadata = {
        "config": asdict(config),
        "feature_columns": columns,
        "train_cells": sorted(train_cells),
        "test_cells": sorted(test_cells),
        "n_train_windows": len(train_meta),
        "n_test_windows": len(test_meta),
        "metrics": result_metrics,
        "leakage_controls": {
            "scaler_fit_cells": sorted(train_cells),
            "test_cell_used_for_scaler": False,
            "target": "log1p(RUL)",
            "input_uses_future_rows": False,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(columns),
            "hidden_dim": args.hidden,
            "layers": args.layers,
            "feature_columns": columns,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "config": asdict(config),
        },
        output_dir / "model.pt",
    )

    pd.DataFrame(
        {"epoch": np.arange(1, len(history) + 1), "train_loss": history}
    ).to_csv(output_dir / "training_history.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    for cell_id, group in predictions.groupby("cell_id", sort=False):
        plt.plot(group["cycle"], group["true_rul"], label=f"{cell_id} true")
        plt.plot(
            group["cycle"],
            group["pred_rul"],
            linestyle="--",
            label=f"{cell_id} prediction",
        )
    plt.xlabel("Cycle")
    plt.ylabel("RUL (cycles)")
    plt.title(f"Leakage-free Mamba RUL - Case {args.case}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rul_prediction.png", dpi=160)
    plt.close()

    print("MSE RMSE MAE R2")
    print(
        f"{result_metrics['mse']:.4f} {result_metrics['rmse']:.4f} "
        f"{result_metrics['mae']:.4f} {result_metrics['r2']:.4f}"
    )
    print(f"Features: {columns}")
    print(f"Train windows: {len(x_train)} | Test windows: {len(x_test)}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
