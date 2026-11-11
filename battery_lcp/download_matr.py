"""Download only the gated BatteryLife MATR data and its life labels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Prefer setting HF_TOKEN instead.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install the downloader first: pip install huggingface_hub"
        ) from exc

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "BatteryLife_Processed is gated. Accept its access conditions at "
            "https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed "
            "and set HF_TOKEN before running this command."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Battery-Life/BatteryLife_Processed",
        repo_type="dataset",
        token=token,
        local_dir=args.output_dir,
        allow_patterns=["MATR/*", "Life labels/MATR_labels.json"],
    )
    print(f"Downloaded MATR data to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
