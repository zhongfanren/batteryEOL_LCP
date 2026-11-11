"""Check whether the official Hugging Face historical snapshot matches Zenodo v4."""

import hashlib
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REVISION = "f7a45ed5598569d3d0fdb3019bc81dd92d756c28"
REPO = "Battery-Life/BatteryLife_Processed"
DATA_ROOT = Path(r"C:\BatteryLife_v4")
FILES = [
    ("Life labels/ISU-ILCC_labels.json", "Life labels.zip"),
    ("UL_PUR/UL-PUR_N10-NA7_18650_NCA_23C_0-100_0.5-0.5C_g.pkl", "UL_PUR.zip"),
    ("HNEI/HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_a.pkl", "HNEI.zip"),
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


for filename, archive in FILES:
    start = time.monotonic()
    local = hf_hub_download(
        repo_id=REPO, repo_type="dataset", revision=REVISION, filename=filename
    )
    elapsed = time.monotonic() - start
    hf_data = Path(local).read_bytes()
    with zipfile.ZipFile(DATA_ROOT / "_archives" / archive) as z:
        v4_data = z.read(filename)
    print(
        f"{filename}: size={len(hf_data)} bytes, "
        f"elapsed={elapsed:.2f}s, MiB/s={len(hf_data) / 1048576 / max(elapsed, 1e-9):.2f}, "
        f"exact_match={sha256(hf_data) == sha256(v4_data)}",
        flush=True,
    )
