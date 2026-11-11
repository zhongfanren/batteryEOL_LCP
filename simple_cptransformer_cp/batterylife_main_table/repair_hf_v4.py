"""Fresh-download failed HF files after a bad HTTP Range resume."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars


HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:\BatteryLife_v4")
STATE = json.loads((HERE / "hf_download_state.json").read_text(encoding="utf-8"))
REPO = STATE["source"]
REVISION = STATE["revision"]
FILES = sorted(STATE["failures"])

disable_progress_bars()
info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
entries = {entry.rfilename: entry for entry in info.siblings}

for name in FILES:
    entry = entries[name]
    path = Path(hf_hub_download(
        repo_id=REPO, repo_type="dataset", revision=REVISION,
        filename=name, cache_dir=ROOT / "_hf_cache",
        local_dir=ROOT / "_hf_staging", local_dir_use_symlinks=False,
        force_download=True, resume_download=False,
    ))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != entry.size or digest != entry.lfs["sha256"]:
        raise ValueError(f"size or SHA-256 mismatch: {name}")
    print(f"repaired and verified: {name} ({entry.size} bytes)", flush=True)
