"""Stage missing MIX_large cells from the official 2025-03-05 Hugging Face snapshot.

The snapshot is pinned to the day of the Zenodo v4 release. Files are downloaded
into a separate directory and checked against their published LFS SHA-256 values
before any dataset directory is moved into the active data root.
"""

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

from audit_mix_large import audit, split_lists


HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:\BatteryLife_v4")
STAGING = ROOT / "_hf_staging"
CACHE = ROOT / "_hf_cache"
STATE = HERE / "hf_download_state.json"
ZENODO_MANIFEST = HERE / "zenodo_tongji_repair_manifest.json"
REPO = "Battery-Life/BatteryLife_Processed"
REVISION = "f7a45ed5598569d3d0fdb3019bc81dd92d756c28"
GROUPS = ("RWTH", "Stanford", "Tongji", "ISU_ILCC", "XJTU")
WORKERS = 8

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
disable_progress_bars()
ZENODO_FILES = (json.loads(ZENODO_MANIFEST.read_text(encoding="utf-8"))["files"]
                if ZENODO_MANIFEST.is_file() else {})


def brief_error(exc):
    return f"{type(exc).__name__}: {str(exc).split(' for url:')[0][:250]}"


def write_state(state, **extra):
    payload = {"state": state, "time": datetime.now().isoformat(timespec="seconds"),
               "source": REPO, "revision": REVISION, **extra}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(100):
        try:
            temporary.replace(STATE)
            break
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(path, entry):
    destination = STAGING / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path in ZENODO_FILES:
        repair = ZENODO_FILES[path]
        if (not destination.is_file() or destination.stat().st_size != repair["size"]
                or sha256(destination) != repair["sha256"]):
            raise ValueError(f"Zenodo v4 repair verification failed: {path}")
        if repair["size"] != entry.size:
            raise ValueError(f"Zenodo/HF published size mismatch: {path}")
        return path, repair["size"]
    expected_size = entry.size
    expected_hash = entry.lfs["sha256"]
    if destination.is_file() and destination.stat().st_size == expected_size:
        if sha256(destination) == expected_hash:
            return path, expected_size
    for attempt in range(1, 7):
        try:
            local = Path(hf_hub_download(
                repo_id=REPO, repo_type="dataset", revision=REVISION,
                filename=path, cache_dir=CACHE, local_dir=STAGING,
                local_dir_use_symlinks=False, resume_download=True,
            ))
            if local.stat().st_size != expected_size or sha256(local) != expected_hash:
                raise ValueError(f"size or SHA-256 mismatch: {path}")
            return path, expected_size
        except Exception as exc:
            if attempt == 6:
                raise RuntimeError(f"{path}: failed after {attempt} attempts: {brief_error(exc)}") from exc
            print(f"retry {path} ({attempt}/6): {brief_error(exc)}", flush=True)
            time.sleep(min(60, attempt * 5))


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError(ROOT)
    splits = split_lists()
    files = sorted({f"{group}/{name}" for group in GROUPS
                    for phase in ("train", "val", "test")
                    for name in splits[f"{group}_{phase}_files"]})
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    entries = {item.rfilename: item for item in info.siblings}
    missing = [path for path in files if path not in entries or not entries[path].lfs
               or not entries[path].lfs.get("sha256")]
    if missing:
        raise RuntimeError(f"Missing official SHA-256 metadata for {missing[:10]}")
    total_bytes = sum(entries[path].size for path in files)
    free = shutil.disk_usage(ROOT).free
    if free < 2 * total_bytes + 10_000_000_000:
        raise RuntimeError(f"Insufficient C: space: {free/1e9:.1f} GB free")
    STAGING.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    write_state("downloading", verified_files=0, total_files=len(files),
                verified_gb=0, total_gb=round(total_bytes / 1e9, 2))
    print(f"Official HF snapshot {REVISION}; {len(files)} files, "
          f"{total_bytes/1e9:.2f} GB; {WORKERS} workers", flush=True)
    verified_files = verified_bytes = 0
    failures = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(fetch, path, entries[path]): path for path in files}
        while futures:
            done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                path = futures.pop(future)
                try:
                    _, size = future.result()
                except Exception as exc:
                    failures[path] = brief_error(exc)
                    print(f"failed {path}: {failures[path]}", flush=True)
                else:
                    verified_files += 1
                    verified_bytes += size
                    print(f"verified {verified_files}/{len(files)} "
                          f"{verified_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB: {path}",
                          flush=True)
                write_state("downloading", verified_files=verified_files,
                            total_files=len(files), verified_gb=round(verified_bytes/1e9, 2),
                            total_gb=round(total_bytes/1e9, 2),
                            failures=len(failures))
            if not done:
                print(f"waiting: verified {verified_files}/{len(files)}, "
                      f"{verified_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB", flush=True)
    if failures:
        write_state("failed", failures=failures, verified_files=verified_files)
        raise RuntimeError(f"Failed to verify {len(failures)} files")
    for group in GROUPS:
        source = STAGING / group
        target = ROOT / group
        if target.exists():
            raise FileExistsError(f"Refusing to replace existing dataset: {target}")
        source.rename(target)
        print(f"activated {target}", flush=True)
    inventory = audit(ROOT)
    if any(inventory["phases"][phase]["files_present"] != inventory["phases"][phase]["expected"]
           for phase in ("train", "val", "test")):
        write_state("audit_failed", inventory=inventory["phases"])
        raise RuntimeError("Full MIX_large file audit failed")
    write_state("complete", verified_files=verified_files, total_files=len(files),
                verified_gb=round(verified_bytes/1e9, 2), total_gb=round(total_bytes/1e9, 2))
    print("v4-equivalent MIX_large files are ready", flush=True)


if __name__ == "__main__":
    main()
