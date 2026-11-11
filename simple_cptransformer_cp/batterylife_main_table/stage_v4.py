"""Download, verify, and extract the missing official v4 MIX_large archives.

Requires network access and a writable data root with at least 70 GB free.
Existing HUST/MATR cell directories may be linked into that root separately.
"""

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
RECORD = json.loads((HERE / "downloads_v4" / "zenodo_record.json").read_text())
FILES = ["UL_PUR.zip", "RWTH.zip", "Stanford.zip", "Tongji.zip",
         "ISU_ILCC.zip", "CALCE.zip", "HNEI.zip", "SNL.zip",
         "MICH.zip", "MICH_EXP.zip", "XJTU.zip", "Life labels.zip"]
METADATA = {entry["key"]: entry for entry in RECORD["files"]}


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name, archives):
    info = METADATA[name]
    target = archives / name
    expected = info["checksum"].removeprefix("md5:")
    url = info["links"]["self"]
    deadline = time.monotonic() + 44 * 3600
    attempt = 0
    while True:
        size = target.stat().st_size if target.is_file() else 0
        if size > info["size"]:
            raise ValueError(f"{name}: archive is larger than published size")
        if size == info["size"]:
            if md5(target) != expected:
                raise ValueError(f"{name}: MD5 mismatch; retained for inspection")
            return target
        result = subprocess.run(["curl.exe", "--fail", "--location", "--silent", "--show-error",
                                 "--connect-timeout", "30", "--speed-limit", "1024",
                                 "--speed-time", "120", "--continue-at", "-",
                                 "--output", str(target), url], capture_output=True, text=True)
        attempt += 1
        if result.returncode == 0:
            continue  # Check the published size and MD5 before accepting a transfer.
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{name}: download timed out after {attempt} attempts: {result.stderr.strip()}")
        print(f"retry {name} attempt {attempt}: curl exited {result.returncode}: "
              f"{result.stderr.strip()}", flush=True)
        time.sleep(min(60, 5 * attempt))


def extract(name, archives, root):
    archive = archives / name
    folder_name = "Life labels" if name == "Life labels.zip" else name.removesuffix(".zip")
    final = root / folder_name
    if final.exists():
        # UL_PUR and Life labels were extracted by the small-file preflight.
        if name in {"UL_PUR.zip", "Life labels.zip"}:
            return
        raise FileExistsError(f"Will not overwrite existing data: {final}")
    temporary = root / ("_extracting_" + folder_name)
    if temporary.exists():
        raise FileExistsError(f"Incomplete extraction needs inspection: {temporary}")
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        for entry in entries:
            parts = PurePosixPath(entry.filename.replace("\\", "/")).parts
            if not parts or parts[0] != folder_name or ".." in parts or PurePosixPath(entry.filename).is_absolute():
                raise ValueError(f"Unsafe ZIP member in {name}: {entry.filename}")
        uncompressed = sum(entry.file_size for entry in entries)
        print(f"extract {name}: {len(entries)} members, {uncompressed/1e9:.2f} GB", flush=True)
        temporary.mkdir()
        z.extractall(temporary)
    (temporary / folder_name).rename(final)
    temporary.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"C:\BatteryLife_v4"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    archives = root / "_archives"
    archives.mkdir(exist_ok=True)
    pending = set(FILES)
    failures = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download, name, archives): name for name in FILES}
        while futures:
            done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                name = futures.pop(future)
                pending.remove(name)
                try:
                    path = future.result()
                except Exception as exc:
                    failures[name] = str(exc)
                    print(f"failed {name}: {exc}", flush=True)
                else:
                    print(f"verified {name}: {path.stat().st_size/1e9:.2f} GB", flush=True)
            if not done:
                sizes = {name: round((archives / name).stat().st_size / 1e9, 2)
                         for name in sorted(pending) if (archives / name).exists()}
                print(f"downloading: {sizes}", flush=True)
    if failures:
        raise RuntimeError(f"Unverified archives: {failures}")
    for name in FILES:
        extract(name, archives, root)
    print("v4 missing-component staging completed", flush=True)


if __name__ == "__main__":
    main()
