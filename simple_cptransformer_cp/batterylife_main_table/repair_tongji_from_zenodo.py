"""Fetch only the five bad Tongji members from the official Zenodo v4 ZIP.

The Hugging Face historical snapshot has inconsistent size/object metadata for
these members. HTTP Range requests fetch the ZIP directory and just the needed
compressed members. Python's ZIP reader verifies each member's CRC32.
"""

import hashlib
import json
import struct
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:\BatteryLife_v4")
STATE = json.loads((HERE / "hf_download_state.json").read_text(encoding="utf-8"))
RECORD = json.loads((HERE / "downloads_v4" / "zenodo_record.json").read_text(encoding="utf-8"))
ENTRY = next(item for item in RECORD["files"] if item["key"] == "Tongji.zip")
URL = ENTRY["links"]["self"]
TOTAL = ENTRY["size"]
NAMES = sorted(STATE["failures"])
SPARSE = ROOT / "_archives" / "Tongji_range_repair.sparse"
TAIL_PROBE = HERE / "tongji_zip_tail_probe.bin"
MANIFEST = HERE / "zenodo_tongji_repair_manifest.json"
SESSION = requests.Session()


def fetch_range(start, end):
    for attempt in range(1, 7):
        try:
            response = SESSION.get(URL, headers={"Range": f"bytes={start}-{end}"},
                                   stream=True, timeout=(30, 120))
            if response.status_code != 206:
                raise ValueError(f"HTTP {response.status_code}, expected 206")
            if response.headers.get("Content-Range") != f"bytes {start}-{end}/{TOTAL}":
                raise ValueError("Content-Range mismatch")
            data = b"".join(response.iter_content(1024 * 1024))
            if len(data) != end - start + 1:
                raise ValueError(f"range length mismatch: {len(data)}")
            return data
        except (requests.RequestException, ValueError) as exc:
            if attempt == 6:
                raise RuntimeError(f"range {start}-{end} failed after {attempt} attempts") from exc
            print(f"retry range {start}-{end} ({attempt}/6): {type(exc).__name__}", flush=True)
            time.sleep(min(30, attempt * 3))


def main():
    if len(NAMES) != 5 or any(not name.startswith("Tongji/") for name in NAMES):
        raise RuntimeError(f"Unexpected failed-file list: {NAMES}")
    tail_start = TOTAL - 1024 * 1024
    tail = TAIL_PROBE.read_bytes() if TAIL_PROBE.is_file() else b""
    if len(tail) != TOTAL - tail_start:
        tail = fetch_range(tail_start, TOTAL - 1)
    with SPARSE.open("w+b") as output:
        output.truncate(TOTAL)
        output.seek(tail_start)
        output.write(tail)
    print("Zenodo v4 ZIP directory fetched", flush=True)

    with zipfile.ZipFile(SPARSE) as archive:
        selected = [(name, archive.getinfo(name)) for name in NAMES]
    manifest = {"source_record": 14969822, "archive": "Tongji.zip",
                "time": datetime.now().isoformat(timespec="seconds"), "files": {}}
    for name, info in selected:
        offset = info.header_offset
        header = fetch_range(offset, offset + 29)
        if header[:4] != b"PK\x03\x04":
            raise ValueError(f"Bad ZIP local header: {name}")
        filename_length, extra_length = struct.unpack_from("<HH", header, 26)
        end = offset + 30 + filename_length + extra_length + info.compress_size - 1
        payload = fetch_range(offset + 30, end)
        with SPARSE.open("r+b") as output:
            output.seek(offset)
            output.write(header)
            output.write(payload)
        with zipfile.ZipFile(SPARSE) as archive:
            data = archive.read(name)  # Verifies the ZIP member's CRC32.
        if len(data) != info.file_size:
            raise ValueError(f"Uncompressed size mismatch: {name}")
        destination = ROOT / "_hf_staging" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".zenodo.tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
        manifest["files"][name] = {
            "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "crc32": f"{info.CRC:08x}", "compressed_bytes": info.compress_size,
        }
        print(f"Zenodo v4 repaired {name}: {len(data)} bytes", flush=True)
    temporary_manifest = MANIFEST.with_suffix(".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(MANIFEST)
    SPARSE.unlink()
    print("Five Tongji files verified from Zenodo v4", flush=True)


if __name__ == "__main__":
    main()
