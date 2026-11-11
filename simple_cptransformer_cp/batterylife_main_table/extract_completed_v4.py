"""Verify and extract v4 archives that completed before switching mirrors."""

from pathlib import Path

from stage_v4 import METADATA, extract, md5


ROOT = Path(r"C:\BatteryLife_v4")
ARCHIVES = ROOT / "_archives"
NAMES = ("CALCE.zip", "HNEI.zip", "SNL.zip", "MICH.zip", "MICH_EXP.zip")

for name in NAMES:
    path = ARCHIVES / name
    entry = METADATA[name]
    if path.stat().st_size != entry["size"] or md5(path) != entry["checksum"].removeprefix("md5:"):
        raise ValueError(f"Published Zenodo v4 MD5 mismatch: {path}")
    print(f"MD5 verified: {name}", flush=True)
    extract(name, ARCHIVES, ROOT)
print("completed v4 archives extracted", flush=True)
