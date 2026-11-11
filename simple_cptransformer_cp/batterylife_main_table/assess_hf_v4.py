"""Assess the exact historical Hugging Face snapshot for MIX_large files."""

from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

from audit_mix_large import split_lists


REVISION = "f7a45ed5598569d3d0fdb3019bc81dd92d756c28"
ROOT = Path(r"C:\BatteryLife_v4")
SPLITS = split_lists()
info = HfApi().dataset_info(
    "Battery-Life/BatteryLife_Processed", revision=REVISION, files_metadata=True
)
sizes = {item.rfilename: item.size for item in info.siblings}
groups = defaultdict(lambda: {"count": 0, "bytes": 0, "missing_hf": 0})
for dataset in (
    "UL_PUR", "RWTH", "HUST", "MATR", "Stanford", "Tongji", "ISU_ILCC",
    "CALCE", "HNEI", "SNL", "MICH", "MICH_EXP", "XJTU",
):
    names = set().union(*(SPLITS[f"{dataset}_{phase}_files"] for phase in ("train", "val", "test")))
    for name in names:
        if (ROOT / dataset / name).is_file():
            continue
        path = f"{dataset}/{name}"
        if path not in sizes:
            groups[dataset]["missing_hf"] += 1
        else:
            groups[dataset]["count"] += 1
            groups[dataset]["bytes"] += sizes[path]

print("revision", REVISION)
for group, row in groups.items():
    print(group, row["count"], f"{row['bytes'] / 1e9:.2f} GB", "missing_hf", row["missing_hf"])
print("TOTAL", sum(r["count"] for r in groups.values()),
      f"{sum(r['bytes'] for r in groups.values()) / 1e9:.2f} GB",
      "missing_hf", sum(r["missing_hf"] for r in groups.values()))
