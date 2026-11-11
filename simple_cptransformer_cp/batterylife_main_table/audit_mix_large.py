"""Read-only inventory for the upstream BatteryLife MIX_large protocol."""

import argparse
import ast
import json
from pathlib import Path


UPSTREAM = Path(__file__).resolve().parents[2] / "third_party" / "BatteryLife"
SPLITS = UPSTREAM / "data_provider" / "data_split_recorder.py"


def split_lists():
    cls = ast.parse(SPLITS.read_text(encoding="utf-8")).body[0]
    values = {}

    def read(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return read(node.left) + read(node.right)
        return ast.literal_eval(node)

    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            values[stmt.targets[0].id] = read(stmt.value)
    return values


def source_path(root, dataset, name):
    return root / dataset / name


def label_path(root, dataset):
    if dataset in {"MICH", "MICH_EXP"}:
        name = "total_MICH_labels.json"
    elif dataset == "Tongji":
        name = "Tongji_labels.json"
    else:
        prefix = {"UL_PUR": "UL-PUR", "ISU_ILCC": "ISU-ILCC"}.get(dataset, dataset)
        name = f"{prefix}_labels.json"
    return root / "Life labels" / name


def audit(root):
    values = split_lists()
    phases = {}
    for phase in ("train", "val", "test"):
        all_names = values[f"MIX_large_{phase}_files"]
        rows = []
        for dataset in ("UL_PUR", "RWTH", "HUST", "MATR", "Stanford", "Tongji",
                        "ISU_ILCC", "CALCE", "HNEI", "SNL", "MICH", "MICH_EXP", "XJTU"):
            names = values[f"{dataset}_{phase}_files"]
            if not names:
                continue
            label_file = label_path(root, dataset)
            labels = json.loads(label_file.read_text(encoding="utf-8")) if label_file.is_file() else {}
            def label_key(name):
                return name.replace("--", "-#") if dataset == "Tongji" else name
            rows.append({"dataset": dataset, "expected": len(names),
                         "files_present": sum(source_path(root, dataset, n).is_file() for n in names),
                         "labels_present": sum(label_key(n) in labels for n in names),
                         "label_file_present": label_file.is_file()})
        assert len(all_names) == sum(r["expected"] for r in rows)
        phases[phase] = {"expected": len(all_names), "unique": len(set(all_names)),
                         "files_present": sum(r["files_present"] for r in rows),
                         "labels_present": sum(r["labels_present"] for r in rows),
                         "datasets": rows}
    sets = [set(values[f"MIX_large_{p}_files"]) for p in ("train", "val", "test")]
    seen_unseen = root / "seen_unseen_labels" / "cal_for_test.json"
    if not seen_unseen.is_file():
        seen_unseen = UPSTREAM / "dataset" / "seen_unseen_labels" / "cal_for_test.json"
    tags = json.loads(seen_unseen.read_text(encoding="utf-8")) if seen_unseen.is_file() else {}
    return {"data_root": str(root.resolve()), "split_source": str(SPLITS),
            "phases": phases, "cross_split_overlap": [len(sets[0] & sets[1]),
            len(sets[0] & sets[2]), len(sets[1] & sets[2])],
            "test_seen_unseen_source": str(seen_unseen) if seen_unseen.is_file() else None,
            "test_seen_unseen_keys_present": len(sets[2] & set(tags))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    args = parser.parse_args()
    print(json.dumps(audit(args.data_root), indent=2))
