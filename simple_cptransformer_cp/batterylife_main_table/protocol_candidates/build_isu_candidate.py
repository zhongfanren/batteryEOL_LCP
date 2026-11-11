"""Build a reproducible ISU-ILCC T=100 candidate inventory.

Membership comes from the earlier mixed-pilot eligibility audit, whose
loader requires EOL>100 and 100 usable prefixes. The script
reads protocol metadata, but deliberately does not inspect EOL labels or
prediction errors when selecting cells or assigning phases.
"""

import hashlib
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
DATA_ROOT = Path(r"C:\BatteryLife_v4")
POOL = MAIN / "mixed_liion_pilot" / "split_manifest.json"
OUT = HERE / "isu_ilcc_t100_candidate.json"
PATTERN = re.compile(r"ISU-ILCC_G([0-9]+)C([0-9]+)\.pkl$")
PHASE_SIZES = {"train": 97, "dev": 20, "calibration": 61, "test": 61}
SEED = 20261026


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "item"):
        return plain(value.item())
    return value


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False)


def main():
    frozen = json.loads(POOL.read_text(encoding="utf-8"))
    names = sorted(name for name, source in frozen["source_of"].items()
                   if source == "ISU_ILCC")
    assert len(names) == sum(PHASE_SIZES.values()) == 239
    rows = []
    for name in names:
        match = PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"Unexpected cell name: {name}")
        path = DATA_ROOT / "ISU_ILCC" / name
        with path.open("rb") as handle:
            cell = pickle.load(handle)
        metadata = plain({key: cell.get(key) for key in (
            "form_factor", "anode_material", "cathode_material",
            "nominal_capacity_in_Ah", "depth_of_charge", "depth_of_discharge",
            "max_voltage_limit_in_V", "min_voltage_limit_in_V",
            "SOC_interval", "charge_protocol", "discharge_protocol")})
        rows.append({"cell_file": name, "condition": f"G{int(match.group(1))}",
                     "replicate": int(match.group(2)), "metadata": metadata})
        del cell

    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    counts = Counter(len(group) for group in groups.values())
    metadata_keys = ("form_factor", "anode_material", "cathode_material",
                     "nominal_capacity_in_Ah", "depth_of_charge",
                     "depth_of_discharge", "max_voltage_limit_in_V",
                     "min_voltage_limit_in_V", "charge_protocol",
                     "discharge_protocol")
    inconsistent = {}
    rate_conflicts = {}
    condition_summaries = []
    for condition, group in groups.items():
        conflicts = [key for key in metadata_keys if
                     len({canonical(row["metadata"][key]) for row in group}) > 1]
        if conflicts:
            inconsistent[condition] = conflicts
        rate_pairs = [
            (tuple(step.get("rate_in_C") for step in row["metadata"]["charge_protocol"] or []),
             tuple(step.get("rate_in_C") for step in row["metadata"]["discharge_protocol"] or []))
            for row in group
        ]
        if len(set(rate_pairs)) > 1:
            rate_conflicts[condition] = [plain(pair) for pair in rate_pairs]
        soc_starts = [row["metadata"]["SOC_interval"][0] for row in group]
        condition_summaries.append({
            "condition": condition, "n_cells": len(group),
            "charge_rates_C": list(rate_pairs[0][0]),
            "discharge_rates_C": list(rate_pairs[0][1]),
            "recorded_start_soc_min": min(soc_starts),
            "recorded_start_soc_max": max(soc_starts),
        })

    # Fixed random assignment is independent of labels and metadata.  Only the
    # sample counts, already decided above, enter the allocation.
    shuffled = names.copy()
    random.Random(SEED).shuffle(shuffled)
    phases, start = {}, 0
    for phase, size in PHASE_SIZES.items():
        phases[phase] = shuffled[start:start + size]
        start += size
    assert start == len(names) and len(set(shuffled)) == len(names)

    summary = {key: dict(Counter(canonical(row["metadata"][key]) for row in rows))
               for key in ("form_factor", "anode_material", "cathode_material",
                           "nominal_capacity_in_Ah", "depth_of_charge",
                           "depth_of_discharge", "max_voltage_limit_in_V",
                           "min_voltage_limit_in_V")}
    result = {
        "status": "candidate; no model trained; test labels previously used in mixed pilot",
        "purpose": "single-source, variable-protocol EOL prediction at cycle 100",
        "data_root": str(DATA_ROOT),
        "source_pool": str(POOL),
        "source_pool_sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(),
        "inclusion_rule": "ISU_ILCC member of frozen mixed-pilot eligible pool: inherited loader requires EOL>100 and 100 usable prefixes; no error-driven additional filtering",
        "filename_rule": "G is experimental condition; C is cell replicate within condition",
        "n_cells": len(rows), "n_conditions": len(groups),
        "condition_size_counts": dict(sorted(counts.items())),
        "within_condition_metadata_conflicts": inconsistent,
        "within_condition_rate_conflicts": rate_conflicts,
        "condition_summaries": sorted(
            condition_summaries, key=lambda row: int(row["condition"][1:])),
        "metadata_caution": "BatteryLife v4 ISU_ILCC SOC_interval values have a documented correction in v5; do not use v4 SOC values as validated protocol factors.",
        "summary": summary,
        "split": {"seed": SEED, "method": "unstratified cell-level random shuffle",
                  "phase_sizes": PHASE_SIZES, "phases": phases},
        "cells": rows,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "n_cells": len(rows),
                      "n_conditions": len(groups), "condition_size_counts": counts,
                      "n_inconsistent_conditions": len(inconsistent),
                      "n_rate_conflicts": len(rate_conflicts),
                      "phase_sizes": PHASE_SIZES}, default=dict), flush=True)


if __name__ == "__main__":
    main()
