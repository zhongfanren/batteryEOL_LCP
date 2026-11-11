"""Read-only audit of cross-source cycling metadata for the 823 T=100 cells.

The output is an inventory, not an assertion that similarly recorded schedules
represent the same physical experiment or an exchangeable CP population.
"""

import gc
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
ROOT = Path(r"C:\BatteryLife_v4")
MANIFEST = MAIN / "mixed_liion_pilot" / "split_manifest.json"
OUT = HERE / "cross_source_protocol_inventory.json"


def data_path(source, name):
    candidates = [ROOT / source / name]
    if source == "MICH":
        candidates.append(ROOT / "total_MICH" / name)
    elif source == "MICH_EXP" and name.startswith("SMICH"):
        candidates.append(ROOT / "MICH_EXP" / name[1:])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"{source}: {name}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows, problems = [], []
    for index, (name, source) in enumerate(manifest["source_of"].items(), 1):
        try:
            with data_path(source, name).open("rb") as stream:
                cell = pickle.load(stream)
            metadata = {key: cell.get(key) for key in (
                "form_factor", "anode_material", "cathode_material",
                "nominal_capacity_in_Ah", "depth_of_charge", "depth_of_discharge",
                "max_voltage_limit_in_V", "min_voltage_limit_in_V",
                "SOC_interval", "charge_protocol", "discharge_protocol")}
            schedule = {key: metadata[key] for key in
                        ("charge_protocol", "discharge_protocol")}
            rates = {key: [step.get("rate_in_C") for step in metadata[key] or []]
                     for key in ("charge_protocol", "discharge_protocol")}
            rows.append({"cell_file": name, "source": source, "schedule": schedule,
                         "metadata": metadata, "rates": rates,
                         "schedule_key": canonical(schedule),
                         "full_metadata_key": canonical(metadata),
                         "rate_key": canonical(rates)})
            del cell
        except Exception as exc:
            problems.append({"cell_file": name, "source": source,
                             "error": f"{type(exc).__name__}: {exc}"})
        if index % 100 == 0:
            print(f"audited={index}/{len(manifest['source_of'])} loaded={len(rows)}", flush=True)
            gc.collect()

    def repeated_groups(key):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        result = []
        for matches in groups.values():
            sources = Counter(row["source"] for row in matches)
            if len(sources) < 2:
                continue
            representative = matches[0]
            result.append({"n_cells": len(matches), "source_counts": dict(sources),
                           "example_cells": [row["cell_file"] for row in matches[:8]],
                           "schedule": representative["schedule"],
                           "rates": representative["rates"],
                           "full_metadata": representative["metadata"] if key == "full_metadata_key" else None})
        return sorted(result, key=lambda group: (-group["n_cells"],
                                                 -len(group["source_counts"])))

    result = {"scope": "T=100 eligible cells in frozen mixed Li-ion pilot pool",
              "data_root": str(ROOT), "n_expected": len(manifest["source_of"]),
              "n_loaded": len(rows), "problems": problems,
              "comparison_warning": "Equal structured charge/discharge fields do not prove equal ambient temperature, cell chemistry, formation, rest steps, EOL definition, measurement or lab batch. Source-specific encoding and filename/metadata conflicts must be checked.",
              "source_counts": dict(Counter(row["source"] for row in rows)),
              "cross_source_exact_schedules": repeated_groups("schedule_key"),
              "cross_source_exact_full_metadata": repeated_groups("full_metadata_key"),
              "cross_source_matching_rate_sequences": repeated_groups("rate_key")}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "n_loaded": len(rows),
                      "n_problems": len(problems),
                      "cross_source_exact_schedules": len(result["cross_source_exact_schedules"]),
                      "cross_source_exact_full_metadata": len(result["cross_source_exact_full_metadata"]),
                      "cross_source_matching_rate_sequences": len(result["cross_source_matching_rate_sequences"])}),
          flush=True)


if __name__ == "__main__":
    main()
