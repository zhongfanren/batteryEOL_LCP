# Cross-source protocol audit (2026-10-26)

The BatteryLife `aging condition ID` is **not a cross-source physical protocol identifier**. `aging_conditions.py` assigns IDs in source-specific branches and sometimes gives nearly every cell its own ID. Comparing ID equality across sources therefore cannot decide whether experimental conditions match.

`audit_cross_source.py` read the structured metadata in all 823 eligible Li-ion cells from the fixed T=100 mixed pilot; no file failed. The detailed, reproducible summary is in `cross_source_protocol_inventory.json`.

| Match rule | Cross-source groups | Important example |
| --- | ---: | --- |
| Exact serialized `charge_protocol` and `discharge_protocol` fields | 1 | HNEI 14 + UL-PUR 2, all recorded as one 2C charge stage and one 1C discharge stage |
| Same charge/discharge **rate sequences only** | 2 | Above plus XJTU 2C cells; separately ISU-ILCC 16 + SNL 5 at 0.5C/0.5C |
| Exact above schedule **and** form factor, electrode labels, nominal capacity, DoC/DoD, voltage limits and SOC interval | 0 | None in the eligible pool |

These matches **do not establish identical physical cycling conditions**. For example, the HNEI filename says `0.5-1.5C` and the UL-PUR filename says `0.5-0.5C`, but both stored protocol fields say 2C charge / 1C discharge. The HNEI and UL-PUR cells also have different nominal capacities (2.8 versus 3.4 Ah), cathode labels (LCO versus a NCA composition), voltage limits (3.0–4.3 versus 2.7–4.2 V) and stated temperatures (25 versus 23 °C). The rate-sequence match between ISU-ILCC and SNL is also across very different cell sizes and formats: the inspected ISU example is a 0.25 Ah 502030-size Li-polymer cell, whereas the inspected SNL NMC example is a 3 Ah 18650 cell.

Conversely, HUST and MATR both contain 1.1 Ah cylindrical LFP/graphite cells, but their inspected charge/discharge step sequences and maximum voltage limits differ. This makes them a **near-match in cell family**, not a verified match in full protocol.

## Interpretation for a curated dataset

1. Define the target population first: identical documented cycling steps, or a broader *protocol family* with tolerated differences. These are different claims.
2. Compare chemistry, form factor, capacity, ambient temperature, SOC/voltage window, charge and discharge stages, rest/CV steps, EOL rule, and provenance. Record missing or conflicting fields instead of silently treating them as matches.
3. Verify candidate pairs against the original dataset documentation or raw curves. The processed metadata alone are not reliable enough for strict equivalence because of the observed filename/field conflict.
4. Only then freeze the inclusion rule and cell IDs, split by cell (and inspect batch dependence), and define whether CP coverage targets the pooled protocol family, each source, or new source transfer. Similar protocols do not by themselves prove exchangeability of calibration and test errors.

The previously prepared `tongji_2x2_candidate.json` remains a **within-source** controlled candidate; it is not evidence that cross-source matching is impossible.

A larger within-source, variable-protocol candidate has now been built from the 239 T=100-eligible ISU-ILCC cells. Its selection rule, metadata caveats, and proposed cell-level split are in [`ISU_ILCC_CANDIDATE.md`](ISU_ILCC_CANDIDATE.md).
