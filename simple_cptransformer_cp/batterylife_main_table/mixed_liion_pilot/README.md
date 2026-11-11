# Mixed Li-ion CPMLP pilot, 2026-10-21

## Fixed protocol

- Population: 823 labeled MIX_large Li-ion cells with 100 usable cycles, identified from the prior main-table loader inventory.
- Cell-level unstratified random partition, seed `20261021`: 450 train, 60 development, 180 calibration, 133 test. The exact names and source mapping were written to `split_manifest.json` before fitting. All four sets are disjoint.
- CPMLP, selected as the best point predictor in the exploratory local main-table reproduction. Train on prefixes 1–100 of train cells. Select checkpoint using development prefix MAPE. Calibration and test each use only prefix 100, one observation per cell.
- Nominal 90% absolute-residual split CP: calibration score `abs(EOL - prediction)`, radius equal to sorted calibration score of rank `ceil((180+1)*0.9)=163`. Intervals are `[max(0, prediction-radius), prediction+radius]` in cycles.
- The local comparator uses Gaussian weights on the frozen point prediction with bandwidth equal to the standard deviation of calibration predictions. Its weighted score quantile is exploratory and has **no finite-sample coverage guarantee**. Bandwidth and method did not use test labels.

## Observed result

| Quantity | Result |
| --- | ---: |
| Best development epoch | 11 (stopped after epoch 16) |
| Calibration MAPE at T=100 | 20.12% |
| Test MAPE at T=100 | 24.26% |
| Split CP radius | 332.75 cycles |
| Split CP test coverage | 113/133 = 84.96% |
| Split CP mean interval width | 658.42 cycles |
| Exploratory local coverage | 116/133 = 87.22% |
| Exploratory local mean width | 659.67 cycles |

The pooled split CP misses are concentrated in HUST (8 of 14 test cells uncovered) and ISU_ILCC (10 of 43 uncovered). MATR has 21/21 covered. The source-level sample sizes are small; these are descriptions of this split, not source-conditional coverage guarantees. The local comparator did not improve pooled coverage to 90% or reduce pooled mean width.

The observed 84.96% is one finite-test realization. Split conformal's standard claim concerns marginal coverage of a randomly sampled future test cell under exchangeability, rather than a deterministic guarantee that each held-out test set reaches 90%. The randomized partition can justify a finite-pool randomization interpretation for a randomly assigned held-out cell if all design choices were fixed independently of held-out labels. Prospective use additionally requires future calibration/test exchangeability from the same cell mixture. Cell-level disjointness alone does not establish that premise; changes in source proportions or test protocols can break it. No source-conditional or shifted-domain coverage claim follows from pooled split CP.

This is exploratory: CPMLP was chosen using earlier reproduction results on the same underlying cells, so the new test is not a pristine confirmatory test for model-family selection. The 24.26% T=100 test MAPE is also not directly comparable to the earlier main-table MAPE, which used another split and all 1–100 prefixes.

## Reproduce / inspect

Run with `D:\Anaconda3\envs\my_torch\python.exe` from `D:\Battery-LCP`:

```powershell
& D:\Anaconda3\envs\my_torch\python.exe simple_cptransformer_cp/batterylife_main_table/run_mixed_liion_pilot.py --prepare
& D:\Anaconda3\envs\my_torch\python.exe -u simple_cptransformer_cp/batterylife_main_table/run_mixed_liion_pilot.py
```

The completed directory intentionally refuses to overwrite its frozen manifest or run. Inspect `results.json` for metrics, `test_intervals.csv` for each test cell, `calibration_predictions.csv` for calibration scores, `config.json` and `best_model.pt` for the predictor, and `run.log` for training progress. `results.json` and `test_intervals.csv` were independently checked against the manifest and calibration score rank after the run.
