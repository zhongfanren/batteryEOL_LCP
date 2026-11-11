# MATR 固定 T100 基线：实现与首轮结果

历史批次资料。当前研究见[研究入口](RESEARCH_START_HERE.md)。

## Fixed-T CPTransformer audit and minimal experiment design

### Audit of the upstream BatteryLife implementation

- `Dataset_original` reads each processed cell, takes at most the first
  `early_cycle_threshold` cycles, and resamples charge and discharge separately.
  The resulting tensor is `[early_cycle_threshold, 3,
  charge_discharge_length]`. The three channels are voltage (normalized by the
  per-cycle maximum), current (C-rate), and charge/discharge capacity
  (normalized by nominal capacity).
- The upstream collate function returns `[B, T, 3, 300]` and zeros cycles whose
  mask is zero. A mask value of one means visible. CPTransformer flattens each
  cycle's `3 x 300` curve, embeds it, adds positional embeddings, applies the
  inter-cycle Transformer, flattens all `T` token embeddings, and regresses one
  standardized EOL value. `return_embedding=True` returns this flattened
  penultimate representation.
- Upstream creates many prefixes per cell (`seq_len ... T`), all with the same
  EOL label. Setting both evaluation-cycle bounds to 100 leaves exactly one
  sample per eligible validation/test cell. Its stock training script does not
  set those bounds, and selects checkpoints using validation MAPE while also
  reporting test metrics each epoch.
- The upstream MATR lists contain 102 train, 34 validation, and 33 test cells.
  This experiment treats upstream validation as conformal calibration and never
  uses it for fitting or checkpoint selection.

### Minimal experiment

The standalone entry point reuses the upstream `models/CPTransformer.py`
unchanged and reproduces the MATR-specific preprocessing locally. This avoids
modifying upstream globals and avoids requiring BatteryLife's unrelated
`batteryml` loader dependency.

- Observation time is fixed at `T=100`; every eligible cell contributes exactly
  one sample in every split.
- A seeded cell-level shuffle holds 12 cells out of the upstream training list
  for model validation. The remaining 90 are model training cells.
- EOL standardization is fit on the 90 training labels only. The epoch with the
  smallest model-validation MAPE is retained.
- Split CP uses the finite-sample rank `ceil((n_cal+1)*(1-alpha))` of calibration
  absolute residuals.
- Asymmetric Split CP uses signed residuals and separately calibrates lower and
  upper one-sided errors. The default allocates `alpha/2` to each tail; this
  prevents a large positive residual from mechanically widening the lower side.
- Basic LCP standardizes frozen penultimate embeddings using training cells,
  applies PCA fit on training cells only, uses a fixed median pairwise-distance
  RBF bandwidth, and takes a weighted calibration residual quantile. It falls
  back to Split CP when ESS is below the configured threshold.
- Each invocation creates a new timestamped directory below `runs/`; existing
  runs are never overwritten or removed.

## Experiment log

### 2026-09-03 — Asymmetric Split CP at fixed T=100

Run: `runs/t100_20260903_171917_179085`

#### Change

Kept the CPTransformer predictor unchanged and added an equal-tailed asymmetric
Split CP interval based on signed calibration residuals:

```text
r = EOL - EOL_hat
q_lower = Q_conf(-r, 1-alpha/2)
q_upper = Q_conf( r, 1-alpha/2)
interval = [EOL_hat-q_lower, EOL_hat+q_upper]
```

The implementation records both one-sided finite-sample ranks and writes signed
calibration residuals to `calibration_predictions.csv`.

#### Results

| Method | Coverage | Mean width | Mean interval score |
|---|---:|---:|---:|
| Symmetric Split CP | 90.91% | 626.63 | 1306.29 |
| Equal-tailed asymmetric Split CP | 87.88% | 1075.51 | 1138.11 |

Symmetric radius: `313.31` cycles.

Asymmetric offsets:

- lower: `116.65` cycles;
- upper: `958.86` cycles.

The asymmetric interval fixes the conceptual defect that a large positive
residual also enlarges the lower side. However, it is much wider in this run.
With 34 calibration cells and total alpha 0.1, equal tails use alpha 0.05 on
each side. The conformal rank is 34 on both sides, so each offset is the maximum
one-sided calibration error. `MATR_b1c2` has signed residual `+958.86`, and
`MATR_b3c38` has `+700.28`; these long-life underpredictions dominate the upper
endpoint.

Observed coverage is 29/33 = 87.88%. A single finite test split need not equal
the marginal target exactly. Four misses all occurred below the lower endpoint,
indicating overprediction for those test cells. The mean interval score improved
despite larger width because the interval direction better matches the strongly
right-skewed residual distribution.

#### Interpretation and next restrained step

Do not discard asymmetric intervals, but do not use separate equal-tailed
quantiles as the final method with only 34 calibration cells. A better next
candidate is a pre-shaped asymmetric conformity score: learn/fix the lower-to-
upper width ratio using Train OOF or model-validation residuals, freeze that
ratio, and use one scalar conformal quantile on all 34 calibration cells. This
retains a single alpha=0.1 calibration rank (32) instead of forcing two alpha=0.05
extreme ranks. It is still ordinary split conformal regression and does not
require CQR or LCP.
