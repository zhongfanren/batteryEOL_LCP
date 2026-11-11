# Experiment log

## 2026-09-03 — Asymmetric Split CP at fixed T=100

Run: `runs/t100_20260903_171917_179085`

### Change

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

### Results

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

### Interpretation and next restrained step

Do not discard asymmetric intervals, but do not use separate equal-tailed
quantiles as the final method with only 34 calibration cells. A better next
candidate is a pre-shaped asymmetric conformity score: learn/fix the lower-to-
upper width ratio using Train OOF or model-validation residuals, freeze that
ratio, and use one scalar conformal quantile on all 34 calibration cells. This
retains a single alpha=0.1 calibration rank (32) instead of forcing two alpha=0.05
extreme ranks. It is still ordinary split conformal regression and does not
require CQR or LCP.

