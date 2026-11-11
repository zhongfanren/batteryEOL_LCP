# Fixed-T CPTransformer audit and minimal experiment design

## Audit of the upstream BatteryLife implementation

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

## Minimal experiment

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
