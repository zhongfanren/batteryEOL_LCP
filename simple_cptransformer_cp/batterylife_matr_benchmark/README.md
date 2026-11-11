# MATR-only BatteryLife predictor benchmark

This directory runs the unmodified BatteryLife model classes on the official
MATR 102/34/33 cell split with exactly one sample per cell at T=100. Curves
are prepared by the existing `run_t100.load_cell` adapter, which mirrors the
upstream MATR charge/discharge resampling. It does **not** run the upstream
multi-prefix `run_main.py` training protocol: this environment lacks its
distributed dependencies, and this is a MATR-only comparison rather than a
reproduction of BatteryLife's pooled Li-ion paper table.

The default model list contains the 11 models that completed on the local
8 GB GPU. Autoformer ran out of GPU memory; upstream Transformer requires
decoder inputs that `run_main.py` does not provide; upstream MICN has a
30,000-versus-90,000 feature-dimension mismatch at T=100. Vanilla GRU,
BiGRU, LSTM and BiLSTM process a 30,000-step sequence and were stopped as
impractical (the BatteryLife paper also reports them as NA).

Use `run_matr_models.py` for a one-seed screen. It selects epochs using 12
cells held from the official train set; official validation/test predictions
are optional and exploratory. Use `run_cv.py` for leakage-free five-fold
development evaluation: an inner eight-cell set selects the epoch, then the
model is refit on the outer training fold before predicting the outer fold.

Current primary outputs:

- One-seed screen with upstream per-model script configuration:
  `runs/20261014_222508_198380/results.json`.
- Nested five-fold, model seed 2024:
  `nested_cv_runs/20261014_222028_539246/summary.json` and
  `nested_cv_runs/20261014_223022_029626/summary.json`.
- Nested five-fold, model seeds 42 and 2021:
  `nested_cv_runs/20261014_222658_043861/summary.json` and
  `nested_cv_runs/20261014_222840_123979/summary.json`.

The 11-model seed-2024 screen already saved all checkpoints and fixed-T=100
predictions needed for a MATR-only conformal pilot: 90 official train cells
fit each model, 12 other train cells selected its epoch, 34 official validation
cells are available for calibration, and 33 official test cells for exploratory
evaluation. `evaluate_frozen_split_cp.py` checks these cell partitions and
computes a 90% ordinary Split CP baseline without retraining. Its first output
is `frozen_split_cp/20261021_193212_072306/` with per-model metrics, test
intervals and a SHA-256 manifest of the frozen source artifacts. The test
labels have previously been viewed, so model ranking and interval conclusions
from that test are exploratory. The 34 calibration cells cannot also be used
to tune a supervised localizer without a separate development split.
An updated output, `frozen_split_cp/20261021_194135_514473/`, also includes
two fixed equal-weight ensembles from the existing predictions: all 11 models
and the three top Li-ion CP models from the paper (CPMLP, CPTransformer,
CPGRU). Neither ensemble uses calibration/test labels to fit weights. On the
exploratory 33-cell test, both have 29/33 Split CP coverage (87.9%); the
all-11 mean has MAPE 15.72% and mean interval width 426.7 cycles, while the
paper-CP-top-3 mean has MAPE 15.21% and width 531.2 cycles. These do not beat
the best individual CPLSTM test MAPE of 13.57%; they are descriptive, not a
test-selected model choice.

The earlier `cv_runs/` directory selected epochs on the outer evaluation
fold and must **not** be used for predictor ranking. Earlier `runs/` entries
were used while fixing model-specific configuration; retain them only as
diagnostics. No earlier research experiment directory was changed.
