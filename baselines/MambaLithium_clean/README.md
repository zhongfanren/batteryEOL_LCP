# Leakage-free MambaLithium RUL baseline

This directory keeps the official `MambaLithium` checkout unchanged and
imports its Mamba implementation read-only.

Compared with the official `main.py`, this baseline:

- fits `StandardScaler` on training cells only;
- uses a fixed historical window and no future feature rows;
- predicts `log1p(RUL)` without test-lifetime normalization;
- removes the sigmoid output constraint;
- saves predictions, residuals, and Mamba embeddings for later conformal
  calibration.

## Run

From this directory:

```cmd
python train_case.py --case A --seed 1 --epochs 100 --device cpu
```

Outputs are written to `outputs/caseA_seed1/`:

- `metrics.json`
- `predictions.csv`
- `model.pt`
- `training_history.csv`
- `rul_prediction.png`

CaseA and CaseB are development checks only. They contain too few independent
cells for a meaningful conformal calibration experiment. The final LCP
experiment should use cell-level train/validation/calibration/test splits from
BatteryLife.
