# BatteryLife MATR + Mamba + LCP

This is the minimal main experiment:

```text
first 50 cycles of one MATR cell
    -> Mamba lifetime prediction and embedding
    -> Split conformal prediction / localized conformal prediction
```

Each cell contributes one sample, and train/validation/calibration/test cells
are disjoint. The official BatteryLife test split is retained. Its validation
split is deterministically divided between model validation and conformal
calibration.

## Data

The processed BatteryLife dataset is gated and MATR is about 4.9 GB. First
accept the conditions at:

https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed

Then install the downloader and set your token:

```cmd
pip install huggingface_hub
set HF_TOKEN=hf_your_token
python download_matr.py --output-dir D:\BatteryLife_Processed
```

The expected structure is:

```text
D:\BatteryLife_Processed\
├── MATR\
│   ├── MATR_b1c0.pkl
│   └── ...
└── Life labels\
    └── MATR_labels.json
```

Do not commit the token or the downloaded pickle files.

## Formal run

```cmd
python run_experiment.py ^
  --data-root D:\BatteryLife_Processed ^
  --early-cycles 50 ^
  --alpha 0.1 ^
  --seed 42 ^
  --device auto
```

Results are saved under `outputs/matr_seed42/`:

- `results.json`
- `test_predictions.csv`
- `training_history.csv`
- `model.pt`

## Smoke test without the gated dataset

```cmd
python make_smoke_data.py --output-dir smoke_data --cells 80
python run_experiment.py --data-root smoke_data --split-mode random ^
  --epochs 5 --patience 5 --seed 42 --device cpu ^
  --output-dir smoke_outputs
```

Smoke metrics are only an integration check and must never be reported as
BatteryLife results.

## Landmark RUL experiment (current main line)

The leakage-free per-cycle RUL experiment is implemented separately so that
the historical fixed-50-cycle lifetime experiment remains reproducible:

```cmd
python run_rul_landmarks.py ^
  --data-root D:\BatteryLife_Processed ^
  --landmarks 50 100 150 200 ^
  --epochs 100 ^
  --patience 20 ^
  --hidden 16 ^
  --layers 2 ^
  --lr 0.01 ^
  --alpha 0.1 ^
  --lcp-k 15 ^
  --min-ess 10 ^
  --seed 42 ^
  --device cuda
```

This run keeps the official 33-cell test set, uses all 34 official validation
cells for conformal calibration, and splits the official training set into 90
model-training and 12 model-validation cells.  Results are written to
`outputs/rul_landmark_seed42/` without overwriting the old experiment.

## CPTransformer + learnable-kernel experiment

The current predictor comparison adapts BatteryLife's CPTransformer components
to causal per-cycle RUL prediction and learns kernel parameters exclusively from
cell-level out-of-fold training residuals. See
`../CPTRANSFORMER_KERNEL_DESIGN.md` for the frozen design.

```cmd
D:\Anaconda3\envs\my_torch\python.exe run_cptransformer_kernels.py ^
  --data-root D:\BatteryLife_Processed ^
  --epochs 100 --patience 20 ^
  --oof-folds 3 --oof-epochs 40 --kernel-steps 400 ^
  --d-model 64 --d-ff 128 --n-heads 4 ^
  --intra-layers 1 --transformer-layers 2 ^
  --alpha 0.1 --min-ess 10 --seed 42 --device cuda
```

Replace `42` with `2021` and `2024` for the other formal runs. Aggregate all
completed runs with:

```cmd
D:\Anaconda3\envs\my_torch\python.exe analyze_cptransformer_kernels.py
```
