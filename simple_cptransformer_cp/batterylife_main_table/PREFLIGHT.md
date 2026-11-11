# BatteryLife Li-ion main-table reproduction: preflight

Date: 2026-10-20. This is the start of experiment 1 in `CONTEXT_HANDOFF.md`, before any new download or training. It does not report a reproduced model result.

## Protocol to reproduce

- Upstream `MIX_large` uses 13 Li-ion components and disjoint **cell** lists: 515 train, 165 validation, 163 test. MATR contributes 102/34/33 cells. `UL_PUR` contributes only two training cells. The upstream loader makes prefixes `i=1..100` per eligible cell (`seq_len=1`, `early_cycle_threshold=100`) with the same EOL label, so metrics in the main table are over prefix samples, not one sample per cell.
- The upstream loader resamples charge/discharge curves to 300 points per cycle, normalizes voltage by the maximum voltage and current/capacity by nominal capacity, and fits `StandardScaler` to unique **training-cell** EOL values. The entry point uses MSE on scaled labels unless a model script changes it, selects the checkpoint by validation MAPE with patience 5 (after at least 5 epochs), and reports `sklearn` MAPE as a fraction plus 15%-Acc from `abs(pred-EOL)/EOL <= 0.15` as a percentage. The repository README table displays 15%-Acc as a fraction, so presentation units must be converted before comparison.
- Use one model and one seed first, then extend to the 11 models already run in the MATR-only screen. DLinear seed 2021 is a low-memory pilot candidate because its upstream shell script explicitly sets `MIX_large`; this is a planning choice, not a result. The upstream CPMLP shell script currently sets `dataset=CALB`, so running it unchanged would **not** reproduce the Li-ion column. The selected-hyperparameters document says the reported results used BatteryLife v4 and two GPUs; the local checkout is a shallow 2026-07-06 commit (`9572e47`) and its CPTransformer/CPLSTM script values differ from the v4 selected-hyperparameters table. Pin code, data, and exact per-model configuration before treating any result as strict replication. Otherwise label it an approximate reproduction with listed differences.

## Local data and dependency inventory

Run `python simple_cptransformer_cp/batterylife_main_table/audit_mix_large.py` from the workspace root to repeat the read-only inventory. Current `D:\BatteryLife_Processed` has only HUST and MATR cells and their labels. All local HUST/MATR files required by the lists are present. The original train/validation/test lists have zero cell-ID overlap.

| Phase | Required cells | Available files and labels | Missing |
| --- | ---: | ---: | ---: |
| Train | 515 | 149 | 366 |
| Validation | 165 | 49 | 116 |
| Test | 163 | 48 | 115 |

Missing components: `UL_PUR` 2 cells, `RWTH` 48, `Stanford` 41, `Tongji` 108, `ISU_ILCC` 240, `CALCE` 13, `HNEI` 14, `SNL` 50, `MICH` 40, `MICH_EXP` 18, and `XJTU` 23. Their label JSON files are also absent. The test seen/unseen map **is** available in `third_party/BatteryLife/dataset/seen_unseen_labels/cal_for_test.json` and covers all 163 test IDs. The local HUST/MATR archive versions cannot be proved from extracted files alone.

The paper-era [Zenodo v4 record](https://zenodo.org/records/14969822) lists these 11 missing archives at about **16.2 GB compressed**, plus `Life labels.zip` at 11.3 kB; extracted size and peak disk use remain unverified. The [newer v11 record](https://zenodo.org/records/19688272) has different file checksums and documents changes since v4, so mixing versions must be recorded. The data root is outside the writable workspace, and drive D currently has about 38.1 GB free. Downloading and extracting all archives at once would risk exhausting the drive. Use staged acquisition with archive checksums and re-run the audit after each stage.

The `my_torch` environment has Python 3.9, PyTorch 2.7.1+cu128 and an RTX 4070 Laptop GPU. It lacks `accelerate`, `deepspeed`, `peft`, `evaluate`, `wandb`, `denseweight`, `BatteryML`, `datasets`, and `reformer_pytorch`, all imported directly or needed by the official entry point. The [upstream requirements](../../third_party/BatteryLife/requirements.txt) target different package versions, including PyTorch 2.4.1 and DeepSpeed 0.15.0. The upstream scripts request two GPUs, 32 data-loader workers, and hard-coded Linux checkpoint/data paths. A Windows single-GPU run therefore needs documented adaptations even with complete data. `run_main.py` also builds the full train, validation, and test datasets before training, increasing memory demand.

## Next execution gate

1. Choose and record a code/data pairing for the comparison. For a close paper comparison, use v4 archives and the matching code/configuration where recoverable; verify local HUST/MATR provenance or reacquire those archives. Keep any updated-data run explicitly separate.
2. Stage the missing files in a new workspace data area with checksum verification and sufficient free space, and install entry-point dependencies in an isolated environment. Do not alter existing experiment directories or the pre-existing edit in the upstream checkout.
3. Run one DLinear seed 2021 pilot with cell lists, 1..100 prefixes, scaler, validation-MAPE selection, and metric units checked. Save its exact configuration, code/data hashes, cell/prefix counts, and predictions. Only then expand to the remaining feasible models.
4. Experiment 2 (MATR-held-out predictor and fixed T=100 CP/LCP) starts after the predictor and non-MATR data protocol are fixed. Do not reuse weights trained on `MIX_large`.

## Progress on 2026-10-20

- The official v4 `Life labels.zip` and `UL_PUR.zip` were downloaded and passed their published MD5 checksums. All 77 HUST and 169 MATR labels in the local data matched v4 exactly. This does **not** establish that their curve files are v4. The v4 labels and UL_PUR cells were extracted to `C:\BatteryLife_v4`; HUST/MATR and the upstream seen/unseen map are linked without changing their source files.
- The v4 split lists include six `MICH_EXP` cells without EOL labels (five train, one test) and another 14 labeled cells with EOL <=100 (7/3/4 by train/validation/test). The upstream loader intentionally skips these from prefix generation; they must not be fabricated or counted as missing downloads. The preflight still requires their source files because the loader opens a file before checking its label.
- The largest v4 archive, `ISU_ILCC.zip`, has 11.64 GB compressed and **30.31 GB extracted** according to its ZIP directory. The required 11 archives total about 16.2 GB compressed. Because D has only about 38 GB free, new data are staged on C. `stage_v4.py` verifies Zenodo MD5 values before extraction. Zenodo transfer is slow; the full data are still downloading as of this note.
- A single-GPU adapter in `run_single_gpu.py` uses the original loader's prefix generation and original DLinear model class. It omits only three imports unused by this unweighted path (`BatteryData`, `accelerate`, `DenseWeight`), changes the launcher to one GPU with gradient accumulation, and sets Windows-safe `num_workers=0`. A one-cell smoke test produced 100 prefixes and a successful model forward pass.
- A **partial-data pipeline trial** succeeded with 151 train, 49 validation, and 48 test cells (15,100 / 4,900 / 4,800 prefix samples). Seed 2021, best epoch 4: validation MAPE 0.22309 and 15%-Acc 0.430; test MAPE 0.21401 and 15%-Acc 0.455. Artifacts: `runs_single_gpu/20261020_213115_981962/`. The official MATR test has been viewed before; these test numbers are exploratory. This trial is **not a Li-ion main-table reproduction** because 595 of the 843 required `MIX_large` cells were unavailable at the time of training.
- A hidden `continue_when_ready.py` helper now checks for all 843 source files; once staging finishes it will run DLinear seed 2021 on the full listed split. Its state is `pipeline_state.json`, and the eventual training output is `full_split_console.log` plus a new `runs_single_gpu/` directory. The Zenodo transfer is still active and is currently the limiting step.

## Download route update, 2026-10-20 22:59

- The Zenodo v4 transfer was too slow. It was stopped without deleting its partial archives. Five complete archives (`CALCE`, `HNEI`, `SNL`, `MICH`, `MICH_EXP`) passed their published MD5 checksums and were extracted into `C:\BatteryLife_v4`.
- The remaining 460 required source files (43.58 GB uncompressed) are downloading with eight workers from the [official BatteryLife Hugging Face repository](https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed), pinned to commit `f7a45ed5598569d3d0fdb3019bc81dd92d756c28` of 2025-03-05. One v4 ISU label JSON, one UL_PUR cell, and one HNEI cell matched the corresponding Zenodo v4 contents byte for byte. Every fetched file is checked against its published Hugging Face LFS SHA-256 before its dataset directory is activated. This is strong evidence for a v4-equivalent snapshot, but it is not a proof that every file is identical to the Zenodo ZIPs.
- Live progress is in `hf_download_state.json` and `hf_stdout.log`; a visible PowerShell monitor displays it. The existing watcher still waits on `C:\BatteryLife_v4` and will start full-split DLinear once all listed files are present. The old `stage_stdout.log` no longer receives download updates.

## Execution update, 2026-10-21

- All 843 `MIX_large` source files are now present (515/165/163 by train/validation/test). The expected EOL-label counts are 510/165/162 because six `MICH_EXP` cells have no EOL label in v4. `hf_download_state.json` is `complete` for 460/460 additional source files.
- Five Tongji files in the pinned Hugging Face snapshot failed with HTTP 416 during resume. For three, the Zenodo v4 bytes match the Hugging Face SHA-256 but the Hugging Face download cache was truncated; for two (`1--3` and `1--7`), the Zenodo v4 bytes have the published file size but differ from the Hugging Face LFS SHA-256. Those five files were fetched individually from the official Zenodo v4 `Tongji.zip` by HTTP Range and verified with the ZIP member CRC32; their SHA-256 and sizes are recorded in `zenodo_tongji_repair_manifest.json`. The other 455 files passed Hugging Face LFS SHA-256 verification. Complete small Zenodo ZIPs (`CALCE`, `HNEI`, `SNL`, `MICH`, `MICH_EXP`) passed published MD5 checks before extraction.
- The original loader expects a combined `total_MICH` directory, so its 40 MICH and 18 MICH_EXP `.pkl` files were copied there, matching `merge_MICH` behavior in upstream `data_loader.py`.
- The full-split, one-GPU DLinear seed 2021 main-table run started at 10:58; status is `pipeline_state.json`, output is `full_split_console.log`, and artifacts go to `runs_single_gpu/`. It remains an **approximate** reproduction because of single-GPU adaptation and unproven HUST/MATR curve provenance.
- A separate `continue_no_matr.py` process waits for the main-table run to succeed, then trains DLinear from scratch with MATR removed from training, validation, scaler fitting and source test. It saves a separate checkpoint and fixed `T=100` predictions for all 169 MATR cells under `runs_no_matr/`. Status is `heldout_pipeline_state.json`. This run does not reuse the main-table weights.

## Loader recovery, 2026-10-21 11:06

- The first full-split attempt reached the MICH_EXP training cells and stopped on an upstream loader bug: for a cell with no EOL label, `read_cell_df` returns five values while `read_samples_from_one_cell` unpacks six. Its console log is preserved as `full_split_failed_loader_20261021.log`.
- The single-GPU adapter now appends the missing sixth `None` only for that unlabeled-cell return. A direct test using one of the six unlabeled v4 MICH_EXP cells confirmed that it is skipped. The upstream source file remains intact.
- Full-split DLinear restarted at 11:06. The held-out DLinear watcher was restarted and is waiting for the main-table run to finish. Both state files are updated atomically: `pipeline_state.json` and `heldout_pipeline_state.json`.
- The 11:06 attempt read all 515 training files, then the original loader's `np.isnan` check tried to materialize the 100 repeated prefixes per cell into one huge array. Available RAM fell to 244 MB before any epoch; the process was stopped deliberately. Its log is preserved as `full_split_stopped_memory_20261021.log`.
- The adapter now runs the same NaN predicate on each distinct in-memory curve array once, avoiding the temporary expansion without changing samples or labels. It passed a normal/NaN equivalence check and the full-split run restarted at 11:15. The no-MATR watcher was restarted with it.

## First full-split DLinear result, 2026-10-21 11:28

- The single-GPU, seed-2021 DLinear run completed on the original `MIX_large` cell lists. Train: 503 usable cells / 50,300 prefix samples; validation: 162 / 16,200; test: 158 / 15,800. The omitted cells lack an EOL label or have EOL at or before cycle 100, as in the original loader.
- Best checkpoint: epoch 2 by validation MAPE 0.60635. Test MAPE 0.57207; test 15%-Acc 0.19551. Artifacts, predictions, config, and checkpoint: `runs_single_gpu/20261021_112227_585339/`.
- The upstream README DLinear Li-ion table reports MAPE 0.586±0.028 and 15%-Acc 0.275±0.017. This one-seed, single-GPU result has similar MAPE but materially lower 15%-Acc. It is an approximate reproduction, not an exact match: upstream uses two GPUs, local HUST/MATR curve file provenance is unproven, and five Tongji files came from Zenodo v4 rather than the pinned HF snapshot. The official test split has been viewed, so test results are exploratory.
- The independent, non-MATR DLinear training started at 11:28. It will save one T=100 prediction for each of the 169 MATR cells plus a SHA-256 freeze manifest. Its checkpoint is selected only using non-MATR validation.

## MATR-held-out DLinear result and next model, 2026-10-21 11:41

- The DLinear model trained without MATR completed. Best epoch 3 selected on non-MATR validation: MAPE 0.67515; non-MATR source test MAPE 0.68377. All 169 MATR T=100 predictions and the source-only checkpoint are frozen in `runs_no_matr/20261021_113413_087865/`, with SHA-256 hashes in `frozen_predictor.json`.
- Target MATR performance is poor: RMSE 5,515.80 cycles, MAPE 7.40750, 15%-Acc 0.01775. Predictions range from about -17,551 to +7,378 cycles while actual EOL ranges from 147 to 2,237. This is a severe cross-domain extrapolation failure for this model; it is not a useful point predictor for the planned LCP experiment. No MATR labels or metrics were used for training, scaler fitting, early stopping, or hyperparameter tuning.
- The next predeclared architecture is CPMLP, using the v4 Li-ion selected-hyperparameter table: seed 2021, per-GPU batch 16, effective batch 32 via accumulation 2, `d_model=32`, `d_ff=256`, `e_layers=12`, `d_layers=7`, dropout 0, learning rate 5e-5. A one-cell CUDA smoke test passed. `continue_cpmlp.py` started the main-table run at 11:41 and will then train CPMLP again with MATR excluded. Status: `cpmlp_pipeline_state.json`; logs: `cpmlp_main_console.log`, `cpmlp_no_matr_console.log`.

## CPMLP main-table result, 2026-10-21 12:08

- Full `MIX_large` CPMLP seed 2021 completed with best epoch 15 by validation MAPE 0.25426. Test MAPE 0.16882 and 15%-Acc 0.63835 over 15,800 prefixes. Run artifacts: `runs_single_gpu/20261021_114809_194841/`.
- Upstream Li-ion CPMLP reports MAPE 0.179±0.003 and 15%-Acc 0.620±0.004 over three seeds. This single-seed, single-GPU result is numerically close, while the previously documented code/data/provenance differences still make it an approximate reproduction. The test is exploratory because it was viewed during development.
- The separate CPMLP training with MATR excluded started automatically at 12:08. It uses non-MATR train/validation for checkpoint selection and will export all 169 MATR T=100 predictions and a freeze manifest.

## MATR-held-out CPMLP result, 2026-10-21 12:24

- CPMLP trained from scratch without MATR, selected epoch 4 on 128 non-MATR validation cells (MAPE 0.31557). Non-MATR source test MAPE is 0.22069 and 15%-Acc is 0.48520. Run artifacts: `runs_no_matr/20261021_121422_040624/`. Its `frozen_predictor.json` verified that the source train/validation/test cell lists exclude all 169 MATR cells, each MATR cell has exactly one T=100 prediction, and the checkpoint/config/predictions are fingerprinted with SHA-256.
- On all 169 MATR cells, exploratory T=100 RMSE is 631.22 cycles, MAE 498.63, MAPE 0.71128, and 15%-Acc 0.08876. The original MATR 33-cell test subset has RMSE 659.76, MAPE 0.6454, and 15%-Acc 0.0909. This is substantially better than the held-out DLinear but still poor transfer. No MATR metric was used to select CPMLP's epoch or hyperparameters.
- A constant baseline fitted only on the 401 non-MATR training-cell labels provides context: source-train mean EOL 907.5 cycles gives MATR RMSE 354.38, MAPE 0.4415, and 15%-Acc 0.3314; source-train median 576 cycles gives RMSE 397.84, MAPE 0.3122, and 15%-Acc 0.1775. Both beat held-out CPMLP on MATR, so this predictor should not yet be treated as a useful LCP point baseline.
- `matr_cal_test_split.json` records the initial cell-level plan: original MATR train+validation, 136 cells for calibration; original test, 33 cells for exploratory testing. Changing this ratio does not require retraining either frozen source-trained predictor; it requires rebuilding calibration and test outputs, and refitting any supervised MATR localizer that used the changed data.
- At the same fixed T=100 input, held-out CPMLP's non-MATR source test MAPE is 0.2085 over 125 cells, compared with MATR MAPE 0.7113 over 169 cells. MATR prediction/EOL Pearson correlation is -0.4117. The fixed-T comparison supports a genuine transfer failure rather than merely a difference between multi-prefix and T=100 evaluation.

## Remaining nine main-table models, 2026-10-21 12:56

- At the user's request, `run_single_gpu.py` now supports the other nine model classes from the agreed list. Each passed a one-cell CUDA forward/backward smoke run at its actual single-GPU microbatch size. The scripts provide their model configuration, with the published v4 Li-ion selected hyperparameters used for CPTransformer, CPGRU and CPLSTM where the scripts differ; the exact values are saved in each run's `config.json`.
- Single-GPU gradient accumulation preserves the original two-GPU effective batch size. CPTransformer uses microbatch 64/accumulation 4, and CNN uses microbatch 16/accumulation 2, both checked by CUDA smoke runs. This remains an approximate reproduction because hardware and launcher differ from the original distributed run.
- `run_remaining_models.py` started a serial queue at 12:56. It attempts MLP, iTransformer, PatchTST, CNN, CPBiGRU, CPBiLSTM, CPGRU, CPLSTM and CPTransformer; failures are recorded and do not stop the remaining attempts. Live progress is `remaining_models_state.json`, per-model consoles are under `remaining_models_logs/`, and new artifacts go into distinct timestamped `runs_single_gpu/` folders. A visible PowerShell monitor displays queue status.
- For later queue members, `run_single_gpu.py` serializes the already constructed original-loader train/validation/test datasets into `processed_dataset_cache/`, keyed by data root, exact split file lists, loader/split/life-class hashes, preprocessing lengths and scaler values. A one-cell save/reload comparison matched curves, masks and labels at three prefix positions. The active MLP process started before this cache code was added and remains unaffected; subsequent model processes will use the cache.

## Completed 11-model main-table screen, 2026-10-21 18:57

The nine-model queue finished with nine successful exits and no failures. Combined with the earlier DLinear and CPMLP runs, all 11 original model classes have a seed-2021, single-GPU approximate `MIX_large` result. Values below are fractions, measured over 15,800 test prefixes; checkpoints were selected by validation MAPE. The original README Li-ion MAPE is a three-seed mean, so differences are descriptive, not a controlled significance test.

| Model | Best epoch | Local test MAPE | Local test 15%-Acc | README Li-ion MAPE |
|---|---:|---:|---:|---:|
| DLinear | 2 | 0.5721 | 0.1955 | 0.586 |
| MLP | 10 | 0.4405 | 0.3170 | 0.233 |
| CPMLP | 15 | 0.1688 | 0.6384 | 0.179 |
| PatchTST | 12 | 0.3551 | 0.3172 | 0.288 |
| iTransformer | 5 | 0.2784 | 0.4270 | 0.209 |
| CPTransformer | 6 | 0.2278 | 0.4259 | 0.184 |
| CNN | 12 | 0.3656 | 0.3445 | 0.337 |
| CPGRU | 26 | 0.1797 | 0.5952 | 0.189 |
| CPBiGRU | 27 | 0.1924 | 0.5920 | 0.190 |
| CPLSTM | 11 | 0.1966 | 0.5946 | 0.196 |
| CPBiLSTM | 8 | 0.2192 | 0.5432 | 0.191 |

`remaining_models_state.json` maps each of the nine queue members to its checkpoint, config, predictions, and metrics directory. DLinear and CPMLP directories are recorded above. Some architectures are close to the README Li-ion column; MLP, PatchTST, iTransformer and CPTransformer are farther away. One seed, the single-GPU training adaptation, and unproven local HUST/MATR curve provenance preclude a strict paper reproduction claim. All these `MIX_large` checkpoints were trained with MATR included and must not be reused for the MATR-held-out predictor experiment. The only independently MATR-excluded checkpoints so far are DLinear and CPMLP; both transfer poorly to MATR at fixed T=100.
