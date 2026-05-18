# AUDIT FINAL360I training protocol and checkpoint lineage

## Verdict

- `blocking_issue`: `true`
- recommended_action: `FINAL360I_full_retrain_for_thesis_main_result`
- conclusion: `FINAL360I` cannot currently be treated as a thesis-grade main model result from full-train execution, because the executed training used a train subset cap of `256` rather than the full train manifest.

## Required answers

| field | value | evidence |
| --- | --- | --- |
| FINAL360I training executed | `true` | `reports/FINAL360I_metrics_val.json`, `reports/FINAL360I_metrics_test.json`, `reports/FINAL360I_final_retrain_and_model_selection.md`, `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` |
| FINAL360I train split | `subset` | `checkpoints/FINAL360I_struct360b_final/seed0/config.yaml` has `train_subset_max: 256`; training code applies `_subset_dataset(..., max_count=data_cfg.get("train_subset_max"))` |
| train sample count | `256 effective train samples` | full train manifest has `12154` lines, but executed config capped train subset to `256` |
| val sample count | `5342` | `reports/FINAL360I_metrics_val.json` count `5342`; val manifest line count `5342` |
| test sample count | `5062` | `reports/FINAL360I_metrics_test.json` count `5062`; test manifest line count `5062` |
| epoch count | `5` | `configs/final360i_struct360b_final.yaml`, `checkpoints/FINAL360I_struct360b_final/seed0/config.yaml`, and `train_log.jsonl` epochs `1..5` |
| best epoch | `1` | `reports/FINAL360I_model_selection_table.json` and per-epoch val scores in `train_log.jsonl` |
| optimizer | `AdamW` | `checkpoints/FINAL360I_struct360b_final/seed0/config.yaml` |
| seed | `0` | top-level config seeds `[0]`; executed config `training.seed: 0` |
| init checkpoint | `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt` | top-level config and executed config |
| selected checkpoint | `FINAL360I_struct360b_final_selected` -> `seed0/best_val.pt` | `reports/FINAL360I_metrics_val.json`, `reports/FINAL360I_metrics_test.json`, `reports/FINAL360I_model_selection_table.json` |
| checkpoint path | `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt` | file exists and is referenced consistently in all FINAL360I reports |
| test used for selection | `false` | `reports/FINAL360I_metrics_val.json`, `reports/FINAL360I_metrics_test.json`, `reports/FINAL360I_model_selection_table.json`, selection script |
| `reports/FINAL360I_metrics_test.json` corresponds to selected checkpoint | `true` | JSON field `selected_checkpoint` equals `seed0/best_val.pt`; script evaluates `selected_checkpoint` on test and writes that payload |
| subset=512 or epochs=3 signs exist | `false for subset=512`, `false for epochs=3`, `true for subset=256` | no `512` or `epochs: 3` evidence found; executed config and inherited base config both show `train_subset_max: 256` |
| mixed with ABLDVO2/3 ablation training | `false` | FINAL360I artifacts were created in commit `5933e2c` on 2026-05-13, before ABLVO361/ABLDVO2/ABLDVO3 commits; file names, checkpoint root, and reports are distinct |
| thesis main model usable | `false` | executed run was real, but it was not full-train |

## Evidence chain

### 1. Config-level protocol

- `configs/final360i_struct360b_final.yaml` defines:
  - train manifest: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`
  - val manifest: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`
  - test manifest: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
  - `training.epochs: 5`
  - `training.seeds: [0]`
  - `training.run_test_during_seed_training: false`
  - init checkpoint: `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`
- But `configs/final360i_struct360b_final.yaml` inherits `inputs.base_config: configs/struct360b_match_free_coarse_to_fine.yaml`.
- That base config contains `data.train_subset_max: 256`.
- The executed seed config at `checkpoints/FINAL360I_struct360b_final/seed0/config.yaml` also contains `data.train_subset_max: 256`.

### 2. Training code-level protocol

- `tools/final360i_retrain_and_select.py` builds the seed config by copying the base config and only overriding selected fields such as:
  - `training.epochs`
  - `training.seed`
  - output paths
  - eval-time `run_test`
- It does not clear or override `data.train_subset_max`.
- `tools/train_struct360b_match_free_coarse_to_fine.py` contains:
  - `_subset_dataset(dataset, max_count, seed)`
  - `train_subset, subset_info = _subset_dataset(ds_train, max_count=data_cfg.get("train_subset_max"), seed=int(train_cfg["seed"]))`
- Therefore the executed FINAL360I training did not use the full `12154`-sample train manifest; it used a 256-sample subset.

### 3. Execution evidence

- `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt` exists.
- `checkpoints/FINAL360I_struct360b_final/seed0/final.pt` exists.
- `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` records epochs `1` through `5`.
- Best validation score was at epoch `1`:
  - epoch 1: `149.8338157649262`
  - epoch 2: `153.22605494308124`
  - epoch 3: `167.80120341825932`
  - epoch 4: `160.00892377531267`
  - epoch 5: `160.52957741302149`
- `reports/FINAL360I_model_selection_table.json` records `best_epoch: 1` and selected checkpoint `seed0/best_val.pt`.

### 4. Metrics lineage

- `reports/FINAL360I_metrics_val.json`:
  - `final_retrain_executed: true`
  - `selected_model_name: FINAL360I_struct360b_final_selected`
  - `selected_checkpoint: /home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
  - `test_used_for_selection: false`
- `reports/FINAL360I_metrics_test.json`:
  - same `selected_checkpoint`
  - same `selected_model_name`
  - `test_used_for_selection: false`
- `reports/FINAL360I_model_selection_table.json`:
  - `selected_seed: 0`
  - `selected_seed_checkpoint: .../seed0/best_val.pt`
  - `final_checkpoint: .../seed0/best_val.pt`
- The training script explicitly evaluates `selected_checkpoint` on both val and test after seed selection and writes those results to the FINAL360I JSON reports.
- So `reports/FINAL360I_metrics_test.json` does correspond to the selected checkpoint.

### 5. Sample counts

- Train manifest line count: `12154`
- Val manifest line count: `5342`
- Test manifest line count: `5062`
- Effective executed training count: `256` because of `train_subset_max: 256`

### 6. Git/report chronology

- commit `5933e2c` dated `2026-05-13 16:08:33 +0800`: `FINAL360I: add final STRUCT360B retrain and model selection`
- commit `270073d` dated `2026-05-14 19:14:04 +0800`: `MAINT15: reduce stale comparison labels after cleanup`
- `reports/FINAL360I_final_retrain_and_model_selection.md` states `final retrain executed: true` and recommends `FINAL360I_struct360b_final_selected`, but it does not disclose that the executed training inherited `train_subset_max: 256`.

## Confusion audit against ablations

- No evidence shows `FINAL360I` directly reusing ABLDVO2 or ABLDVO3 report names or checkpoint roots.
- ABLVO361/ABLDVO2/ABLDVO3 reports live under separate filenames and separate checkpoint directories.
- The current working branch is unrelated to the original FINAL360I execution branch, but the FINAL360I artifact lineage itself predates those ablation reports.
- So this is not primarily an ABLDVO2/3 mix-up.
- The real problem is that the FINAL360I run itself was a subset-limited retrain inherited from the STRUCT360B base config.

## Final judgment

- `FINAL360I` was really executed.
- The selected checkpoint and reported test metrics are internally consistent.
- The run was not a `3 epoch` ablation.
- The run was also not a full-train main-model retrain, because the effective train split was a `256`-sample subset.
- This is a blocking issue for using `FINAL360I_struct360b_final_selected` as the thesis main model result.

