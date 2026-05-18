# Delivery Handoff

## Stable Release

- release base commit: `b52f2f7`
- release tag: `release/20260514-mainline-b52f2f7`
- current branch at handoff write-up: `main`

## Main Model

- retained pair-level main model:
  - `FINAL360M_fulltrain_struct360b_thesis_main_guarded`
- structure lineage:
  - `STRUCT360B_match_free_coarse_to_fine`
- main config:
  - `configs/final360m_fulltrain_struct360b_thesis_main_guarded.yaml`
- main trainer:
  - `tools/train_final360m_fulltrain_thesis_main.py`
- main model file:
  - `models/struct360b_match_free_coarse_to_fine.py`
- selected checkpoint:
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- old FINAL360I:
  - `subset-trained candidate only`

## Current Entrypoints

- quick mainline overview:
  - `tools/current/show_mainline.py`
- current artifact sanity check:
  - `tools/current/check_current_artifacts.py`
- trajectory refresh:
  - `tools/final360m_trajectory_thesis_refresh.py`
- eval-only trajectory backends:
  - `tools/odom360a_lightweight_trajectory_fusion.py`
  - `tools/odom360b_local_pose_graph_kstep.py`
- retained sequence-scale variant:
  - `tools/train_seq360b_lightweight_scale_smoothing.py`

## Core Code Layout

- reusable core package:
  - `train360/core/`
- retained datasets:
  - `datasets/dset2c_manifest_dataset.py`
  - `datasets/dset2c_sequence_clip_dataset.py`
- retained configs:
  - `configs/final360m_fulltrain_struct360b_thesis_main_guarded.yaml`
  - `configs/seq360b_lightweight_scale_smoothing.yaml`

## Thesis Materials

The thesis-ready experiment material pack is under `thesis/`, especially:

- `thesis/THESIS361_final_experiment_chapter_zh.md`
- `thesis/THESIS361_final_experiment_chapter_en.md`
- `thesis/THESIS361_final_results_tables.md`
- `thesis/THESIS361_final_insert_pack.md`
- `thesis/THESIS361_defense_experiment_script.md`
- `thesis/THESIS361_claims_and_caveats_checklist.md`

## Key Reports

- pair-level main result:
  - `reports/FINAL360M_metrics_test.json`
  - `reports/FINAL360M_final_result_verification.md`
- trajectory evaluation:
  - `reports/FINAL360M_trajectory_metrics_test.json`
  - `reports/FINAL360M_trajectory_level_evaluation.md`
  - `reports/FINAL360M_vs_old_trajectory_backends.md`
- downgraded subset candidate:
  - `reports/FINAL360I_metrics_test.json`
  - `reports/AUDIT_FINAL360I_training_protocol_and_checkpoint_lineage.md`
- retained variant:
  - `reports/SEQ360B_metrics_test.json`
  - `reports/SEQ360B_trajectory_metrics_test.json`
  - `reports/SEQ360B_train_lightweight_scale_smoothing_head.md`
- external baseline:
  - `reports/BASE360D_component_metric_alignment.md`
  - `reports/BASE360D_metrics_test.json`
- consolidated result narrative:
  - `reports/RESULTS360_main_results_table.md`
  - `reports/RESULTS360_experiment_narrative.md`

## Artifact Status

- tracked model-weight files (`.pt/.pth/.ckpt`): `none`
- tracked raw data under `data/`: `none`
- tracked heavy trajectory export files such as `pred_tum.txt` / `adjacent_pair_predictions.jsonl`: `none`

## Caveat

This repository is stable for delivery, but it is not a perfectly minimal archive. A small number of legacy compatibility or historical text/image artifacts still remain tracked, including:

- `dataset_pano_only.py`
- `train_mvp.py`
- `checkpoints/S1d5_*` text/image artifacts

These files are not part of the current recommended mainline workflow and are not required to run the retained `FINAL360M + refreshed trajectory eval + SEQ360B reference` path.

## Recommended First Steps For A Reader

1. Read `README.md`.
2. Run `python tools/current/show_mainline.py`.
3. Run `python tools/current/check_current_artifacts.py`.
4. Use `CURRENT_MAINLINE.md` and `PROJECT_STRUCTURE.md` to navigate the retained code and reports.
