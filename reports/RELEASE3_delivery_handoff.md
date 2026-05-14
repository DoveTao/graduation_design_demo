# RELEASE3 Delivery Handoff

## Summary

- release tag target commit: `b52f2f7`
- release tag: `release/20260514-mainline-b52f2f7`
- training executed: `false`
- metrics modified: `false`
- checkpoint weights committed: `false`
- raw data committed: `false`

## Stable Mainline

- main model:
  - `FINAL360I_struct360b_final_selected`
- structure lineage:
  - `STRUCT360B_match_free_coarse_to_fine`
- current mainline overview:
  - `tools/current/show_mainline.py`
- current artifact check:
  - `tools/current/check_current_artifacts.py`

## Retained Deliverables

- core package:
  - `train360/core/`
- retained datasets:
  - `datasets/dset2c_manifest_dataset.py`
  - `datasets/dset2c_sequence_clip_dataset.py`
- retained pair/trajectory tools:
  - `tools/final360i_retrain_and_select.py`
  - `tools/train360e_sequence_trajectory_export_and_ate_eval.py`
  - `tools/train_seq360b_lightweight_scale_smoothing.py`
- thesis materials:
  - `thesis/THESIS361_*`
- key reports:
  - `reports/FINAL360I*`
  - `reports/TRAIN360E*`
  - `reports/SEQ360B*`
  - `reports/BASE360D*`
  - `reports/RESULTS360*`

## Artifact / Noise Check

- tracked `.pt/.pth/.ckpt` files: `none`
- tracked raw data under `data/`: `none`
- tracked `pred_tum.txt` / `adjacent_pair_predictions.jsonl` / `train_log.jsonl`: `none`
- legacy-noise-free repository: `false`

Remaining caveat:

- a small number of legacy compatibility or historical non-weight artifacts remain tracked, notably `dataset_pano_only.py`, `train_mvp.py`, and `checkpoints/S1d5_*`.

These do not change the retained mainline conclusion, but they mean the repository is a stable delivery build rather than a fully minimal archival export.

## Recommended Follow-Up

- if a stricter archival release is needed, do one last non-experimental hygiene pass to remove the remaining tracked legacy compatibility files and `S1d5` checkpoint text/image artifacts
- otherwise, this state is ready for ordinary handoff, thesis submission support, and code browsing
