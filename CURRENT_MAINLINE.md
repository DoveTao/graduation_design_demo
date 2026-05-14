# Current Mainline

## Open These First

- pair-level thesis mainline config: `configs/final360i_struct360b_final.yaml`
- pair-level mainline trainer/selector: `tools/final360i_retrain_and_select.py`
- pair-level base model family: `models/struct360b_match_free_coarse_to_fine.py`
- pair-level manifest-native dataset: `datasets/dset2c_manifest_dataset.py`
- trajectory export/eval path: `tools/train360e_sequence_trajectory_export_and_ate_eval.py`
- current core package root: `train360/core/`
- current entrypoint index: `tools/current/README.md`

## Current Best-Known Main Result

Current thesis main model:
- `FINAL360I_struct360b_final_selected`

Current pair-level test reference:
- `signed_tdir_mean = 45.264702006380205`
- `anti_parallel_rate = 0.20169893322797314`
- `tmag_median_ratio = 0.8344251368086006`
- `path_ratio = 0.6403519796204528`

Trajectory caveat:
- use `tools/train360e_sequence_trajectory_export_and_ate_eval.py`
- adjacent-pair composition still shows noticeable drift

## Active Challenger Families

- `STRUCT360C`: `configs/struct360c_rotation_aware_fine_refinement.yaml`, `tools/train_struct360c_rotation_aware_fine_refinement.py`
- `SEQ360A`: `configs/seq360a_sequence_consistency_scale_drift.yaml`, `tools/train_seq360a_sequence_consistency_scale_drift.py`
- `TRAIN360D`: `configs/train360d_observability_kstep_scale.yaml`, `tools/train360d_observability_kstep_scale.py`
- `STRUCT360A`: historical active challenger, still useful for comparison

## Runtime Notes

- current DSET2C mainline uses `datasets/dset2c_manifest_dataset.py`
- sequence experiments use `datasets/dset2c_sequence_clip_dataset.py`
- root-level `config.py`, `model.py`, `interaction.py`, `losses.py`, and `pose_head.py` are now compatibility wrappers
- legacy raw-scan path lives under `train360/legacy/` and is not the current mainline

