# DISK FINAL360M safe cleanup execution

## Summary

- `safe checkpoint cleanup executed`: `true`
- `FINAL360M process alive before`: `true`
- `FINAL360M process alive after`: `true`
- `disk free before`: `4.6G`
- `disk free after`: `7.4G`
- `deleted total size`: `~2.8G`
- `protected checkpoint deleted`: `false`
- `current FINAL360M checkpoint deleted`: `false`
- `metrics modified`: `false`
- `nan_in_train_loss`: `true`
- `nan_in_grad`: `false` (no direct grad metric was logged; no explicit grad-norm field available)
- `nan_only_in_eval_placeholder`: `false`
- `training_valid_so_far`: `false`
- `recommended action`: `stop_and_restart_required`

## Phase A

- process alive before cleanup: `true`
- PID/etime/cmd before cleanup: `15106 40:45 /home/dovetao/miniconda3/envs/pytorch/bin/python tools/train_final360m_fulltrain_thesis_main.py configs/final360m_fulltrain_struct360b_thesis_main.yaml`
- current visible epoch before cleanup:
  - main log still only shows `epoch 1/20 train start`
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/train_log.jsonl` shows `epoch 1-3`
- current disk free before cleanup: `4.6G`

## NaN Diagnosis

### Findings

- `train_loss_total` is `NaN` in epochs `1`, `2`, and `3`
- `train_loss_rot` is `NaN` in epochs `1`, `2`, and `3`
- `train_loss_tdir` is `NaN` in epochs `1`, `2`, and `3`
- `train_loss_tmag` is `NaN` in epochs `1`, `2`, and `3`
- `train_loss_scale_stability` is `NaN` in epochs `1`, `2`, and `3`
- `mini_val_score` is `NaN` in epochs `1`, `2`, and `3`
- `train_nan_inf_count` is extremely large:
  - epoch 1: `159300`
  - epoch 2: `164079`
  - epoch 3: `164079`
- `mini_val_metrics.nan_inf_count` is also non-zero:
  - epoch 1-3: `13312`

### Interpretation

- this is **not** a case where NaN appears only in unavailable eval placeholder fields.
- the training loss fields themselves are NaN, which is a blocker-level signal.
- there is no explicit `grad_norm` field in the current logs, so direct gradient NaN cannot be proven from logging alone.
- however, model outputs and/or loss inputs are already producing NaN/Inf during train/eval accounting.

### Required judgment

- `nan_in_train_loss`: `true`
- `nan_in_grad`: `false` (not directly logged)
- `nan_only_in_eval_placeholder`: `false`
- `training_valid_so_far`: `false`
- `possible_training_corruption`: `true`
- `recommended_nan_action`: `stop_and_restart_required`

## Deleted Checkpoint Directories

All deleted paths were checked to ensure:

- not under `FINAL360M`
- not `FINAL360I_struct360b_final`
- not `TRAIN360D`
- not `TRAIN360H`
- not `DSET2C`
- not raw data
- not reports
- not thesis assets

Deleted directories:

- `checkpoints/FINAL360J_tdir_loss_reweight` `288M`
- `checkpoints/FINAL360K_conservative_observability_weighting` `288M`
- `checkpoints/FINAL360L_translation_head_factorization` `123M`
- `checkpoints/SEQ360A_sequence_consistency_scale_drift` `288M`
- `checkpoints/STRUCT360A_implicit_spherical_cross_attention` `269M`
- `checkpoints/STRUCT360C_rotation_aware_fine_refinement` `248M`
- `checkpoints/ABLDVO2_PlainPairVO` `123M`
- `checkpoints/ABLDVO2_NoCrossImageInteraction` `129M`
- `checkpoints/ABLDVO2_NoSphericalGeometry` `239M`
- `checkpoints/ABLDVO3_SingleStagePoseRegression` `129M`
- `checkpoints/ABLVO360_PlainPairVO` `123M`
- `checkpoints/ABLVO360_NoCrossImageInteraction` `129M`
- `checkpoints/ABLVO360_NoSphericalGeometry` `239M`
- `checkpoints/TRAIN360C_spherical_pose_baseline` `175M`
- `checkpoints/T57b_no_dt_multiscale_tmag_head_400` `88M`

Deleted total size estimate:

- before cleanup `checkpoints`: `4.1G`
- after cleanup `checkpoints`: `1.3G`
- estimated reclaimed checkpoint space: `~2.8G`

## Protected Directories Checked

- `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/` kept
- `logs/FINAL360M_fulltrain_20260518_022354.log` kept
- `reports/FINAL360M_training_protocol.json` kept
- `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt` kept
- `checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/best_val.pt` kept
- `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt` kept
- `external_baselines/results/dset2c_360dvo_canonical/` kept
- `reports/*.json` kept
- `reports/*.md` kept
- `thesis/final_assets/` kept
- `data/` kept

## Post-cleanup Check

- disk free after cleanup: `7.4G`
- process alive after cleanup: `true`
- PID/etime/cmd after cleanup: `15106 41:38 /home/dovetao/miniconda3/envs/pytorch/bin/python tools/train_final360m_fulltrain_thesis_main.py configs/final360m_fulltrain_struct360b_thesis_main.yaml`
- current FINAL360M directory still present:
  - `candidate_epoch_01.pt`
  - `candidate_epoch_02.pt`
  - `candidate_epoch_03.pt`
  - `last.pt`
  - `train_log.jsonl`

## Remaining Gap

- target `>= 10G free`: not reached
- minimum `>= 8G free`: not reached
- current free `7.4G` is improved but still below target

Next batch requiring manual confirmation if more space is needed:

- `checkpoints/STRUCT360B_match_free_coarse_to_fine` `248M`
- non-directory, more surgical options not executed here:
  - unprotected `final.pt` files in otherwise protected directories
  - archive or non-checkpoint historical assets outside protected lists

## Recommended Next Action

- for disk: `manual confirmation cleanup still needed if you require >=8G free before more checkpoint churn`
- for training validity: `stop_and_restart_required`
- this task did **not** stop the running FINAL360M process, per instruction

## Final Summary

- `safe checkpoint cleanup executed`: `true`
- `FINAL360M process alive before`: `true`
- `FINAL360M process alive after`: `true`
- `disk free before`: `4.6G`
- `disk free after`: `7.4G`
- `deleted total size`: `~2.8G`
- `deleted checkpoint dirs`: `15`
- `protected checkpoint deleted`: `false`
- `current FINAL360M checkpoint deleted`: `false`
- `metrics modified`: `false`
- `nan_in_train_loss`: `true`
- `nan_in_grad`: `false`
- `nan_only_in_eval_placeholder`: `false`
- `training_valid_so_far`: `false`
- `recommended action`: `stop_and_restart_required`

