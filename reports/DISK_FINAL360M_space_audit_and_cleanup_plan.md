# DISK FINAL360M space audit and cleanup plan

## Status

- `disk audit executed`: `true`
- `FINAL360M process alive`: `true`
- `elapsed_time`: `35:53`
- `current_epoch/update if visible`: `epoch 3 visible in checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/train_log.jsonl`
- `current training note`: `epoch 1-3 logs currently show NaN-heavy training stats; this audit did not modify or stop training`
- `current disk free before`: `4.6G`
- `current disk free after cache cleanup`: `4.6G`
- `estimated_required_free_space`: `~0.6-1.0 GiB additional for remaining FINAL360M checkpoint churn, but >=8 GiB is safer operational headroom`
- `current_training_alive`: `true`
- `urgent cleanup needed`: `true`
- `safe cache cleanup executed`: `true`
- `checkpoint deleted`: `false`
- `metrics modified`: `false`
- `reports written`: `true`
- `recommended_action`: `urgent_cleanup_recommended`

## Phase A

- process alive: `true`
- PID/etime/cmd: `15106 35:53 /home/dovetao/miniconda3/envs/pytorch/bin/python tools/train_final360m_fulltrain_thesis_main.py configs/final360m_fulltrain_struct360b_thesis_main.yaml`
- log visibility:
  - `epoch 1/20 train start` is present in the main log
  - `train_log.jsonl` shows progress through `epoch 3`
  - latest visible row: `epoch=3`, `elapsed_time_sec=2007.8507`, `mini_val_score=NaN`, `train_nan_inf_count=164079`
- current disk free: `/dev/nvme1n1p3 -> 4.6G free`

## Top Disk Consumers

- repo root: `11G`
- `data`: `4.2G`
- `checkpoints`: `4.1G`
- `.git`: `1.3G`
- `external_baselines`: `1.3G`
- `archive`: `87M`
- `thesis`: `26M`
- `reports`: `2.6M`

## Top Checkpoint Consumers

- `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main`: `415M`
- `checkpoints/FINAL360J_tdir_loss_reweight`: `288M`
- `checkpoints/FINAL360K_conservative_observability_weighting`: `288M`
- `checkpoints/SEQ360A_sequence_consistency_scale_drift`: `288M`
- `checkpoints/STRUCT360A_implicit_spherical_cross_attention`: `269M`
- `checkpoints/FINAL360I_struct360b_final`: `248M`
- `checkpoints/STRUCT360B_match_free_coarse_to_fine`: `248M`
- `checkpoints/STRUCT360C_rotation_aware_fine_refinement`: `248M`
- `checkpoints/ABLDVO2_NoSphericalGeometry`: `239M`
- `checkpoints/ABLVO360_NoSphericalGeometry`: `239M`

## Large Files Over 100M

- current FINAL360M active files:
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/last.pt` `~104M`
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/candidate_epoch_01.pt` `~104M`
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/candidate_epoch_02.pt` `~104M`
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/candidate_epoch_03.pt` `~104M`
- protected mainline files:
  - `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt` `~91.6M`
  - `checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/best_val.pt` `~91.6M`
  - `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt` `~108.8M`
- large non-current experiment files exist across `ABLDVO2`, `ABLDVO3`, `ABLVO360`, `FINAL360J/K/L`, `SEQ360A`, `STRUCT360A/C`

## Safe To Delete Now

These were cache-only items and are the only items auto-deleted in this audit.

- `tools/__pycache__` `4.9M`
- `tests/__pycache__` `572K`
- `train360/legacy/__pycache__` `448K`
- `train360/core/__pycache__` `272K`
- `models/__pycache__` `164K`
- `datasets/__pycache__` `68K`
- `tools/current/__pycache__` `12K`
- `train360/__pycache__` `8.0K`
- total safe cache cleanup: `6.4M`

## Delete Only After User Confirmation

These are the highest-signal candidates if more space is needed. They were not deleted.

- current FINAL360M rotating candidates:
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/candidate_epoch_01.pt` `~104M`
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/candidate_epoch_02.pt` `~104M`
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/candidate_epoch_03.pt` `~104M`
  - note: these belong to the current run, so do not delete while training is active unless you explicitly choose to sacrifice resume/selection flexibility
- older non-mainline experiment directories:
  - `checkpoints/FINAL360J_tdir_loss_reweight` `288M`
  - `checkpoints/FINAL360K_conservative_observability_weighting` `288M`
  - `checkpoints/SEQ360A_sequence_consistency_scale_drift` `288M`
  - `checkpoints/STRUCT360A_implicit_spherical_cross_attention` `269M`
  - `checkpoints/STRUCT360C_rotation_aware_fine_refinement` `248M`
  - `checkpoints/ABLDVO2_NoSphericalGeometry` `239M`
  - `checkpoints/ABLVO360_NoSphericalGeometry` `239M`
  - `checkpoints/ABLDVO3_SingleStagePoseRegression` `129M`
  - `checkpoints/ABLDVO2_NoCrossImageInteraction` `129M`
  - `checkpoints/ABLDVO2_PlainPairVO` `123M`
  - `checkpoints/ABLVO360_NoCrossImageInteraction` `129M`
  - `checkpoints/ABLVO360_PlainPairVO` `123M`
- optional older baseline / experiment singles:
  - `checkpoints/FINAL360L_translation_head_factorization` `123M`
  - `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` `~91M`
  - `checkpoints/TRAIN360C_spherical_pose_baseline` `175M`

## Do Not Delete

- `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main/`
- `logs/FINAL360M_fulltrain_20260518_022354.log`
- `reports/FINAL360M_training_protocol.json`
- `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/best_val.pt`
- `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- `external_baselines/results/dset2c_360dvo_canonical/`
- `reports/*.json`
- `reports/*.md`
- `thesis/final_assets/`
- `data/`

## Judgment

- current free space is below both the `8 GiB` caution line and the `5 GiB` urgent line.
- the safe cache cleanup only recovered `6.4M`, which is too small to materially change risk.
- for the current FINAL360M run alone, remaining checkpoint churn likely needs less than `1 GiB`.
- however, `4.6G` free leaves weak safety margin for additional candidate saves, full-val/best checkpoint writes, unrelated temp files, or any parallel tooling.

## Recommended Action

- `recommended_action = urgent_cleanup_recommended`
- first choice: free space from older non-mainline experiment checkpoints after user confirmation
- second choice: if absolutely necessary, prune old `candidate_epoch_*.pt` only after explicit confirmation and only with awareness that this weakens FINAL360M resume/selection safety
- do not stop FINAL360M during cleanup unless a separate task explicitly authorizes that

## Final Summary

- `disk audit executed`: `true`
- `FINAL360M process alive`: `true`
- `current disk free before`: `4.6G`
- `current disk free after cache cleanup`: `4.6G`
- `urgent cleanup needed`: `true`
- `safe cache cleanup executed`: `true`
- `checkpoint deleted`: `false`
- `metrics modified`: `false`
- `reports written`: `true`
- `recommended action`: `urgent_cleanup_recommended`

