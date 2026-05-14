# MAINT5A Checkpoint Pruning Candidates

This file records future pruning candidates only. No checkpoint was deleted in MAINT5A.

## Preserve

- `checkpoints/TRAIN360C_spherical_pose_baseline/best_val.pt`
- `checkpoints/TRAIN360C_spherical_pose_baseline/final.pt`
- `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`
- `checkpoints/TRAIN360D_observability_kstep_scale/final.pt`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/best_val.pt`
- `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`

## TRAIN360H best run

- best run_id: `TRAIN360H_F01_TRAIN360H_S1_best_seed1`
- best checkpoint path: `checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/best_val.pt`
- source: `reports/TRAIN360H_hyperparameter_sweep.md`

## Future pruning candidates after approval

Each non-best TRAIN360H run directory is about `175M`, with `best_val.pt` and `final.pt` each about `91.6M`.

- `checkpoints/TRAIN360H_sweep/TRAIN360H_A0_reproduce_D_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_A1_obs_only_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_A3_scale_only_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_A4_obs_kstep_no_scale_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_C01_lr3e5_dir2p0_scale0p05_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_C02_lr5e5_dir2p0_scale0p1_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_C03_lr1e4_dir1p5_scale0p2_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_C04_obs_scale_no_k_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_C05_obs_k_no_scale_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_C06_balanced_scale_recovery_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_R01_lower_lr_more_tdir_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_R02_more_scale_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_R03_tighter_clip_seed0`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_S1_best_seed1`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_S2_best_seed2`

Estimated reclaim if pruning all non-best TRAIN360H `best_val.pt` + `final.pt` pairs:

- about `15 * 183M ≈ 2.7G`

## Other checkpoint notes

- `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` is about `91.1M` and should be preserved.
- `TRAIN360C` and `TRAIN360D` each have `best_val.pt` and `final.pt` around `91.6M` and should be preserved.
- Phase C threshold was `>100M`, so these `.pt` files do not appear in the large-file listing even though they are meaningful storage consumers.
