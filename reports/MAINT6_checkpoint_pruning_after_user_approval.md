# MAINT6 Checkpoint Pruning After User Approval

## 1. Executive summary

- disk free before: `6.5G`
- disk free after: `9.0G`
- space freed: `2,747,489,949` bytes, about `2.56G`
- deleted checkpoint count: `30`
- must-keep checkpoints preserved: yes
- TRAIN360 sanity result: pass

## 2. Pruning policy

- kept `T57b`, `TRAIN360C`, `TRAIN360D`, and `TRAIN360H` best checkpoint
- kept `checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/final.pt` by default
- deleted only non-best `TRAIN360H_sweep` `.pt` checkpoints
- preserved `config.yaml`, `metadata.json`, `metrics_val.json`, `metrics_test.json`, `run_report.md`, and small `train_log.jsonl`
- no `reports/`, source code, data, or BASE360 outputs were deleted

## 3. Deleted files manifest

All deleted files had reason `non_best_train360h_sweep_checkpoint`.

| path | size | reason |
| --- | --- | --- |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_A0_reproduce_D_seed0/best_val.pt` | `91600420` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_A0_reproduce_D_seed0/final.pt` | `91565615` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_A1_obs_only_seed0/best_val.pt` | `91600420` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_A1_obs_only_seed0/final.pt` | `91565615` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_A3_scale_only_seed0/best_val.pt` | `91600420` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_A3_scale_only_seed0/final.pt` | `91565615` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_A4_obs_kstep_no_scale_seed0/best_val.pt` | `91600484` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_A4_obs_kstep_no_scale_seed0/final.pt` | `91565679` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C01_lr3e5_dir2p0_scale0p05_seed0/best_val.pt` | `91600228` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C01_lr3e5_dir2p0_scale0p05_seed0/final.pt` | `91565423` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C02_lr5e5_dir2p0_scale0p1_seed0/best_val.pt` | `91600484` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C02_lr5e5_dir2p0_scale0p1_seed0/final.pt` | `91565679` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C03_lr1e4_dir1p5_scale0p2_seed0/best_val.pt` | `91600484` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C03_lr1e4_dir1p5_scale0p2_seed0/final.pt` | `91565679` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C04_obs_scale_no_k_seed0/best_val.pt` | `91600484` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C04_obs_scale_no_k_seed0/final.pt` | `91565615` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C05_obs_k_no_scale_seed0/best_val.pt` | `91600484` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C05_obs_k_no_scale_seed0/final.pt` | `91565615` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C06_balanced_scale_recovery_seed0/best_val.pt` | `91600484` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_C06_balanced_scale_recovery_seed0/final.pt` | `91565679` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_R01_lower_lr_more_tdir_seed0/best_val.pt` | `91600484` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_R01_lower_lr_more_tdir_seed0/final.pt` | `91565679` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_R02_more_scale_seed0/best_val.pt` | `91600420` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_R02_more_scale_seed0/final.pt` | `91565615` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_R03_tighter_clip_seed0/best_val.pt` | `91600484` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_R03_tighter_clip_seed0/final.pt` | `91565615` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_S1_best_seed1/best_val.pt` | `91600164` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_S1_best_seed1/final.pt` | `91565359` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_S2_best_seed2/best_val.pt` | `91600164` | non-best TRAIN360H sweep checkpoint |
| `checkpoints/TRAIN360H_sweep/TRAIN360H_S2_best_seed2/final.pt` | `91565359` | non-best TRAIN360H sweep checkpoint |

## 4. Preserved checkpoints

- `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`: exists
- `checkpoints/TRAIN360C_spherical_pose_baseline/best_val.pt`: exists
- `checkpoints/TRAIN360C_spherical_pose_baseline/final.pt`: exists
- `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`: exists
- `checkpoints/TRAIN360D_observability_kstep_scale/final.pt`: exists
- `checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/best_val.pt`: exists
- `checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/final.pt`: exists

## 5. Remaining large disk users

- `checkpoints`: `663M`
- `checkpoints/TRAIN360H_sweep`: `176M`
- `data`: `4.2G`
- `external_baselines`: `1.4G`
- conda envs still remain large from previous audit:
  - `pytorch`: `16G`
  - `base360dvo_rebuild`: `8.1G`

## 6. Next recommendation

- `no_more_cleanup_needed`

Reason:

- disk free is now `9.0G`
- major low-risk checkpoint reclaim has already been completed
- further cleanup would start trading off convenience or reproducibility more aggressively

## Compliance checklist

- `training_executed = false`
- `fine_tune_executed = false`
- `packages_uninstalled = false`
- `pytorch_env_removed = false`
- `base360dvo_rebuild_removed = false`
- `torch_cuda_stack_modified = false`
- `train360c_checkpoints_preserved = true`
- `train360d_checkpoints_preserved = true`
- `train360h_best_checkpoint_preserved = true`
- `t57b_checkpoint_preserved = true`
- `reports_deleted = false`
- `data_deleted = false`
- `base360_outputs_deleted = false`
- `git_reset_hard_used = false`
- `git_clean_fdx_used = false`
