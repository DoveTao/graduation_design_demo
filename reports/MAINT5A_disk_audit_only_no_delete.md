# MAINT5A Disk Audit Only No Delete

## 1. Executive summary

- disk free current: `/dev/nvme1n1p3` had `897M` free at audit start (`df -h` in `logs/maint5a_disk_audit/phaseA_df_h.txt`); by the end of this audit only about `6.7M` remained because two recursive import-audit logs under `logs/maint5a_disk_audit/` grew very large. No cleanup or deletion was performed in this task.
- largest disk users under `/home/dovetao`: `miniconda3 29G`, `graduation_design_demo 12G`, `.cache 3.5G`, `.config 3.1G`, `Documents 3.0G`, `vcpkg 2.9G`, `.local 2.3G`.
- largest disk users inside project: `data 4.2G`, `checkpoints 3.3G`, `external_baselines 1.4G`, `.git 1.3G`, `logs 1.1G`.
- conda env sizes: `pytorch 16G`, `base360dvo_rebuild 8.1G`, base prefix `/home/dovetao/miniconda3 29G`.
- largest cache locations: `miniconda3/pkgs 19G`, `.cache/vscode-cpptools 1.3G`, `.cache/google-chrome 881M`, `.cache/thumbnails 534M`, `.cache/pip 489M`, `.cache/torch 98M`.
- largest checkpoint directories: `checkpoints/TRAIN360H_sweep 2.8G`, `checkpoints/TRAIN360C_spherical_pose_baseline 175M`, `checkpoints/TRAIN360D_observability_kstep_scale 175M`, `checkpoints/T57b_no_dt_multiscale_tmag_head_400 88M`.
- largest external baseline result directory: `external_baselines/results/base360_hkust_360dvo_official 1.3G`.
- pytorch suspected BASE360-only packages: `evo` and `viser` are installed in `pytorch`, are also available in `base360dvo_rebuild`, and were not found in TRAIN360 code imports. `cuda_ba` and editable `360dvo` were not found in `pytorch`.
- pytorch CUDA sanity: pass. `torch 2.9.1+cu128`, CUDA available, GPU matmul succeeded. `pytorch CUDA stack must be preserved`.
- BASE360 env sanity: pass. `base360dvo_rebuild` imports `torch`, `evo`, `viser`, and `cuda_ba`.
- no deletion performed: true.

## 2. Disk usage table

| scope | size / status | notes |
| --- | --- | --- |
| filesystem `/` | `115G` total, `109G` used, `897M` free at start, about `6.7M` free at end | end-state free space became critical during audit |
| project root | `12G` | dominated by `data`, `checkpoints`, `external_baselines`, `.git`, `logs` |
| conda root `/home/dovetao/miniconda3` | `29G` | `envs 9.5G`, `pkgs 19G` |
| env `pytorch` | `16G` | TRAIN360 main environment |
| env `base360dvo_rebuild` | `8.1G` | BASE360 official environment |
| `~/.cache` | `3.5G` | largest items are `vscode-cpptools 1.3G`, `google-chrome 881M`, `thumbnails 534M`, `pip 489M`, `torch 98M` |
| conda package cache | `19G` dir size | `conda clean --all --dry-run` estimates about `3.2G` immediately removable without touching envs |
| `checkpoints` | `3.3G` | `TRAIN360H_sweep 2.8G` is dominant |
| `external_baselines` | `1.4G` | `base360_hkust_360dvo_official 1.3G` dominates |
| `logs` | `1.1G` | this audit itself generated two huge recursive grep logs; not committed |
| `reports` | `1.9M` | small |
| `/tmp` | `44M` | `/tmp/360DVO_official 42M` |

## 3. Environment audit

### pytorch CUDA sanity

- python: `/home/dovetao/miniconda3/envs/pytorch/bin/python`
- torch: `2.9.1+cu128`
- torch cuda: `12.8`
- cuda available: `True`
- device: `NVIDIA GeForce RTX 3060 Laptop GPU`
- smoke result: CUDA matmul passed
- support libs imported: `cv2 4.13.0`, `numpy 2.4.4`, `scipy 1.17.1`
- conclusion: `pytorch CUDA stack must be preserved`

### base360dvo_rebuild sanity

- python: `/home/dovetao/miniconda3/envs/base360dvo_rebuild/bin/python`
- torch: `2.3.1`
- torch cuda: `12.1`
- cuda available: `True`
- torch cxx11 abi: `False`
- imports passed: `evo`, `viser`, `cuda_ba`
- `phaseF_base360_suspect_packages.txt` also shows editable `360dvo 0.0.0` from `/tmp/360DVO_official`

### final environment responsibility

- `TRAIN360 = pytorch`
- `BASE360 = base360dvo_rebuild via tools/base360c_official_python`

## 4. Suspected redundant pytorch packages

| package | found_in_pytorch | found_in_base360dvo_rebuild | TRAIN360_code_imports_it | BASE360_code_imports_it | recommendation | reason |
| --- | --- | --- | --- | --- | --- | --- |
| `torch` | yes | yes | yes | yes | `do_not_remove` | core GPU training stack for TRAIN360 |
| `opencv` / `cv2` | yes | not audited explicitly | yes | yes | `do_not_remove` | TRAIN360 tools and BASE360 runner both import it |
| `numpy` | yes | yes | yes | yes | `do_not_remove` | core numeric dependency |
| `scipy` | yes | yes | indirect/likely yes | yes | `do_not_remove` | core numeric dependency |
| `evo` | yes | yes | no | inferred yes | `possible_remove_later` | appears BASE360/eval-oriented; not imported by TRAIN360 code search |
| `viser` | yes | yes | no | inferred yes | `possible_remove_later` | likely installed for BASE360 visualization/debug, not imported by TRAIN360 code search |
| `cuda_ba` | no | yes | no | inferred yes | `keep` | already isolated to `base360dvo_rebuild`; no action needed in pytorch |
| editable `360dvo` | no | yes (`/tmp/360DVO_official`) | no | yes | `keep` | already isolated to `base360dvo_rebuild`; no action needed in pytorch |
| `pybind11` | not found in pytorch suspect audit | yes | no | likely yes | `keep` | only present in BASE360 env audit output |
| `ninja` | not found in pytorch suspect audit | yes | no | likely yes | `keep` | only present in BASE360 env audit output |

Notes:

- `TRAIN360_code_imports_it` was determined with `rg` over repository code while excluding `logs/`, `reports/`, `envs/`, `checkpoints/`, and `external_baselines/results/`.
- `BASE360_code_imports_it` is partly inferred from the dedicated BASE360 environment and editable official install, because the official source tree lives under `/tmp/360DVO_official` rather than inside this repo.
- No evidence was found that `cuda_ba` or editable `360dvo` polluted the `pytorch` environment.

## 5. Packages that must not be removed

- `torch`
- `torchvision` if used
- `pytorch-cuda` / `cudatoolkit` / CUDA runtime packages
- `cuda-runtime`
- `cuda-libraries`
- `nvidia-cuda-*`
- `nvidia-cublas`
- `nvidia-cudnn`
- `nvidia-cusolver`
- `nvidia-cusparse`
- `nvidia-nccl`
- `nvidia-curand`
- `triton`
- `cv2` / opencv
- `numpy`
- `scipy`
- `pillow`
- `pyyaml`
- `tqdm`
- `CUDA_HOME` toolkit, `/usr/local/cuda`, system CUDA, `nvcc`, GPU driver

## 6. Cache cleanup candidates for later MAINT5B

These are candidates only. Nothing was deleted here.

- conda package cache dir: `/home/dovetao/miniconda3/pkgs` is `19G`; `conda clean --all --dry-run` reports about `2.91G` tarballs + `288.3MB` packages + `1` index cache immediately removable.
- pip cache: `~/.cache/pip` is `489M`; `pip cache info` reports `506.4MB` HTTP cache.
- torch cache: `~/.cache/torch` is `98M`; protected in this task and not recommended as first target.
- torch extensions cache: not present / no size reported.
- NVIDIA compute cache: `~/.nv/ComputeCache` is `96K`.
- `/tmp` build area: total `/tmp` is `44M`; `/tmp/360DVO_official` is `42M` and explicitly protected, so only other tiny `/tmp` leftovers would be future candidates.
- project `__pycache__`: repo-local `__pycache__` is about `932K`.
- non-project desktop caches also exist if broader cleanup is allowed later: `~/.cache/vscode-cpptools 1.3G`, `~/.cache/google-chrome 881M`, `~/.cache/thumbnails 534M`.

## 7. Checkpoint pruning candidates for later MAINT6

Must preserve:

- `checkpoints/TRAIN360C_spherical_pose_baseline/best_val.pt`
- `checkpoints/TRAIN360C_spherical_pose_baseline/final.pt`
- `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`
- `checkpoints/TRAIN360D_observability_kstep_scale/final.pt`
- `checkpoints/TRAIN360H_sweep/TRAIN360H_F01_TRAIN360H_S1_best_seed1/best_val.pt`
- `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`

TRAIN360H best run according to existing report:

- best run_id: `TRAIN360H_F01_TRAIN360H_S1_best_seed1`
- source: `reports/TRAIN360H_hyperparameter_sweep.md`

Future pruning candidates only after approval:

- all non-best TRAIN360H run directories except `TRAIN360H_F01_TRAIN360H_S1_best_seed1`
- candidate run ids:
  - `TRAIN360H_A0_reproduce_D_seed0`
  - `TRAIN360H_A1_obs_only_seed0`
  - `TRAIN360H_A3_scale_only_seed0`
  - `TRAIN360H_A4_obs_kstep_no_scale_seed0`
  - `TRAIN360H_C01_lr3e5_dir2p0_scale0p05_seed0`
  - `TRAIN360H_C02_lr5e5_dir2p0_scale0p1_seed0`
  - `TRAIN360H_C03_lr1e4_dir1p5_scale0p2_seed0`
  - `TRAIN360H_C04_obs_scale_no_k_seed0`
  - `TRAIN360H_C05_obs_k_no_scale_seed0`
  - `TRAIN360H_C06_balanced_scale_recovery_seed0`
  - `TRAIN360H_R01_lower_lr_more_tdir_seed0`
  - `TRAIN360H_R02_more_scale_seed0`
  - `TRAIN360H_R03_tighter_clip_seed0`
  - `TRAIN360H_S1_best_seed1`
  - `TRAIN360H_S2_best_seed2`
- estimated space if pruning non-best TRAIN360H `best_val.pt` + `final.pt`: about `15 * 183MB ≈ 2.7G`
- no checkpoint was deleted in this task

## 8. Recommended next action

- `proceed_to_MAINT5B_safe_cache_cleanup_only`

Reason:

- the largest safely-addressable space is in caches, especially conda package cache and pip cache
- pytorch shows only a small number of suspected BASE360-only packages, and removing packages from a live CUDA training env is riskier than cache cleanup

## 9. Compliance checklist

- `deletion_performed = false`
- `packages_uninstalled = false`
- `conda_clean_executed = false`
- `pip_cache_purged = false`
- `checkpoints_deleted = false`
- `pytorch_env_modified = false`
- `base360dvo_rebuild_modified = false`
- `training_executed = false`
- `train360_cuda_sanity_checked = true`
- `base360_sanity_checked = true`

