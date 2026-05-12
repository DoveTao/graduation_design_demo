# MAINT1 Conda and Git Cleanup Summary

## 1. Conda Summary

- existing envs: `base`, `base360dvo`, `base360dvo_rebuild`, `pytorch`
- selected main env for TRAIN360: `pytorch`
- selected official env for BASE360: `base360dvo_rebuild`
- duplicate / stale env candidates: `base360dvo`
- any env removed: false
- env export paths:
  - `envs/pytorch.lock.yml`
  - `envs/pytorch.pip-freeze.txt`
  - `envs/base360dvo_rebuild.lock.yml`
  - `envs/base360dvo_rebuild.pip-freeze.txt`
  - `envs/base360dvo.candidate_old.lock.yml`
  - `envs/base360dvo.candidate_old.pip-freeze.txt`

## 2. Runtime Wrappers

- `tools/base360c_official_python` status: kept and stabilized; now resolves repo-relative shim path and supports override env vars `BASE360C_OFFICIAL_ENV_ROOT` and `BASE360C_ITTNOTIFY_SHIM`.
- selected official python path: `/home/dovetao/graduation_design_demo/tools/base360c_official_python`
- plain `conda run -n base360dvo_rebuild python` import status: loader issue remains in inventory.
- wrapper import status: `torch/evo/viser/cuda_ba` import pass.

## 3. Git Summary

- cleanup started from branch: `experiment/dset2c-360dvo-dataset-hygiene`
- current branch: `maintenance/env-git-cleanup`
- remote: `origin	git@github.com:DoveTao/graduation_design_demo.git (fetch)`
- ahead/behind status: no upstream configured yet for `maintenance/env-git-cleanup`.
- uncommitted files before cleanup: captured in `reports/MAINT1_git_inventory.md` and included TRAIN360, BASE360, RESULTS360, env exports, and maintenance reports.
- committed files in this task: `.gitignore`, `tools/base360c_official_python`, `envs/*.yml`, `envs/*.txt`, and the `reports/MAINT1*` maintenance reports in commit `0f00dfd`.
- ignored files: generated smoke images, official resized inputs, training logs, `logs/`, checkpoints, and prior heavy artifact patterns remain local.
- large artifacts intentionally not committed: TRAIN360C weights and the 1.3G `external_baselines/results/base360_hkust_360dvo_official` directory.

## 4. Checkpoint / Artifact Policy

- TRAIN360C checkpoint retained locally: true
- checkpoint committed: false
- BASE360 results retained locally: true
- reports committed: true, for `reports/MAINT1*` only

## 5. Push Status

- branch pushed: false
- remote branch: none yet
- latest commit hash: `0f00dfd`
- tags pushed if any: none in this task

## 6. Next Recommendation

- next recommendation: `proceed_to_TRAIN360D_observability_kstep_scale_stabilization` after committing the intended TRAIN360 / BASE360 / RESULTS360 artifacts in reviewable batches.

## Compliance Checklist

- `training_executed = false`
- `fine_tune_executed = false`
- `train360_weights_modified = false`
- `train360_checkpoints_deleted = false`
- `base360_outputs_deleted = false`
- `reports_deleted = false`
- `s5_locked_metrics_modified = false`
- `git_reset_hard_used = false`
- `git_clean_fdx_used = false`
- `force_push_used = false`
- `conda_env_removed = false`
- `if conda_env_removed, backup_export_exists = false`
