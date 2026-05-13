# MAINT5B Safe Cache Cleanup Only

## 1. Executive summary

- disk free before: `/` had `146M` free at cleanup start
- disk free after: `/` had `6.5G` free after cleanup
- space freed: about `6.35G`
- cleanup performed:
  - deleted `logs/maint5a_disk_audit`
  - ran `conda clean --all -y`
  - purged shared pip cache via `conda run -n pytorch python -m pip cache purge`
  - removed `/tmp/360DVO_official` `__pycache__`
- TRAIN360 sanity: pass
- BASE360 sanity: pass

## 2. Deleted cache/log locations

- `logs/maint5a_disk_audit`
  - before: `1.9G`
  - result: deleted
  - note: this was the accidental MAINT5A temporary audit output and was not committed
- conda package cache
  - target: `/home/dovetao/miniconda3/pkgs`
  - before: `19G`
  - after: `15G`
  - `conda clean --all --dry-run` estimate: `2.91G` tarballs + `288.3MB` packages + `1` index cache
- pip cache
  - target: `~/.cache/pip`
  - before: `489M`
  - after: `4.3M`
  - purge output: `Files removed: 602 (506.4 MB)`
- torch extension cache
  - target: `~/.cache/torch_extensions`
  - result: nothing to clean, directory absent / empty
- desktop/browser caches
  - not cleaned
  - reason: after logs + conda + pip cleanup, free space already recovered to `6.5G`, so further desktop disruption was unnecessary
- `/tmp` build cache
  - preserved `/tmp/360DVO_official` root directory
  - cleaned `/tmp/360DVO_official/dpvo/__pycache__`
  - before: `42M`
  - after: `41M`

## 3. Explicitly preserved

- `pytorch` env
- `base360dvo_rebuild` env
- base env
- PyTorch CUDA stack
- CUDA runtime and `nvidia-*` packages
- all checkpoints
- all reports
- source code
- `data/360DVO`
- `external_baselines/results` metrics, provenance, and reports
- `/tmp/360DVO_official` source directory
- browser caches under `~/.cache/google-chrome`
- `~/.cache/vscode-cpptools`
- `~/.cache/thumbnails`

## 4. Sanity checks

### TRAIN360 / pytorch

- torch: `2.9.1+cu128`
- torch cuda: `12.8`
- cuda available: `True`
- device count: `1`
- device: `NVIDIA GeForce RTX 3060 Laptop GPU`
- CUDA matmul: pass
- `cv2` import: pass (`4.13.0`)
- `numpy` / `scipy` import: pass

### BASE360 / base360dvo_rebuild

- `torch`, `evo`, `viser`, `cuda_ba` import: pass
- output: `BASE360 imports ok`

## 5. Remaining disk pressure

- `/home/dovetao/miniconda3` still large at `25G+`, with `pkgs` still `15G`
- `pytorch` env remains `16G`
- `base360dvo_rebuild` env remains `8.1G`
- `checkpoints` remains `3.3G`
- `data` remains `4.2G`
- `external_baselines` remains `1.4G`
- browser/desktop caches still present:
  - `~/.cache/vscode-cpptools 1.3G`
  - `~/.cache/google-chrome 881M`
  - `~/.cache/thumbnails 534M`

## 6. Next recommendation

- `proceed_to_MAINT6_checkpoint_pruning_after_user_approval`

Reason:

- safe cache cleanup has already recovered several GB
- the next major reclaim opportunity is checkpoint pruning, especially non-best `TRAIN360H` sweep runs
- package removal from `pytorch` is still a higher-risk operation than checkpoint pruning

## Compliance checklist

- `packages_uninstalled = false`
- `pytorch_env_removed = false`
- `base360dvo_rebuild_removed = false`
- `torch_cuda_stack_modified = false`
- `cuda_runtime_deleted = false`
- `checkpoints_deleted = false`
- `reports_deleted = false`
- `data_deleted = false`
- `training_executed = false`
- `conda_cache_cleaned = true`
- `pip_cache_cleaned = true`
- `torch_extension_cache_cleaned = false`
- `maint5a_logs_deleted = true`
- `train360_cuda_sanity_passed = true`
- `base360_sanity_passed = true`
