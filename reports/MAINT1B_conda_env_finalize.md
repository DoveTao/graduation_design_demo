# MAINT1B Conda Environment Finalize

## 1. Executive Summary

- `pytorch` status: pass
- `base360dvo_rebuild` status: pass via `tools/base360c_official_python`
- `base360dvo` status before cleanup: existed, incomplete, and failed required imports
- `base360dvo` removed: true

## 2. Final Environment Policy

- TRAIN360 env = `pytorch`
- BASE360 official env = `base360dvo_rebuild` via `tools/base360c_official_python`
- `base` = retained
- `base360dvo` = removed after backup/export verification and dependency audit

## 3. Environment Presence Check

```text
# conda environments:
#
# * -> active
# + -> frozen
base                     /home/dovetao/miniconda3
base360dvo_rebuild       /home/dovetao/miniconda3/envs/base360dvo_rebuild
pytorch                  /home/dovetao/miniconda3/envs/pytorch
```

- existed before cleanup:
  - `pytorch`: yes
  - `base360dvo_rebuild`: yes
  - `base360dvo`: yes
  - `base`: yes
- exists after cleanup:
  - `pytorch`: yes
  - `base360dvo_rebuild`: yes
  - `base360dvo`: no
  - `base`: not listed by `conda env list` output style? actually listed above as base root and retained

## 4. Import Checks

### `pytorch`

```text
python: /home/dovetao/miniconda3/envs/pytorch/bin/python
version: 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:16:04) [GCC 11.2.0]
torch: 2.9.1+cu128
torch cuda: 12.8
cuda available: True
device: NVIDIA GeForce RTX 3060 Laptop GPU
cv2: 4.13.0
numpy: 2.4.4
```

- result: `torch` import pass, `cv2` import pass, CUDA visible

### `base360dvo_rebuild` via wrapper

```text
python: /home/dovetao/miniconda3/envs/base360dvo_rebuild/bin/python
version: 3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]
torch: 2.3.1
torch cuda: 12.1
cuda available: True
torch cxx11 abi: False
evo: import ok
viser: import ok
cuda_ba: import ok
```

- result: `torch`, `evo`, `viser`, and `cuda_ba` all pass when launched through `tools/base360c_official_python`
- wrapper target check: `tools/base360c_official_python` points to `base360dvo_rebuild`, not `base360dvo`

### `base360dvo` old status before removal

```text
python: /home/dovetao/miniconda3/envs/base360dvo/bin/python
torch import failed: ImportError('/home/dovetao/miniconda3/envs/base360dvo/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')
evo import failed: ModuleNotFoundError("No module named 'evo'")
viser import failed: ModuleNotFoundError("No module named 'viser'")
cuda_ba import failed: ModuleNotFoundError("No module named 'cuda_ba'")
```

- result: old official env candidate was incomplete and not suitable to retain

## 5. Dependency Audit

- active script dependency on `base360dvo`: none found
- `rg -n "base360dvo" tools configs scripts envs reports .gitignore` only found:
  - historical references in reports
  - backup export files under `envs/`
  - the live wrapper pointing to `base360dvo_rebuild`
- deletion safety conclusion: no active runtime script still depended on `base360dvo`

## 6. Backup Files

- `envs/base360dvo.candidate_old.lock.yml`: present
- `envs/base360dvo.candidate_old.pip-freeze.txt`: present

## 7. Deletion Log

- first attempted command: `conda env remove -n base360dvo`
- first attempt result: package removal executed but interactive confirmation caused `CondaSystemExit: Exiting.` before final noninteractive completion
- final command used: `conda env remove -n base360dvo -y`
- final command result: success
- post-delete filesystem check:

```text
ls: cannot access '/home/dovetao/miniconda3/envs/base360dvo': No such file or directory
```

- post-delete runtime check:

```text
EnvironmentLocationNotFound: Not a conda environment: /home/dovetao/miniconda3/envs/base360dvo
```

- final `conda env list`:

```text
# conda environments:
#
# * -> active
# + -> frozen
base                     /home/dovetao/miniconda3
base360dvo_rebuild       /home/dovetao/miniconda3/envs/base360dvo_rebuild
pytorch                  /home/dovetao/miniconda3/envs/pytorch
```

## 8. Compliance Checklist

- `pytorch_removed = false`
- `base360dvo_rebuild_removed = false`
- `base_removed = false`
- `duplicate_base360dvo_removed = true`
- `backup_export_exists = true`
- `train360_checkpoints_deleted = false`
- `base360_outputs_deleted = false`
- `reports_deleted = false`
- `training_executed = false`
