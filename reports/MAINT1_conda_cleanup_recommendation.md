# MAINT1 Conda Cleanup Recommendation

## Keep Environments

- `pytorch`: primary TRAIN360 development environment; imports `torch/cv2/numpy/scipy/evo/viser` and sees CUDA successfully.
- `base360dvo_rebuild`: keep as the HKUST official 360DVO environment, but document that it must be launched via `tools/base360c_official_python` because plain `conda run -n base360dvo_rebuild python` still hits the known `iJIT_NotifyEvent` / `libc10.so` loader path issue.
- `base`: keep as the general-purpose root conda environment and not as a project runtime.

## Candidate Removal Environment

- `base360dvo`: removal candidate only, do not delete by default.

## Candidate Analysis

- `base360dvo` fails `import torch` with `undefined symbol: iJIT_NotifyEvent`.
- `base360dvo` cannot import `cv2`, `scipy`, `evo`, `viser`, or `cuda_ba` from a plain environment invocation.
- Repository search found no active runner scripts that require `base360dvo`; the selected wrapper points to `base360dvo_rebuild` instead.
- Backup export exists at `envs/base360dvo.candidate_old.lock.yml` and `envs/base360dvo.candidate_old.pip-freeze.txt`.
- A replacement exists: `base360dvo_rebuild`, validated through `tools/base360c_official_python` with `torch/evo/viser/cuda_ba` import success.

## Runtime Wrapper Status

- `tools/base360c_official_python` currently targets `base360dvo_rebuild`.
- Wrapper test output:

```text
torch 2.3.1
evo v1.36.4
viser 1.0.27
cuda_ba ok
```

- Plain `conda run` diagnostics for `base360dvo_rebuild` should remain recorded in `reports/MAINT1_conda_env_inventory.md` so future maintainers understand why the wrapper is mandatory.

## Removal Gate Check

- `envs/base360dvo.candidate_old.lock.yml` exists: yes.
- Grep confirms no active script dependency on `base360dvo`: yes.
- `base360dvo_rebuild` can import `torch/evo/viser/cuda_ba`: yes, through `tools/base360c_official_python`.
- `tools/base360c_official_python` points to `base360dvo`: no, it points to `base360dvo_rebuild`.
- Deletion action recorded in report: recommendation only.

## Recommended But Not Executed

- Suggested command: `conda env remove -n base360dvo`
- Execution status: not executed in this maintenance task.
- Reason: conservative cleanup is safer until you explicitly want to reclaim the old environment.
