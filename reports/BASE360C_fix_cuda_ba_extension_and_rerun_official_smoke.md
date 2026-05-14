# BASE360C fix cuda_ba extension and rerun official smoke

## 1. Executive summary
- `BASE360C execution = partial`
- `cuda_ba build = success`
- `cuda_ba import = success`
- `official smoke test = success` on a canonical `mountains` clip using a `0.5x` image-directory adapter
- canonical val/test rerun completed and produced real `pred_tum.txt` files plus trajectory metrics
- remaining caveat: BASE360 outputs are currently comparable to TRAIN360C only `partial`, because the official demo path exports trajectories but not native pairwise signed-direction / scale-ratio metrics

## 2. cuda_ba inventory
- source files:
  - `/tmp/360DVO_official/dpvo/fastba/ba.cpp`
  - `/tmp/360DVO_official/dpvo/fastba/ba_cuda.cu`
- build entrypoint: `/tmp/360DVO_official/setup.py`
- extension module name: `cuda_ba`
- related compiled modules:
  - `cuda_corr`
  - `lietorch_backends`
- official install command from README: `pip install .`
- supported alternative build path: `python setup.py build_ext --inplace` via `BuildExtension`
- expected import path: top-level `import cuda_ba`
- `pip install .` / `pip install -e .` is required to register the package cleanly; inplace build is technically possible but was not the final successful path
- actual successful build command:
  - `LD_PRELOAD=logs/base360c_cuda_ba/libittnotify_shim.so`
  - `CUDA_HOME=/home/dovetao/miniconda3/envs/base360dvo_rebuild`
  - `TORCH_CUDA_ARCH_LIST=8.6`
  - `MAX_JOBS=1`
  - `CPATH=/home/dovetao/miniconda3/envs/base360dvo_rebuild/targets/x86_64-linux/include/cccl:/home/dovetao/miniconda3/envs/base360dvo_rebuild/targets/x86_64-linux/include:/home/dovetao/miniconda3/envs/base360dvo_rebuild/include`
  - `CPLUS_INCLUDE_PATH=$CPATH`
  - `pip install -v -e . --no-build-isolation`

## 3. ABI / CUDA / compiler diagnostics
- selected env: `base360dvo_rebuild`
- selected python: `/home/dovetao/miniconda3/envs/base360dvo_rebuild/bin/python`
- python version: `3.11`
- torch version: `2.3.1`
- torch CUDA version: `12.1`
- torch cxx11 ABI: `False`
- nvcc version: `12.1.105`
- gcc / g++ version: `11.4.0`
- CUDA_HOME: `/home/dovetao/miniconda3/envs/base360dvo_rebuild`
- TORCH_CUDA_ARCH_LIST: `8.6`
- import status:
  - `torch import`: pass only after preloading a minimal `iJIT` shim
  - `cuda_ba import`: pass
  - `evo / viser / cv2 / yacs / torch_scatter`: pass
- notable blocked envs:
  - existing `pytorch` env (`torch 2.9.1+cu128`) failed official extension compilation on old CUDA/Torch API usage in `cuda_corr`
  - existing `base360dvo` env imported `torch` with `undefined symbol: iJIT_NotifyEvent`

## 4. Build attempts
- attempt 1:
  - command: `pip install -v -e .`
  - env: `base360dvo_rebuild`
  - log: `logs/base360c_cuda_ba/build_base360dvo_env.log`
  - result: fail
  - root cause: editable build isolation hid `torch`, so setup bootstrap stopped at `ModuleNotFoundError: No module named 'torch'`
- attempt 2:
  - command: `pip install -v -e . --no-build-isolation`
  - env: `base360dvo_rebuild`
  - log: `logs/base360c_cuda_ba/build_base360dvo_env_nobi.log`
  - result: fail
  - root cause: CUDA dev headers missing, `cuda_runtime.h` not found
- attempt 3:
  - command: `pip install -v -e . --no-build-isolation`
  - env: `base360dvo_rebuild` after `cuda-cudart-dev` install
  - log: `logs/base360c_cuda_ba/build_base360dvo_env_nobi_headers.log`
  - result: fail
  - root cause: `nv/target` / CCCL include path missing from default compiler search path
- attempt 4:
  - command: `pip install -v -e . --no-build-isolation` with explicit `CPATH` / `CPLUS_INCLUDE_PATH`
  - env: `base360dvo_rebuild`
  - log: `logs/base360c_cuda_ba/build_base360dvo_env_nobi_cpath.log`
  - result: success
  - root cause fixed: CUDA/CCCL header search path
- auxiliary fixes:
  - `logs/base360c_cuda_ba/create_base360dvo_rebuild.log`
  - `logs/base360c_cuda_ba/install_base360dvo_rebuild_deps.log`
  - `logs/base360c_cuda_ba/install_cuda_dev_headers.log`
  - `logs/base360c_cuda_ba/pin_numpy_base360dvo_rebuild.log`
  - `logs/base360c_cuda_ba/import_test.log`

## 5. Runtime patch / wrapper
- minimal shim source: `tools/base360c_ittnotify_shim.c`
- compiled shim: `logs/base360c_cuda_ba/libittnotify_shim.so`
- purpose: satisfy `iJIT_NotifyEvent`, `iJIT_IsProfilingActive`, `iJIT_GetNewMethodID` so `torch 2.3.1` loads on this machine
- official runtime wrapper: `tools/base360c_official_python`
- wrapper exports:
  - `LD_PRELOAD=logs/base360c_cuda_ba/libittnotify_shim.so`
  - `LD_LIBRARY_PATH=<base360dvo_rebuild torch/lib>:<base360dvo_rebuild lib>:...`

## 6. Smoke test result
- successful smoke command:
  - `tools/base360c_official_python /tmp/360DVO_official/demo.py --network /tmp/360DVO_official/360dvo.pth --imagedir external_baselines/results/base360_hkust_360dvo_official/smoke/mountains_clip8_resized --config /tmp/360DVO_official/config/360.yaml --save_trajectory --name base360c_smoke_mountains8_resized`
- official weight used: `/tmp/360DVO_official/360dvo.pth`
- stdout: `external_baselines/results/base360_hkust_360dvo_official/smoke/stdout_base360c_clip8_resized.log`
- stderr: `external_baselines/results/base360_hkust_360dvo_official/smoke/stderr_base360c_clip8_resized.log`
- smoke summary log: `logs/base360c_cuda_ba/smoke_test_clip8_resized.log`
- output trajectory: `external_baselines/results/base360_hkust_360dvo_official/smoke/pred_tum_base360c_smoke_mountains8_resized.txt`
- notes:
  - raw full-resolution image-directory smoke (`clip32`, `clip8`) OOMed
  - root cause: official `image_stream()` keeps `if 0:` and does not downscale image-directory inputs, unlike `video_stream()`
  - fix used: input adapter that writes `0.5x` resized images before calling official demo

## 7. BASE360 rerun result
- runner used: `tools/run_base360_hkust_360dvo_official.py --official-python /home/dovetao/graduation_design_demo/tools/base360c_official_python`
- runner fix:
  - canonical sequence inputs are materialized into `official_demo_input_x0p5/`
  - successful runs now copy `/tmp/360DVO_official/saved_trajectories/<split>_<sequence>.txt` into each sequence `pred_tum.txt`
- post-run evaluation fix:
  - official trajectories were timestamped with frame indices
  - `pred_tum.txt` timestamps were remapped 1:1 onto canonical GT timestamps before trajectory evaluation
- val coverage: `100.00%`
- test coverage: `100.00%`
- val aggregate:
  - `ate_none = 80.793159`
  - `ate_se3 = 57.726244`
  - `ate_sim3 = 1.728649`
  - `path_ratio = 0.166154685`
- test aggregate:
  - `ate_none = 101.616185`
  - `ate_se3 = 78.585742`
  - `ate_sim3 = 2.931001`
  - `path_ratio = 0.043618001`
- per-sequence:
  - `val/mountains`: `sim3_ATE = 0.522377`, `path_ratio = 0.409545`
  - `val/downhill_biking`: `sim3_ATE = 3.330729`, `path_ratio = 0.110386`
  - `test/snowmobile`: `sim3_ATE = 3.374917`, `path_ratio = 0.034876`
  - `test/ridge_to_lake`: `sim3_ATE = 2.352784`, `path_ratio = 0.050396`
- artifact locations:
  - per-sequence `pred_tum.txt`, `gt_tum.txt`, `run_metadata.json`, `stdout.log`, `stderr.log`
  - split aggregates:
    - `reports/BASE360_metrics_val.json`
    - `reports/BASE360_metrics_test.json`

## 8. Unified comparison update
- T57b remains an external legacy reference only
- TRAIN360C remains the in-repo self-developed canonical baseline with full pairwise metrics
- BASE360 is now available with real canonical val/test trajectory metrics
- BASE360 pairwise signed-direction / scale-ratio fields remain `N/A` in the unified summary because the official public demo path does not emit native pairwise relative pose outputs

## 9. Next recommendation
- `proceed_to_prepare_main_results_table`

## 10. Compliance checklist
- `cuda_ba_build_attempted = true`
- `cuda_ba_import_passed = true`
- `official_weight_used = true`
- `base360_inference_executed = true`
- `fake_metrics_generated = false`
- `train360_training_executed = false`
- `train360_weights_modified = false`
- `official_method_used_as_teacher = false`
- `random_pair_split_used = false`
- `dset2c_canonical_split_used = true`
- `s5_locked_metrics_modified = false`
