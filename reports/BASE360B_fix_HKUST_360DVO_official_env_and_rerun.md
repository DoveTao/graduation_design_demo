# BASE360B fix HKUST 360DVO official env and rerun

## 1. Executive summary
- `BASE360B execution = blocked_env_install`
- torch/evo/viser in existing `pytorch` env: repaired to importable state
- official `360dvo.pth`: found and hashed
- official `demo.py`: smoke test still fails
- val coverage: `0.00%`
- test coverage: `0.00%`
- comparable to TRAIN360C: `no`
- main blocker after dependency repair: official CUDA extensions were not installed successfully, so runtime stops at `ModuleNotFoundError: No module named cuda_ba`.

## 2. Environment setup
- existing_pytorch_env_checked = `true`
- existing_pytorch_env_usable = `false`
- selected_runtime_env = `blocked`
- checked env name: `pytorch`
- python version: `3.12.12`
- torch version: `2.9.1+cu128`
- CUDA availability: `True`
- evo version: `1.36.4`
- viser version: `1.0.27`
- `pip install .` status in `pytorch` env: failed
- eigen status: `downloaded and unpacked to /tmp/360DVO_official/thirdparty/eigen-3.4.0`
- failure detail: `pip install --no-build-isolation .` advanced to CUDA compilation, but `dpvo/altcorr/correlation_kernel.cu` failed against current Torch headers (`DeprecatedTypeProperties` / old API use).
- official_new_env attempt: `conda env create -f environment.yml` failed to solve under pinned versions; fallback manual `base360dvo` env creation started but was still downloading packages at snapshot time.

## 3. Official weight
- weight path: `/tmp/360DVO_official/360dvo.pth`
- source: official README Google Drive link
- file size: `13573188` bytes
- sha256: `ae4c7d6c96f1fe9eb784174e757b173a80beff8654c7572cde817df95d602d2c`
- load status: `not reached in successful runtime because compiled extensions are missing`

## 4. Smoke test result
- command:
  - `MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU python demo.py --network /tmp/360DVO_official/360dvo.pth --imagedir /home/dovetao/graduation_design_demo/data/360DVO/Sequences/mountains --save_trajectory --name base360b_smoke_mkl`
- stdout path: `/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/smoke/stdout_mkl.log`
- stderr path: `/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/smoke/stderr_mkl.log`
- success/failure: `failure`
- failure reason: `ModuleNotFoundError: No module named cuda_ba`

## 5. Canonical split compliance
- val sequences: `['mountains', 'downhill_biking']`
- test sequences: `['snowmobile', 'ridge_to_lake']`
- pair counts: `val=5342`, `test=5062`
- frame counts: `mountains=765`, `downhill_biking=576`, `snowmobile=719`, `ridge_to_lake=552`
- no random split: `true`
- DSET2C manifest used: `true`
- no split glob: `true` for split definition; raw sequence directories only used after manifest recovery to provide ordered images/GT files

## 6. BASE360 inference results
- per-sequence status: all canonical val/test sequences attempted and blocked before trajectory export
- pred_tum path: `not generated`
- gt_tum path: generated for each sequence under `external_baselines/results/base360_hkust_360dvo_official/{val,test}/<sequence>/gt_tum.txt`
- coverage: `val=0.00%`, `test=0.00%`
- failed frames: `mountains=765`, `downhill_biking=576`, `snowmobile=719`, `ridge_to_lake=552`
- runtime blocker in latest runner attempt: `cuda_ba` missing after import chain enters `dpvo.fastba.ba`

## 7. BASE360 metrics
- val aggregate: `{"split": "val", "sequence_count": 2, "pair_count": 5342, "adjacent_pair_count": 1339, "coverage": 0.0, "ate_none": null, "ate_se3": null, "ate_sim3": null, "path_ratio": null, "predicted_path_length": null, "gt_path_length": null, "rot_mean_deg": null, "rot_median_deg": null, "signed_tdir_mean_deg": null, "signed_tdir_median_deg": null, "unsigned_tdir_mean_deg": null, "anti_parallel_rate": null, "tmag_median_ratio": null, "tmag_mean_ratio": null, "log_tmag_mae": null, "nan_inf_count": null, "execution_status": "blocked", "per_sequence": {"downhill_biking": {"frame_count": 576, "pair_count": 2293, "adjacent_pair_count": 575, "image_dir": "/home/dovetao/graduation_design_demo/data/360DVO/Sequences/downhill_biking", "gt_tum": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/gt_tum.txt", "pred_tum": null, "run_metadata": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/run_metadata.json", "metrics": {"available": false, "coverage": 0.0, "failed_frames": 576, "trajectory_eval": {}, "relative_metrics": {}}}, "mountains": {"frame_count": 765, "pair_count": 3049, "adjacent_pair_count": 764, "image_dir": "/home/dovetao/graduation_design_demo/data/360DVO/Sequences/mountains", "gt_tum": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/val/mountains/gt_tum.txt", "pred_tum": null, "run_metadata": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/val/mountains/run_metadata.json", "metrics": {"available": false, "coverage": 0.0, "failed_frames": 765, "trajectory_eval": {}, "relative_metrics": {}}}}}`
- test aggregate: `{"split": "test", "sequence_count": 2, "pair_count": 5062, "adjacent_pair_count": 1269, "coverage": 0.0, "ate_none": null, "ate_se3": null, "ate_sim3": null, "path_ratio": null, "predicted_path_length": null, "gt_path_length": null, "rot_mean_deg": null, "rot_median_deg": null, "signed_tdir_mean_deg": null, "signed_tdir_median_deg": null, "unsigned_tdir_mean_deg": null, "anti_parallel_rate": null, "tmag_median_ratio": null, "tmag_mean_ratio": null, "log_tmag_mae": null, "nan_inf_count": null, "execution_status": "blocked", "per_sequence": {"ridge_to_lake": {"frame_count": 552, "pair_count": 2197, "adjacent_pair_count": 551, "image_dir": "/home/dovetao/graduation_design_demo/data/360DVO/Sequences/ridge_to_lake", "gt_tum": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/gt_tum.txt", "pred_tum": null, "run_metadata": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/run_metadata.json", "metrics": {"available": false, "coverage": 0.0, "failed_frames": 552, "trajectory_eval": {}, "relative_metrics": {}}}, "snowmobile": {"frame_count": 719, "pair_count": 2865, "adjacent_pair_count": 718, "image_dir": "/home/dovetao/graduation_design_demo/data/360DVO/Sequences/snowmobile", "gt_tum": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/gt_tum.txt", "pred_tum": null, "run_metadata": "/home/dovetao/graduation_design_demo/external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/run_metadata.json", "metrics": {"available": false, "coverage": 0.0, "failed_frames": 719, "trajectory_eval": {}, "relative_metrics": {}}}}}`
- per-sequence metrics remain empty because no `pred_tum.txt` was produced.

## 8. Unified comparison
| model | split | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | ate_none | ate_se3 | ate_sim3 | coverage | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T57b recovered legacy image-pair baseline | external_ref_only | 2.120668 | 111.964932 | 0.674672 | 0.175156 | 0.148787 | 28.308558 | 4.353307 | 2.065264 | 100.00% | Legacy/external reference only; not DSET2C canonical val/test. |
| TRAIN360C self-developed baseline | val | 1.620782 | 111.861396 | 0.713216 | 1.050027 | 0.586027 | N/A | N/A | N/A | 100.00% | Canonical DSET2C val split. |
| TRAIN360C self-developed baseline | test | 3.101081 | 54.956992 | 0.215725 | 0.769285 | 0.588866 | N/A | N/A | N/A | 100.00% | Canonical DSET2C test split. |
| BASE360 HKUST official method | val | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.00% | Torch/evo/viser and official weight fixed in pytorch env, but official CUDA extensions not installed; demo blocks on missing cuda_ba. |
| BASE360 HKUST official method | test | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.00% | Torch/evo/viser and official weight fixed in pytorch env, but official CUDA extensions not installed; demo blocks on missing cuda_ba. |

## 9. Caveats
- current BASE360 output has no trajectory, so scale ambiguity and relative-pose comparability cannot yet be assessed on official outputs.
- `tools/run_base360_hkust_360dvo_official.py` now supports `--official-python`, but successful runtime still depends on compiled `cuda_ba/cuda_corr/lietorch_backends`.
- the official repo appears to target an older Torch C++/CUDA extension API than the existing `pytorch` env provides.
- manual `base360dvo` env creation is still in progress; until it completes, the official-new-env path cannot be validated.
- no fake metrics were generated; BASE360 rows remain `N/A` where predictions do not exist.

## 10. Next recommendation
- `fix_BASE360_remaining_blockers`

## 11. Compliance checklist
- `base360_inference_executed = true`
- `official_method_used_as_teacher = false`
- `train360_training_executed = false`
- `train360_weights_modified = false`
- `s5_locked_metrics_modified = false`
- `random_pair_split_used = false`
- `dset2c_canonical_split_used = true`
- `val_used_for_tuning = false`
- `test_used_for_tuning = false`
- `fake_metrics_generated = false`
