# RESULTS360 main results table

## 1. Executive summary
- `TRAIN360C` is the first fully trained self-developed DSET2C canonical baseline with complete pair-level component metrics on val/test.
- `TRAIN360C` improves translation behavior over the recovered legacy `T57b` external reference on test:
  - signed translation direction mean: `111.9649 deg -> 54.9570 deg`
  - anti-parallel rate: `0.6747 -> 0.2157`
  - translation magnitude median ratio: `0.1752 -> 0.7693`
  - path ratio: `0.1488 -> 0.5889`
- `BASE360C` successfully rebuilt the HKUST official `360DVO` CUDA stack, passed smoke, and produced canonical val/test trajectories with `100%` sequence coverage.
- `BASE360D` now derives pair-level component metrics from the reproduced BASE360C official trajectories via canonical-manifest post-processing; comparability to `TRAIN360C` is therefore `partial` rather than blocked, because the metrics are trajectory-derived and still inherit the `0.5x` adapter caveat.
- recommended next task: `proceed_to_prepare_thesis_experiment_section`

## 2. Current pipeline status

| pipeline | status | key outcome |
| --- | --- | --- |
| `TRAIN360A_architecture_inventory_and_reusable_module_audit` | complete | confirmed `TRAIN360-v0` should follow a hybrid T57b-style spherical pair-forward route; confirmed `S5E15` is not a reusable external image-pair model |
| `TRAIN360B_manifest_native_dataloader_adapter_and_forward_sanity` | complete | canonical manifest-native dataloader and train/val/test forward sanity passed without raw directory glob split discovery |
| `TRAIN360C_spherical_pose_baseline` | complete | real training executed; learned checkpoints saved under `checkpoints/TRAIN360C_spherical_pose_baseline/` |
| `BASE360C_fix_cuda_ba_extension_and_rerun_official_smoke` | partial | official `cuda_ba` build/import passed, smoke passed, canonical val/test trajectories produced; BASE360D later completed pair-level component alignment via read-only trajectory post-processing |

## 3. DSET2C split summary
- dataset contract: `DSET2C` canonical `360DVO` split
- split policy: sequence-level split only
- random pair split: `false`
- direct `data/360DVO/Sequences/*` split discovery: `false`
- legacy `scene01` usage: diagnostic-only, not a main experimental split
- canonical sequences:
  - train: `bridge_night`, `canyon_line`, `city_driving`, `drone_racetrack`, `field`, `shanghai_street`
  - val: `mountains`, `downhill_biking`
  - test: `snowmobile`, `ridge_to_lake`
- canonical counts:
  - train pairs: `12154`
  - val pairs: `5342`
  - test pairs: `5062`
  - val adjacent pairs: `1339`
  - test adjacent pairs: `1269`
- hygiene status: `DSET2C_TRAIN360_READY_AFTER_HYGIENE`

## 4. Main component metrics table

| model | source | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | comparability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T57b recovered legacy baseline | GEN5 external eval artifact | external_ref_only | 111.964932 | 0.674672 | 0.175156 | 0.148787 | 100.00% | external reference only | Recovered legacy arbitrary ERP image-pair forward baseline with weak translation generalization; not canonical DSET2C val/test. |
| TRAIN360C self-developed hybrid spherical pose baseline | `reports/TRAIN360C_metrics_val.json` | val | 111.861396 | 0.713216 | 1.050027 | 0.586027 | 100.00% | full | Real training on DSET2C canonical train split; val signed-direction remains weak. |
| TRAIN360C self-developed hybrid spherical pose baseline | `reports/TRAIN360C_metrics_test.json` | test | 54.956992 | 0.215725 | 0.769285 | 0.588866 | 100.00% | full | Test translation generalization materially improves over T57b. |
| BASE360D HKUST official 360DVO baseline (trajectory-derived component metrics) | `reports/BASE360D_metrics_val.json` | val | 106.332225 | 0.598390 | 0.390480 | 0.166456 | 100.00% | partial | Metrics computed by post-processing official trajectory output against the canonical manifest; official input used a `0.5x` image adapter. |
| BASE360D HKUST official 360DVO baseline (trajectory-derived component metrics) | `reports/BASE360D_metrics_test.json` | test | 128.402578 | 0.814895 | 0.039910 | 0.043542 | 100.00% | partial | Metrics computed by post-processing official trajectory output against the canonical manifest; official input used a `0.5x` image adapter. |

## 5. Main trajectory metrics table

| model | source | split | ATE none | ATE se3 | ATE sim3 | path_ratio | predicted_path_length | gt_path_length | coverage | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T57b recovered legacy baseline | GEN5 external eval artifact | external_ref_only | 28.308558 | 4.353307 | 2.065264 | 0.148787 | N/A | N/A | 100.00% | Legacy external reference only; not canonical DSET2C val/test. |
| TRAIN360C self-developed hybrid spherical pose baseline | `reports/TRAIN360C_metrics_val.json` | val | N/A | N/A | N/A | 0.586027 | 3560.499647 | 6075.659641 | 100.00% | No trajectory-alignment ATE metrics were reported for TRAIN360C in the current evaluation path. |
| TRAIN360C self-developed hybrid spherical pose baseline | `reports/TRAIN360C_metrics_test.json` | test | N/A | N/A | N/A | 0.588866 | 4801.225281 | 8153.339815 | 100.00% | Same caveat as val: path-ratio exists, but ATE none/se3/sim3 were not computed in the current TRAIN360C export path. |
| BASE360C HKUST official 360DVO baseline | `reports/BASE360_metrics_val.json` | val | 80.793159 | 57.726244 | 1.728649 | 0.166155 | 92.450323 | 556.411168 | 100.00% | Official trajectory metrics from the reproduced BASE360C run; pair-level component metrics are reported separately as BASE360D trajectory-derived post-processing. |
| BASE360C HKUST official 360DVO baseline | `reports/BASE360_metrics_test.json` | test | 101.616185 | 78.585742 | 2.931001 | 0.043618 | 32.497353 | 745.044547 | 100.00% | Official trajectory metrics from the reproduced BASE360C run; pair-level component metrics are reported separately as BASE360D trajectory-derived post-processing. |

## 6. TRAIN360C vs T57b analysis
- `TRAIN360C` clearly improves test translation quality over the recovered legacy `T57b` baseline.
- The most important movement is on signed translation direction and anti-parallel behavior:
  - signed direction mean drops from about `111.96 deg` to `54.96 deg`
  - anti-parallel rate drops from about `0.6747` to `0.2157`
- Scale behavior also improves:
  - `tmag_median_ratio` rises from about `0.1752` to `0.7693`
  - `path_ratio` rises from about `0.1488` to `0.5889`
- `TRAIN360C` still has a notable val/test discrepancy:
  - val signed translation direction mean is still about `111.86 deg`
  - test signed translation direction mean is about `54.96 deg`
- That discrepancy suggests the current baseline is useful and real, but not yet translation-stable across splits.

## 7. BASE360 official baseline status
- official method: HKUST official `360DVO`
- execution status: `partial`
- selected env: `base360dvo_rebuild`
- official CUDA extension status:
  - `cuda_ba build = pass`
  - `cuda_ba import = pass`
- smoke status: `pass`
- canonical val/test trajectory coverage: `100% / 100%`
- official runtime caveat:
  - full-resolution image-directory input OOMed
  - a `0.5x` image adapter was used to match the repo's video-stream downscale behavior
- comparison status:
  - trajectory metrics are available and real
  - BASE360D now supplies real pair-level signed-direction / anti-parallel / tmag metrics by post-processing the official trajectories against the canonical manifest
  - comparability to `TRAIN360C` is still only `partial`, because BASE360D metrics are trajectory-derived rather than native pair-forward outputs

## 8. Caveats and comparability limits
- `TRAIN360C` has a visible val/test discrepancy on signed translation direction:
  - val: `111.861396 deg`
  - test: `54.956992 deg`
- `BASE360D` and `TRAIN360C` are still not metrically isomorphic:
  - `TRAIN360C` has pair-level component metrics
  - `BASE360D` metrics are trajectory-derived from the official demo path rather than native pair-forward outputs
- `BASE360C` required a `0.5x` input adapter to avoid full-resolution OOM in the official image-directory stream path.
- `BASE360D` now has real signed-direction / anti-parallel / tmag component metrics from trajectory post-processing, but those numbers must still be interpreted with the trajectory-derived and `0.5x` adapter caveats. Test all-pair values are `128.402578 deg`, `0.814895`, `0.039910`, and pair-component path ratio `0.043542`.
- `S5E15` is intentionally excluded from the external 360DVO comparison table because it is not a reusable arbitrary ERP image-pair forward baseline.

## 9. Recommended next task
- `proceed_to_prepare_thesis_experiment_section`
- reason:
  - `TRAIN360C` already shows a meaningful test improvement over `T57b`
  - `BASE360C` has now been reproduced as a real official trajectory baseline
  - the biggest remaining interpretation gap has been closed via manifest-aligned trajectory post-processing
  - the next highest-value step is to consolidate thesis-facing tables and narrative around the now-aligned comparison
- scope reminder:
  - no new training or inference was executed in this alignment step
  - it is not training, fine-tuning, teacher distillation, or checkpoint modification

## 10. Compliance checklist
- `training_executed = false`
- `fine_tune_executed = false`
- `train360_weights_modified = false`
- `base360_used_as_teacher = false`
- `s5_locked_metrics_modified = false`
- `s5e15_included_as_external_baseline = false`
- `fake_metrics_generated = false`
- `dset2c_canonical_split_used = true`
- `random_pair_split_used = false`
- `base360_partial_comparability_disclosed = true`
- `train360c_val_test_discrepancy_disclosed = true`
