# BASE360D component metric alignment

## 1. Executive summary
- `BASE360D alignment status = success`
- reliable component metrics were generated from existing BASE360C trajectory outputs only; no official inference rerun, training, or checkpoint modification was performed
- unified comparison files were updated with real BASE360D trajectory-derived component metrics
- maximum caveat: these metrics are derived from trajectory post-processing rather than native image-pair forward output, and the official demo still used a `0.5x` image adapter

## 2. Input inventory
- `val` split:
  - `downhill_biking`: pred=576, gt=576, images=576, manifest_ts=576, alignable=True, exact_match=True, missing_vs_gt=0, duplicate_pred_ts=0, pred_nan_inf=0, adapter_0p5x=True
  - `mountains`: pred=765, gt=765, images=765, manifest_ts=765, alignable=True, exact_match=True, missing_vs_gt=0, duplicate_pred_ts=0, pred_nan_inf=0, adapter_0p5x=True

- `test` split:
  - `ridge_to_lake`: pred=552, gt=552, images=552, manifest_ts=552, alignable=True, exact_match=True, missing_vs_gt=0, duplicate_pred_ts=0, pred_nan_inf=0, adapter_0p5x=True
  - `snowmobile`: pred=719, gt=830, images=719, manifest_ts=719, alignable=True, exact_match=False, missing_vs_gt=111, duplicate_pred_ts=0, pred_nan_inf=0, adapter_0p5x=True

## 3. Pair mapping
- manifest val fields: `R_BA, dataset, image_path_a, image_path_b, k, pair_index, pair_type, seq_id, split, t_BA_B, tdir_B, timestamp_a, timestamp_b, tmag, valid_pose, valid_timestamp`
- manifest test fields: `R_BA, dataset, image_path_a, image_path_b, k, pair_index, pair_type, seq_id, split, t_BA_B, tdir_B, timestamp_a, timestamp_b, tmag, valid_pose, valid_timestamp`
- actual discovered field names: `seq_id`, `image_path_a`, `image_path_b`, `timestamp_a`, `timestamp_b`, `R_BA`, `t_BA_B`, `tdir_B`, `tmag`, `pair_type`, `k`, `split`, `pair_index`
- val unmatched pairs: `0` / `5342`
- test unmatched pairs: `0` / `5062`
- val pair coverage: `1.000000`
- test pair coverage: `1.000000`

## 4. Coordinate convention
- `val` GT-vs-GT sanity: `pass`
- `val` BA: rot_mean=`0.000000`, signed_tdir_mean=`0.000000`
- `val` AB: rot_mean=`1.538834`, signed_tdir_mean=`179.337358`
- `val` selected convention: `BA`
- `val` selection reason: Manifest `R_BA` / `t_BA_B` matches `T_BA = inv(T_B) @ T_A` under GT-vs-GT sanity.
- `test` GT-vs-GT sanity: `pass`
- `test` BA: rot_mean=`0.000000`, signed_tdir_mean=`0.000000`
- `test` AB: rot_mean=`4.634694`, signed_tdir_mean=`177.789690`
- `test` selected convention: `BA`
- `test` selection reason: Manifest `R_BA` / `t_BA_B` matches `T_BA = inv(T_B) @ T_A` under GT-vs-GT sanity.
- identity / same-pose check: rot_error=`0.000000` deg, translation_norm=`0.000000`

## 5. Component metric definitions
- rotation: geodesic angle between predicted relative rotation and manifest GT `R_BA`
- signed tdir: angle between predicted and GT translation vectors after normalization; pairs with near-zero GT translation are counted separately and do not contribute direction angles
- unsigned tdir: absolute-angle variant using `abs(dot(pred_tdir, gt_tdir))`
- anti-parallel rate: fraction of direction-valid pairs with signed tdir angle greater than `90 deg`
- tmag ratios: `pred_tmag / gt_tmag` with epsilon guard
- `log_tmag_mae`: mean absolute difference of log magnitudes with epsilon guard
- `trajectory_path_ratio`: existing sequence-trajectory metric from official BASE360 evaluation
- `pair_component_path_ratio`: sum of per-pair predicted relative translation magnitudes divided by sum of GT magnitudes
- `pose_coverage`: matched predicted poses divided by GT TUM pose count
- `manifest_pose_coverage`: matched predicted poses on manifest timestamps divided by manifest unique timestamp count

## 6. Val results
- `downhill_biking` all-pair: pair_coverage=`1.000000`, signed_tdir_mean=`110.482098`, anti_parallel=`0.686001`, tmag_median_ratio=`0.111364`, pair_component_path_ratio=`0.110411`
- `downhill_biking` adjacent-only: pair_coverage=`1.000000`, signed_tdir_mean=`110.456731`, anti_parallel=`0.686957`, tmag_median_ratio=`0.111574`, pair_component_path_ratio=`0.110386`
- `mountains` all-pair: pair_coverage=`1.000000`, signed_tdir_mean=`103.210291`, anti_parallel=`0.532480`, tmag_median_ratio=`0.416448`, pair_component_path_ratio=`0.409547`
- `mountains` adjacent-only: pair_coverage=`1.000000`, signed_tdir_mean=`103.262655`, anti_parallel=`0.532110`, tmag_median_ratio=`0.415705`, pair_component_path_ratio=`0.409545`
- val aggregate all-pair: rot_mean=`0.766392`, rot_median=`0.399943`, signed_tdir_mean=`106.332225`, signed_tdir_median=`112.478247`, unsigned_tdir_mean=`51.812377`, anti_parallel=`0.598390`, tmag_median_ratio=`0.390480`, tmag_mean_ratio=`0.283682`, log_tmag_mae=`1.450394`, pair_component_path_ratio=`0.166456`, trajectory_path_ratio=`0.166155`
- val aggregate adjacent-only: rot_mean=`0.322704`, signed_tdir_mean=`106.354279`, anti_parallel=`0.598655`, tmag_median_ratio=`0.390273`, pair_component_path_ratio=`0.166155`

## 7. Test results
- `ridge_to_lake` all-pair: pair_coverage=`1.000000`, signed_tdir_mean=`97.748032`, anti_parallel=`0.573964`, tmag_median_ratio=`0.050018`, pair_component_path_ratio=`0.050266`
- `ridge_to_lake` adjacent-only: pair_coverage=`1.000000`, signed_tdir_mean=`97.926023`, anti_parallel=`0.575318`, tmag_median_ratio=`0.050012`, pair_component_path_ratio=`0.050396`
- `snowmobile` all-pair: pair_coverage=`1.000000`, signed_tdir_mean=`151.909746`, anti_parallel=`0.999651`, tmag_median_ratio=`0.034058`, pair_component_path_ratio=`0.034882`
- `snowmobile` adjacent-only: pair_coverage=`1.000000`, signed_tdir_mean=`151.760266`, anti_parallel=`0.998607`, tmag_median_ratio=`0.034148`, pair_component_path_ratio=`0.034876`
- test aggregate all-pair: rot_mean=`0.856182`, rot_median=`0.141955`, signed_tdir_mean=`128.402578`, signed_tdir_median=`145.285520`, unsigned_tdir_mean=`33.729779`, anti_parallel=`0.814895`, tmag_median_ratio=`0.039910`, tmag_mean_ratio=`0.042295`, log_tmag_mae=`3.187603`, pair_component_path_ratio=`0.043542`, trajectory_path_ratio=`0.043618`
- test aggregate adjacent-only: rot_mean=`0.330165`, signed_tdir_mean=`128.385429`, anti_parallel=`0.814815`, tmag_median_ratio=`0.039883`, pair_component_path_ratio=`0.043618`

## 8. Comparison update
- BASE360D vs TRAIN360C on test all-pair metrics: signed_tdir_mean=`128.402578` vs `54.956992`, anti_parallel=`0.814895` vs `0.215725`, tmag_median_ratio=`0.039910` vs `0.769285`, pair_component_path_ratio=`0.043542` vs `0.588866`
- comparable metrics now available: rotation, signed/unsigned translation direction, anti-parallel rate, translation magnitude ratios, log magnitude MAE, pair-component path ratio, pair coverage
- still not fully comparable: BASE360D metrics remain trajectory-derived rather than native pair-forward outputs, and trajectory-level ATE/path metrics are not definition-identical to TRAIN360C pair export metrics

## 9. Caveats
- derived from trajectory output, not direct image-pair model output
- official public demo used a `0.5x` image adapter to avoid full-resolution OOM
- pair-component path ratio and trajectory path ratio are distinct metrics and must not be conflated
- `snowmobile` has `830` GT TUM poses but only `719` manifest/image frames; therefore test `pose_coverage` is `< 1.0` against GT TUM while manifest pair coverage remains `1.0`
- adjacent-only and all-pair metrics are both reported; the unified table uses all-pair numbers for comparability with TRAIN360C

## 10. Next recommendation
- `proceed_to_prepare_thesis_experiment_section`

## 11. Compliance checklist
- `training_executed = false`
- `fine_tune_executed = false`
- `train360_weights_modified = false`
- `base360_inference_rerun = false`
- `base360_used_as_teacher = false`
- `fake_metrics_generated = false`
- `s5_locked_metrics_modified = false`
- `s5e15_included_as_external_baseline = false`
- `dset2c_canonical_split_used = true`
- `random_pair_split_used = false`
- `component_metrics_from_trajectory_disclosed = true`
- `image_adapter_0p5x_disclosed = true`
