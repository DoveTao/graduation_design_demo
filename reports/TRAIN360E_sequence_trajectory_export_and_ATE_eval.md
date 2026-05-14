# TRAIN360E sequence trajectory export and ATE eval

## 1. Executive summary
- inference executed: `true`
- training executed: `false`
- checkpoint used: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- selected convention: `BA`
- val coverage: `1.0`
- test coverage: `1.0`
- val ATE none / SE3 / Sim3 RMSE: `304.2349877681277` / `23.34782337155058` / `20.000730606702437`
- test ATE none / SE3 / Sim3 RMSE: `222.56856382785097` / `118.6037794846689` / `27.564661865900444`
- comparison to BASE360D: `comparable=partial`, `better=no`
- main conclusion: `FINAL360I pair-level quality transfers into a valid trajectory export, but sequence-level drift remains a distinct evaluation axis against BASE360D.`

## 2. Data protocol
- canonical split source: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_{val,test}.jsonl`
- val sequences: `mountains`, `downhill_biking`
- test sequences: `snowmobile`, `ridge_to_lake`
- manifest-native recovery: `true`
- random split used: `false`
- direct glob split used: `false`

## 3. Model
- checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- model source: `FINAL360I_struct360b_final_selected`
- output type: `pair-level R_BA / tdir_B / tmag composed into trajectory`
- explicit matching: `false`
- RANSAC / PnP / BA: `false / false / false`
- teacher used: `false`

## 4. Relative pose convention
- selected convention: `BA`
- GT-vs-GT sanity BA mean rot error: `0.004027692669841184`
- GT-vs-GT sanity BA mean tdir error: `5.946313589798715e-07`
- inverse sanity mean rot error: `1.154071471297172`
- inverse sanity mean tdir error: `179.465653132341`
- warnings: `[]`

## 5. Trajectory composition
- adjacent-pair inference only: `true`
- composition formula: `R_{w,b} = R_{w,a} @ R_BA^T`, `t_{w,b} = t_{w,a} - R_{w,b} @ t_BA_B`
- start anchor: `GT first pose aligned start`
- tmag handling: `pred_t = pred_tdir * pred_tmag; non-finite pairs are skipped and recorded`
- val skipped pairs: `0`
- test skipped pairs: `0`

## 6. TUM export
- val/mountains: pred=`/home/dovetao/graduation_design_demo/external_baselines/results/train360e_final360i_trajectory/val/mountains/pred_tum.txt`, gt=`/home/dovetao/graduation_design_demo/external_baselines/results/train360e_final360i_trajectory/val/mountains/gt_tum.txt`
- val/downhill_biking: pred=`/home/dovetao/graduation_design_demo/external_baselines/results/train360e_final360i_trajectory/val/downhill_biking/pred_tum.txt`, gt=`/home/dovetao/graduation_design_demo/external_baselines/results/train360e_final360i_trajectory/val/downhill_biking/gt_tum.txt`
- test/snowmobile: pred=`/home/dovetao/graduation_design_demo/external_baselines/results/train360e_final360i_trajectory/test/snowmobile/pred_tum.txt`, gt=`/home/dovetao/graduation_design_demo/external_baselines/results/train360e_final360i_trajectory/test/snowmobile/gt_tum.txt`
- test/ridge_to_lake: pred=`/home/dovetao/graduation_design_demo/external_baselines/results/train360e_final360i_trajectory/test/ridge_to_lake/pred_tum.txt`, gt=`/home/dovetao/graduation_design_demo/external_baselines/results/train360e_final360i_trajectory/test/ridge_to_lake/gt_tum.txt`
- timestamp alignment: `pred_tum and gt_tum use canonical manifest timestamps`

## 7. Val trajectory results
- aggregate pose coverage: `1.0`
- aggregate pair coverage: `1.0`
- aggregate ATE none / SE3 / Sim3 RMSE: `304.2349877681277` / `23.34782337155058` / `20.000730606702437`
- aggregate trajectory path ratio: `1.5332363998677985`
- mountains: none=`70.73071576313058`, se3=`22.754925500702942`, sim3=`19.668481923126773`, path_ratio=`3.613210088687113`, full_gt_coverage=`1.0`
- downhill_biking: none=`456.994928455989`, se3=`24.112737698434216`, sim3=`20.43364799205653`, path_ratio=`1.0566490611504509`, full_gt_coverage=`1.0`

## 8. Test trajectory results
- aggregate pose coverage: `1.0`
- aggregate pair coverage: `1.0`
- aggregate ATE none / SE3 / Sim3 RMSE: `222.56856382785097` / `118.6037794846689` / `27.564661865900444`
- aggregate trajectory path ratio: `1.756343083453392`
- snowmobile: none=`260.0540005752298`, se3=`128.3745239130996`, sim3=`10.023026961125153`, path_ratio=`2.360466323368215`, full_gt_coverage=`0.8662650602409638`
- ridge_to_lake: none=`161.1588313863587`, se3=`104.51615053911365`, sim3=`40.23229070853109`, path_ratio=`1.2879418776044127`, full_gt_coverage=`1.0`

## 9. Comparison with BASE360D
- val BASE360D ATE none / SE3 / Sim3 RMSE: `94.45665281446705` / `75.94183588649744` / `2.2182811693187587`
- test BASE360D ATE none / SE3 / Sim3 RMSE: `103.25625158853231` / `79.29521397328098` / `2.974465919050361`
- comparable to BASE360D: `partial`
- better than BASE360D on trajectory: `no`
- caveat: `BASE360D` is an official sequence pipeline while `TRAIN360E` is a composed pair-level trajectory export.

## 10. Pair-level vs trajectory-level discussion
- FINAL360I pair-level metrics remain strong on direction/sign and scale/path balance.
- Sequential composition exposes accumulated drift that is not visible in isolated pair metrics.
- ATE evaluates trajectory behavior directly, so it complements but does not replace pair-level summaries.

## 11. Failure / caveat analysis
- drift: sequential drift can dominate long sequences even when adjacent pair direction is reasonable.
- scale error: Sim3-vs-SE3 gap highlights monocular scale instability under trajectory composition.
- rotation accumulation: low pair-level rotation error can still integrate into path deviation over long chains.
- sequence difficulty: snowmobile keeps a manifest/full-GT coverage distinction and should be read with that denominator in mind.
- val/test discrepancy: sequence-level behavior should be interpreted separately from FINAL360I pair-level val/test discrepancy.

## 12. Next recommendation
- `prepare_thesis_experiment_section`

## 13. Compliance checklist
- `training_executed = false`
- `fine_tune_executed = false`
- `learned_weights_modified = false`
- `final360i_checkpoint_modified = false`
- `explicit_matching_used = false`
- `match_list_output = false`
- `ransac_used = false`
- `pnp_used = false`
- `bundle_adjustment_used = false`
- `hkust_360dvo_teacher_used = false`
- `base360_outputs_used_as_training_input = false`
- `dset2c_canonical_split_used = true`
- `random_pair_split_used = false`
- `direct_glob_data_360dvo_sequences = false`
- `s5_locked_metrics_modified = false`
- `large_checkpoints_committed_to_git = false`
