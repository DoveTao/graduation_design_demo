# FINAL360M final result verification

## Result

- `FINAL360M final verification executed`: `true`
- `train subset disabled`: `true`
- `train sample count`: `12154`
- `val sample count`: `5342`
- `test sample count`: `5062`
- `epochs completed`: `20`
- `best epoch`: `10`
- `selected by full val`: `true`
- `test used for selection`: `false`
- `selected checkpoint`: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- `existing metrics modified`: `false`
- `checkpoints committed`: `false`
- `current artifact check`: `pass`
- `committed to git`: `false`
- `pushed to remote`: `false`

## Protocol Verification

Verified from:

- `reports/FINAL360M_fulltrain_guarded_protocol.json`
- `reports/FINAL360M_metrics_val.json`
- `reports/FINAL360M_metrics_test.json`
- `reports/FINAL360M_model_selection_table.json`

Required protocol fields:

- `train_subset_disabled = true`
- `full_train_manifest_used = true`
- `train_sample_count = 12154`
- `val_sample_count = 5342`
- `test_sample_count = 5062`
- `selected_by_full_val = true`
- `test_used_for_selection = false`
- `epochs_completed = 20`
- `best_epoch = 10`
- `selected_checkpoint = best_full_val.pt`
- `nonfinite_detected = false`

Judgment:

- `FINAL360M` satisfies the full-train thesis-main protocol.

## FINAL360M Test Metrics

- `rot_mean_deg = 2.322841551013553`
- `rot_median_deg = 0.18974563479423523`
- `signed_tdir_mean_deg = 56.07180781302633`
- `signed_tdir_median_deg = 38.65517540513344`
- `unsigned_tdir_mean_deg = 38.102262497908754`
- `anti_parallel_rate = 0.20762544448834452`
- `tmag_median_ratio = 0.8838564229456283`
- `tmag_mean_ratio = 1.1898223756928266`
- `tmag_ratio_p10 = 0.29401428173601196`
- `tmag_ratio_p90 = 2.684782132949467`
- `log_tmag_mae = 0.6691029947236421`
- `scale_collapse_rate = 0.0`
- `scale_explosion_rate = 0.0`
- `path_ratio = 0.6814703365128686`
- `coverage = 1.0`
- `nan_inf_count = 0`

## Required Answers

1. `FINAL360M` 是否满足 full-train thesis main model 协议？
   - `yes`
2. `FINAL360M` 相比旧 `FINAL360I` 是 `better / worse / mixed`？
   - `mixed`
3. `rot` 是否改善？
   - `yes`
4. `signed_tdir` 是否改善？
   - `no`
5. `anti_parallel` 是否改善？
   - `no`
6. `tmag/path` 是否改善？
   - `yes`
7. 是否推荐 `FINAL360M` 替代旧 `FINAL360I` 成为论文主模型？
   - `yes`, on protocol validity grounds, with explicit note that translation direction worsened while scale/path improved
8. 是否需要更新 `THESIS362` 的最终表格和图？
   - `true`
9. 是否需要重新跑 trajectory-level evaluation？
   - `true`
10. 旧 `FINAL360I` 在论文中应该如何降级表述？
   - as a `subset-trained candidate` / preliminary selection result, not the final full-train thesis main model

## Final Summary

- `FINAL360M final verification executed`: `true`
- `train subset disabled`: `true`
- `train sample count`: `12154`
- `val sample count`: `5342`
- `test sample count`: `5062`
- `epochs completed`: `20`
- `best epoch`: `10`
- `selected by full val`: `true`
- `test used for selection`: `false`
- `selected checkpoint`: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- `FINAL360M test rot_mean`: `2.322841551013553`
- `FINAL360M test signed_tdir_mean`: `56.07180781302633`
- `FINAL360M test anti_parallel_rate`: `0.20762544448834452`
- `FINAL360M test tmag_median_ratio`: `0.8838564229456283`
- `FINAL360M test path_ratio`: `0.6814703365128686`
- `old FINAL360I test rot_mean`: `2.330108616583517`
- `old FINAL360I test signed_tdir_mean`: `45.264702006380205`
- `old FINAL360I test anti_parallel_rate`: `0.20169893322797314`
- `old FINAL360I test tmag_median_ratio`: `0.8344251368086006`
- `old FINAL360I test path_ratio`: `0.6403519796204528`
- `compared to old FINAL360I`: `mixed`
- `thesis main model recommendation`: `FINAL360M_fulltrain_struct360b_thesis_main_guarded`
- `needs THESIS362 table update`: `true`
- `needs trajectory reevaluation`: `true`
- `old FINAL360I thesis status`: `subset-trained candidate only`
- `existing metrics modified`: `false`
- `checkpoints committed`: `false`
- `current artifact check`: `pass`
- `committed to git`: `false`
- `pushed to remote`: `false`
- `recommended next task`: `FINAL360M_trajectory_level_evaluation_and_THESIS362_table_refresh`

