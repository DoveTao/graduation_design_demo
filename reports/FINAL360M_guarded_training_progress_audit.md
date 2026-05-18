# FINAL360M guarded training progress audit

## Status

- `progress audit executed`: `true`
- `FINAL360M guarded process alive`: `false`
- `elapsed_time`: `completed`
- `current_epoch`: `20`
- `current_update`: `unknown`
- `recent loss finite`: `true`
- `recent grad finite`: `unknown`
- `nonfinite detected after guarded restart`: `false`
- `full val executed`: `true`
- `best full-val checkpoint exists`: `true`
- `full test metrics generated`: `true`
- `disk free`: `7.9G`
- `recommended action`: `final_evaluate`

## Process

- `process_alive = false`
- `training_status = completed`
- `ps -p 18450` returned no live process line during this audit

## Log Visibility

Current guarded log shows full completion through:

- epochs `1` through `20`
- full-val runs at epochs `5`, `10`, `15`, `20`
- best full-val checkpoint updates at epochs `5` and `10`
- final selected checkpoint:
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- final log line indicates:
  - `selected checkpoint ... best_epoch=10 test_signed_tdir=56.07180781302633 test_anti=0.20762544448834452`
  - `finished classification=thesis_main_model`

## Checkpoint Directory

Path:

- `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded`

Current observed contents:

- `best_full_val.pt`
- `candidate_epoch_10.pt`
- `candidate_epoch_15.pt`
- `candidate_epoch_19.pt`
- `config.yaml`
- `last.pt`
- `train_log.jsonl`

Current size:

- `719M`

## Metrics / Selection Artifacts

Present:

- `reports/FINAL360M_fulltrain_guarded_metrics_val.json`
- `reports/FINAL360M_fulltrain_guarded_metrics_test.json`
- `reports/FINAL360M_fulltrain_guarded_model_selection_table.json`
- `reports/FINAL360M_fulltrain_guarded_protocol.json`

Not present:

- `reports/FINAL360M_metrics_val.json`
- `reports/FINAL360M_metrics_test.json`
- `reports/FINAL360M_model_selection_table.json`

Interpretation:

- the guarded run wrote its own guarded-named outputs rather than the old non-guarded names.
- full validation and final test both executed under the guarded naming scheme.

## Parsed Recent Training Rows

`train_log.jsonl` exists and currently contains `20` rows.

Latest parsed values:

- `latest epoch = 20`
- `latest update = unknown`
- `latest train_loss_total = 0.19977737911710214`
- `latest loss_rot = 0.01935336256374617`
- `latest loss_tdir = 0.0024645835187553774`
- `latest loss_tmag = 0.18615497406691844`
- `latest grad_norm = null`
- `train_nan_inf_count = 0`
- `mini_val_score = 168.08821424095544`
- `full_val_score = 166.32906043512162`

Recent finite-status judgment:

- `loss_finite_recently = true`
- `grad_finite_recently = unknown`
  - no explicit `grad_norm` field is logged in `train_log.jsonl`
- `nonfinite_count_recent50 = 0`
- `nonfinite_after_restart = false`

## Disk

- current disk free: `7.9G`
- disk is slightly below the earlier `8G` comfort threshold, but the guarded training already finished
- this is now more of a cleanup / archiving concern than a training-survival concern

## Required Answers

- `process_alive`: `false`
- `elapsed_time`: `completed`
- `current_epoch`: `20`
- `current_update`: `unknown`
- `loss_finite_recently`: `true`
- `grad_finite_recently`: `unknown`
- `nonfinite_after_restart`: `false`
- `current_disk_free`: `7.9G`
- `full_val_executed`: `true`
- `best_full_val_exists`: `true`
- `test_executed`: `true`
- `training_status`: `completed`
- `recommended_action`: `final_evaluate`

## Final Summary

- `progress audit executed`: `true`
- `FINAL360M guarded process alive`: `false`
- `current epoch`: `20`
- `current update`: `unknown`
- `recent loss finite`: `true`
- `recent grad finite`: `unknown`
- `nonfinite detected after guarded restart`: `false`
- `full val executed`: `true`
- `best full-val checkpoint exists`: `true`
- `full test metrics generated`: `true`
- `disk free`: `7.9G`
- `recommended action`: `final_evaluate`

