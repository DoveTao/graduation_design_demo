# S5 tmag regime-aware calibration report

## Executive summary

- S2b baseline target: drift=`1.327402`, ATE=`7.352371`, path_ratio=`0.934984`
- final classification: `TMAG-CALIBRATION-CLEAN-GAIN`
- selected candidate: `highpred_q90_mid0p95_high0p80`

## Baseline reproduction

- policy: `/home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json`
- base checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- official S2b reference: drift=`1.327402`, ATE=`7.352371`, path_ratio=`0.934984`
- reproduced baseline test: drift=`1.327402`, ATE=`7.352371`, path_ratio=`0.934984`
- train-CV baseline mean drift=`1.363857`, mean ATE=`6.654055`, mean path_ratio=`1.005719`

## Candidate definitions

- `baseline_noop` (noop): clean_eligible=`True` params=`{}`
- `fit_piecewise_clip_0p70_1p10` (pred_piecewise_fit): clean_eligible=`True` params=`{'clip_lo': 0.7, 'clip_hi': 1.1, 'quantiles': [0.0, 0.67, 0.9, 1.0]}`
- `fit_piecewise_clip_0p80_1p05` (pred_piecewise_fit): clean_eligible=`True` params=`{'clip_lo': 0.8, 'clip_hi': 1.05, 'quantiles': [0.0, 0.67, 0.9, 1.0]}`
- `highpred_q80_mid1p00_high0p90` (high_pred_shrink): clean_eligible=`True` params=`{'q_mid': 0.8, 'mid_scale': 1.0, 'high_scale': 0.9}`
- `highpred_q90_mid0p95_high0p80` (high_pred_shrink): clean_eligible=`True` params=`{'q_mid': 0.9, 'mid_scale': 0.95, 'high_scale': 0.8}`
- `interaction_dt1_k20_scale0p85` (dtk_interaction): clean_eligible=`True` params=`{'pred_q': 0.9, 'interaction_scale': 0.85}`
- `winsor_q90` (winsorize): clean_eligible=`True` params=`{'pred_q': 0.9}`
- `oracle_gt_piecewise_clip_0p70_1p10` (oracle_gt_piecewise): clean_eligible=`False` params=`{'clip_lo': 0.7, 'clip_hi': 1.1, 'quantiles': [0.0, 0.67, 0.9, 1.0]}`
- train-side calibration fit used deterministic subsampling with `max_fit_pairs=512` per fold/full-train fit stage; held-out CV eval and final test eval remained full-split.

## Train-CV selection protocol

- folds: `['fold_0_scene01_seq01', 'fold_1_scene01_seq02']`
- no model parameter training
- fit uses train-only labels on each fold train subset
- final selection excludes oracle-only candidates
- hard gates: mean ATE improves over S2b fold baseline, mean drift does not worsen, path_ratio stays within baseline +/- 0.03, worst-fold drift within baseline worst + 0.03, unexpected unchanged

## CV results table

| candidate | family | mean_drift | mean_ATE | mean_path_ratio | mean_pair_pos_err | worst_fold_drift | clean_gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_noop | noop | 1.363857 | 6.654055 | 1.005719 | 0.350118 | 1.608874 | False |
| fit_piecewise_clip_0p70_1p10 | pred_piecewise_fit | 1.286079 | 6.664230 | 0.987515 | 0.338237 | 1.544826 | False |
| fit_piecewise_clip_0p80_1p05 | pred_piecewise_fit | 1.310565 | 6.654230 | 0.988701 | 0.341857 | 1.565124 | False |
| highpred_q80_mid1p00_high0p90 | high_pred_shrink | 1.363857 | 6.654055 | 1.005719 | 0.351048 | 1.608874 | False |
| highpred_q90_mid0p95_high0p80 | high_pred_shrink | 1.363763 | 6.645021 | 0.989624 | 0.354285 | 1.607115 | True |
| interaction_dt1_k20_scale0p85 | dtk_interaction | 1.363857 | 6.654055 | 1.005719 | 0.350124 | 1.608874 | False |
| winsor_q90 | winsorize | 1.365569 | 6.630005 | 0.918020 | 0.365184 | 1.603919 | False |
| oracle_gt_piecewise_clip_0p70_1p10 | oracle_gt_piecewise | 1.285428 | 6.615701 | 0.972201 | 0.328535 | 1.542314 | False |

## Hard-gate audit

- `baseline_noop`: mean_ATE `6.654055` vs baseline `6.654055`, mean_drift `1.363857` vs baseline `1.363857`, mean_path `1.005719` vs baseline `1.005719`, worst_fold_drift `1.608874` vs baseline worst `1.608874`, per_fold_path_ok=`True` -> clean_gate=`False`
- `fit_piecewise_clip_0p70_1p10`: mean_ATE `6.664230` vs baseline `6.654055`, mean_drift `1.286079` vs baseline `1.363857`, mean_path `0.987515` vs baseline `1.005719`, worst_fold_drift `1.544826` vs baseline worst `1.608874`, per_fold_path_ok=`False` -> clean_gate=`False`
- `fit_piecewise_clip_0p80_1p05`: mean_ATE `6.654230` vs baseline `6.654055`, mean_drift `1.310565` vs baseline `1.363857`, mean_path `0.988701` vs baseline `1.005719`, worst_fold_drift `1.565124` vs baseline worst `1.608874`, per_fold_path_ok=`False` -> clean_gate=`False`
- `highpred_q80_mid1p00_high0p90`: mean_ATE `6.654055` vs baseline `6.654055`, mean_drift `1.363857` vs baseline `1.363857`, mean_path `1.005719` vs baseline `1.005719`, worst_fold_drift `1.608874` vs baseline worst `1.608874`, per_fold_path_ok=`True` -> clean_gate=`False`
- `highpred_q90_mid0p95_high0p80`: mean_ATE `6.645021` vs baseline `6.654055`, mean_drift `1.363763` vs baseline `1.363857`, mean_path `0.989624` vs baseline `1.005719`, worst_fold_drift `1.607115` vs baseline worst `1.608874`, per_fold_path_ok=`True` -> clean_gate=`True`
- `interaction_dt1_k20_scale0p85`: mean_ATE `6.654055` vs baseline `6.654055`, mean_drift `1.363857` vs baseline `1.363857`, mean_path `1.005719` vs baseline `1.005719`, worst_fold_drift `1.608874` vs baseline worst `1.608874`, per_fold_path_ok=`True` -> clean_gate=`False`
- `winsor_q90`: mean_ATE `6.630005` vs baseline `6.654055`, mean_drift `1.365569` vs baseline `1.363857`, mean_path `0.918020` vs baseline `1.005719`, worst_fold_drift `1.603919` vs baseline worst `1.608874`, per_fold_path_ok=`False` -> clean_gate=`False`
- `oracle_gt_piecewise_clip_0p70_1p10`: mean_ATE `6.615701` vs baseline `6.654055`, mean_drift `1.285428` vs baseline `1.363857`, mean_path `0.972201` vs baseline `1.005719`, worst_fold_drift `1.542314` vs baseline worst `1.608874`, per_fold_path_ok=`False` -> clean_gate=`False`

## Final test result

- candidate: `highpred_q90_mid0p95_high0p80`
- selected train-fit rule: shrink `pred_tmag >= train q90` by `0.95`, and shrink the upper train `q95+` tail by `0.80`
- drift = `1.327343`
- ATE = `7.352288`
- path_ratio = `0.932379`
- RPE_rot = `20.715353`
- RPE_trans_dir = `72.349483`
- RPE_trans_mag = `0.074128`
- pair_pos_err_mean = `0.348233`
- worst_bucket_pair_pos_err = `2.780795`
- tmag_ratio_mean = `2.165193`

## Regime-level before/after comparison

- baseline worst-fold drift = `1.608874`
- baseline test pair_pos_err_mean = `0.342572`
- baseline test worst_bucket_pair_pos_err = `2.780788`
- baseline test tmag_ratio_mean = `2.175224`
- selected candidate final pair_pos_err_mean = `0.348233`
- selected candidate final worst_bucket_pair_pos_err = `2.780795`
- selected candidate final tmag_ratio_mean = `2.165193`
- test delta vs S2b: drift `-0.000059`, ATE `-0.000082`, path_ratio `-0.002605`
- S4 indicated the dominant removable mass was in high tmag regime; S5 tests whether train-CV-selected inference-time shrinkage/capping can exploit that cleanly.

## Oracle-only diagnostic

- oracle candidate: `oracle_gt_piecewise_clip_0p70_1p10`
- oracle drift = `1.262237`
- oracle ATE = `7.214477`
- oracle path_ratio = `0.898234`
- oracle uses gt_tmag bucket at fit/application time for upper-bound diagnosis only and is not deployment-eligible

## Final classification

- `TMAG-CALIBRATION-CLEAN-GAIN`
- candidate is eligible to replace S2b as the current clean candidate
