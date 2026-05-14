# S12 Regime Balanced Sampling Report

## Executive summary

- final classification: `NO-STABLE-REGIME-SAMPLING-GAIN`
- baseline gate passed: `True`
- run mode: `cv`
- device used: `cuda`
- training budget: `lightweight two-fold CV, samplers=A/D/E/F, max_steps=100`
- odometry eval run: `True`
- best diagnostic sampler: `D_dt_k_balanced_sampling`
- S12 replaces S5: `False`
- S12b full clean CV recommended: `False`

## Motivation

- S4 identified tmag-regime-dominant error structure and a worst bucket around `dt>=1.0, k=20`.
- S8/S9/S10/S11 suggest that changing heads, routers, smoothers, or tmag losses alone is unlikely to replace S5.
- S12 therefore shifts to data-centric regime-balanced sampling and curriculum redesign.

## Baseline gate using S8b contract

- contract path: `/home/dovetao/graduation_design_demo/checkpoints/S8b_reproduction_contract.json`
- load_missing / load_unexpected: `14 / 0`
- cited locked S5 metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Lightweight two-fold CV setting

- folds: `heldout_scene01_seq01`, `heldout_scene01_seq02`
- samplers run: `A_uniform`, `D_dt_k_balanced`, `E_hard_regime_oversampling`, `F_mixed_balanced`
- updates per sampler per fold: `100`
- batch_size: `1`
- model structure kept fixed; sampler is the main variable.

## Manifest paths used

### A_baseline_uniform_sampling

- `heldout_scene01_seq01`: `/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq01_A_baseline_uniform_sampling.json`
- `heldout_scene01_seq02`: `/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq02_A_baseline_uniform_sampling.json`

### D_dt_k_balanced_sampling

- `heldout_scene01_seq01`: `/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq01_D_dt_k_balanced_sampling.json`
- `heldout_scene01_seq02`: `/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq02_D_dt_k_balanced_sampling.json`

### F_mixed_balanced_sampling

- `heldout_scene01_seq01`: `/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq01_F_mixed_balanced_sampling.json`
- `heldout_scene01_seq02`: `/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq02_F_mixed_balanced_sampling.json`

### E_hard_regime_oversampling

- `heldout_scene01_seq01`: `/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq01_E_hard_regime_oversampling.json`
- `heldout_scene01_seq02`: `/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq02_E_hard_regime_oversampling.json`

## Sampling distribution table

### heldout_scene01_seq01

| sampler | rows | unique_pairs | high_pred_mass | high_risk_mass | mixed_hard_mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_baseline_uniform_sampling | 1903 | 1903 | 191 | 91 | 29 |
| D_dt_k_balanced_sampling | 4609 | 1903 | 545 | 289 | 107 |
| E_hard_regime_oversampling | 2276 | 1903 | 402 | 283 | 78 |
| F_mixed_balanced_sampling | 4747 | 1903 | 683 | 298 | 116 |

### heldout_scene01_seq02

| sampler | rows | unique_pairs | high_pred_mass | high_risk_mass | mixed_hard_mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_baseline_uniform_sampling | 2743 | 2743 | 275 | 94 | 80 |
| D_dt_k_balanced_sampling | 5744 | 2743 | 1027 | 376 | 320 |
| E_hard_regime_oversampling | 3206 | 2743 | 614 | 314 | 224 |
| F_mixed_balanced_sampling | 5792 | 2743 | 1075 | 376 | 320 |

## Fold metrics table

### A_baseline_uniform_sampling

- `heldout_scene01_seq01`: manifest=`/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq01_A_baseline_uniform_sampling.json`, train_loss=`0.775646`, val_tmag_log=`1.828207`, val_speed_log=`1.828207`, val_chain_sum_log=`2.599425`, high_pred_err=`1.810579`, high_risk_err=`3.236576`, dt1_k20_err=`3.236576`, ATE=`5.202232`, drift=`2.351256`, path_ratio=`nan`, missing/unexpected=`14/0`, path_safe=`False`, R/tdir_unchanged=`True`
- `heldout_scene01_seq02`: manifest=`/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq02_A_baseline_uniform_sampling.json`, train_loss=`1.204372`, val_tmag_log=`1.497829`, val_speed_log=`1.497829`, val_chain_sum_log=`1.695834`, high_pred_err=`nan`, high_risk_err=`3.875071`, dt1_k20_err=`3.875071`, ATE=`3.935548`, drift=`2.889164`, path_ratio=`nan`, missing/unexpected=`14/0`, path_safe=`False`, R/tdir_unchanged=`True`

### D_dt_k_balanced_sampling

- `heldout_scene01_seq01`: manifest=`/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq01_D_dt_k_balanced_sampling.json`, train_loss=`0.944185`, val_tmag_log=`2.176791`, val_speed_log=`2.176791`, val_chain_sum_log=`3.124917`, high_pred_err=`2.176791`, high_risk_err=`2.512871`, dt1_k20_err=`2.512871`, ATE=`5.145181`, drift=`2.377594`, path_ratio=`nan`, missing/unexpected=`14/0`, path_safe=`False`, R/tdir_unchanged=`True`
- `heldout_scene01_seq02`: manifest=`/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq02_D_dt_k_balanced_sampling.json`, train_loss=`0.923631`, val_tmag_log=`1.600166`, val_speed_log=`1.600166`, val_chain_sum_log=`2.187178`, high_pred_err=`1.600166`, high_risk_err=`2.975725`, dt1_k20_err=`2.975725`, ATE=`3.859358`, drift=`2.806121`, path_ratio=`nan`, missing/unexpected=`14/0`, path_safe=`False`, R/tdir_unchanged=`True`

### F_mixed_balanced_sampling

- `heldout_scene01_seq01`: manifest=`/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq01_F_mixed_balanced_sampling.json`, train_loss=`0.702999`, val_tmag_log=`2.361119`, val_speed_log=`2.361119`, val_chain_sum_log=`3.388634`, high_pred_err=`2.361119`, high_risk_err=`2.230075`, dt1_k20_err=`2.230075`, ATE=`5.172763`, drift=`2.457642`, path_ratio=`nan`, missing/unexpected=`14/0`, path_safe=`False`, R/tdir_unchanged=`True`
- `heldout_scene01_seq02`: manifest=`/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq02_F_mixed_balanced_sampling.json`, train_loss=`0.910604`, val_tmag_log=`1.517050`, val_speed_log=`1.517050`, val_chain_sum_log=`1.987996`, high_pred_err=`0.694714`, high_risk_err=`3.279809`, dt1_k20_err=`3.279809`, ATE=`3.806110`, drift=`2.774865`, path_ratio=`nan`, missing/unexpected=`14/0`, path_safe=`False`, R/tdir_unchanged=`True`

### E_hard_regime_oversampling

- `heldout_scene01_seq01`: manifest=`/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq01_E_hard_regime_oversampling.json`, train_loss=`1.082102`, val_tmag_log=`1.837198`, val_speed_log=`1.837198`, val_chain_sum_log=`2.630886`, high_pred_err=`1.837198`, high_risk_err=`3.133259`, dt1_k20_err=`3.133259`, ATE=`5.211131`, drift=`2.360743`, path_ratio=`nan`, missing/unexpected=`14/0`, path_safe=`False`, R/tdir_unchanged=`True`
- `heldout_scene01_seq02`: manifest=`/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests/heldout_scene01_seq02_E_hard_regime_oversampling.json`, train_loss=`0.625562`, val_tmag_log=`1.493123`, val_speed_log=`1.493123`, val_chain_sum_log=`1.657090`, high_pred_err=`nan`, high_risk_err=`3.932962`, dt1_k20_err=`3.932962`, ATE=`3.911178`, drift=`2.878732`, path_ratio=`nan`, missing/unexpected=`14/0`, path_safe=`False`, R/tdir_unchanged=`True`

## Hard-gate audit

- `A_baseline_uniform_sampling`: init_ok=`True`, path_safe_all_folds=`False`, r_tdir_safe_all_folds=`True`, high_risk_improved_both_folds=`False`, eligible_for_full_s12b_cv=`False`
- `D_dt_k_balanced_sampling`: init_ok=`True`, path_safe_all_folds=`False`, r_tdir_safe_all_folds=`True`, high_risk_improved_both_folds=`True`, eligible_for_full_s12b_cv=`False`
- `F_mixed_balanced_sampling`: init_ok=`True`, path_safe_all_folds=`False`, r_tdir_safe_all_folds=`True`, high_risk_improved_both_folds=`True`, eligible_for_full_s12b_cv=`False`
- `E_hard_regime_oversampling`: init_ok=`True`, path_safe_all_folds=`False`, r_tdir_safe_all_folds=`True`, high_risk_improved_both_folds=`False`, eligible_for_full_s12b_cv=`False`

## Best diagnostic sampler

- `D_dt_k_balanced_sampling`

## Whether S12b full clean CV is recommended

- `False`

## Whether S12 replaces S5

- `False`

## Leakage audit

- passed: `True`
- test_used_for_selection: `False`
- gt_tmag_as_inference_feature: `False`
- gt_pose_as_inference_feature: `False`
- train_only_regime_labels_used_for_sampling: `True`
- post_processing_router_used: `False`
- train_cv_selection_only: `True`

## Final classification

- `NO-STABLE-REGIME-SAMPLING-GAIN`
- rationale: `sampler improved high-risk proxy but odometry/path safety was not stable enough for promotion`
