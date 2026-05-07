# S12 Regime Balanced Sampling Report

## Executive summary

- final classification: `INCONCLUSIVE`
- baseline gate passed: `True`
- run mode: `audit`
- device used: `cuda`
- manifest dir: `/home/dovetao/graduation_design_demo/checkpoints/S12_regime_balanced_sampling_manifests`
- current run status: sampler audit / manifest generation only

## Motivation

- S4 identified tmag-regime-dominant error structure and a worst bucket around `dt>=1.0, k=20`.
- S8/S9/S10/S11 suggest that changing heads, routers, smoothers, or tmag losses alone is unlikely to replace S5.
- S12 therefore shifts to data-centric regime-balanced sampling and curriculum redesign.

## Baseline gate

- contract path: `/home/dovetao/graduation_design_demo/checkpoints/S8b_reproduction_contract.json`
- load_missing / load_unexpected: `14 / 0`
- cited locked S5 metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Candidate sampler families

- `A_baseline_uniform_sampling`: `baseline_uniform_sampling`
- `B_pred_tmag_balanced_sampling`: `pred_tmag_balanced_sampling`
- `C_gt_tmag_balanced_sampling`: `gt_tmag_balanced_sampling`
- `D_dt_k_balanced_sampling`: `dt_k_balanced_sampling`
- `E_hard_regime_oversampling`: `hard_regime_oversampling`
- `F_mixed_balanced_sampling`: `mixed_balanced_sampling`

## Fold sampler distribution summary

### heldout_scene01_seq01

- base rows: `1903`
- pred_q90 threshold: `0.134666`
- `A_baseline_uniform_sampling`: rows=`1903`, unique_pairs=`1903`, high_pred_mass=`191`, high_risk_mass=`91`, mixed_hard_mass=`29`
- `B_pred_tmag_balanced_sampling`: rows=`2285`, unique_pairs=`1903`, high_pred_mass=`573`, high_risk_mass=`111`, mixed_hard_mass=`87`
- `C_gt_tmag_balanced_sampling`: rows=`2285`, unique_pairs=`1903`, high_pred_mass=`239`, high_risk_mass=`257`, mixed_hard_mass=`77`
- `D_dt_k_balanced_sampling`: rows=`4609`, unique_pairs=`1903`, high_pred_mass=`545`, high_risk_mass=`289`, mixed_hard_mass=`107`
- `E_hard_regime_oversampling`: rows=`2276`, unique_pairs=`1903`, high_pred_mass=`402`, high_risk_mass=`283`, mixed_hard_mass=`78`
- `F_mixed_balanced_sampling`: rows=`4747`, unique_pairs=`1903`, high_pred_mass=`683`, high_risk_mass=`298`, mixed_hard_mass=`116`

### heldout_scene01_seq02

- base rows: `2743`
- pred_q90 threshold: `0.147780`
- `A_baseline_uniform_sampling`: rows=`2743`, unique_pairs=`2743`, high_pred_mass=`275`, high_risk_mass=`94`, mixed_hard_mass=`80`
- `B_pred_tmag_balanced_sampling`: rows=`3293`, unique_pairs=`2743`, high_pred_mass=`825`, high_risk_mass=`158`, mixed_hard_mass=`240`
- `C_gt_tmag_balanced_sampling`: rows=`3293`, unique_pairs=`2743`, high_pred_mass=`453`, high_risk_mass=`282`, mixed_hard_mass=`240`
- `D_dt_k_balanced_sampling`: rows=`5744`, unique_pairs=`2743`, high_pred_mass=`1027`, high_risk_mass=`376`, mixed_hard_mass=`320`
- `E_hard_regime_oversampling`: rows=`3206`, unique_pairs=`2743`, high_pred_mass=`614`, high_risk_mass=`314`, mixed_hard_mass=`224`
- `F_mixed_balanced_sampling`: rows=`5792`, unique_pairs=`2743`, high_pred_mass=`1075`, high_risk_mass=`376`, mixed_hard_mass=`320`

## Leakage audit

- train-only labels used for sampling: `True`
- gt_tmag used as inference feature: `False`
- test set used to define buckets or thresholds: `False`

## Final classification

- `INCONCLUSIVE`
