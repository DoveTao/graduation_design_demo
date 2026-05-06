# S9 Regime-Only Reliability Router Report

## Executive summary

- final classification: `NO-STABLE-REGIME-ROUTER-GAIN`
- S9 replaces S5: `False`
- selected candidate: `R3_dt1_k20_fallback`

## Motivation from S4/S5/S8

- S4 localized dominant removable error mass to regime structure, especially long-step / larger-k cases.
- S5 showed a clean but marginal gain from fixed pred_tmag-aware shrinkage.
- S8 ruled out added value from token-based reliability features in the current representation.
- S9 therefore tests whether a regime-only router using deployable inference-visible features can improve over S5 cleanly.

## Why token features are excluded

- S8 token-only and token-plus-regime probes did not beat regime-only.
- To avoid feature leakage and unnecessary complexity, S9 restricts routing to `pred_tmag`, `dt`, `k`, and derived regime buckets only.

## Baseline gate using S8b contract

- gate passed: `True`
- contract: `/home/dovetao/graduation_design_demo/checkpoints/S8b_reproduction_contract.json`
- current-architecture benign path accepted: `True`
- locked S5 metrics preserved: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Regime feature definition

| feature | inference-visible | inference-derived-only | allowed |
| --- | --- | --- | --- |
| pred_tmag_s2b | True | True | True |
| dt_world | True | True | True |
| k | True | True | True |
| s5_base_scale | True | True | True |
| pred_tmag_bucket | True | True | True |
| train-derived quantile bucket id | True | True | True |
| fine token / spherical token | True | True | False |
| gt_tmag / gt pose / pair_pos_err | False | False | False |

## Candidate router definitions

- `R1_predq90_conservative` (pred_q90_router): actions=`['A0', 'A1']` params=`{'family': 'pred_q90_router', 'q90': 0.34848379194736484, 'a1_scale': 0.9}`
- `R2_predq90_q95_upper_tail` (pred_upper_tail_router): actions=`['A0', 'A1', 'A2']` params=`{'family': 'pred_upper_tail_router', 'q90': 0.34848379194736484, 'q95': 0.5246053218841553, 'a1_scale': 0.9, 'a2_scale': 0.8}`
- `R4_combined_pred_dt_k` (combined_router): actions=`['A0', 'A1', 'A2']` params=`{'family': 'combined_router', 'q90': 0.34848379194736484, 'a1_scale': 0.9, 'a2_scale': 0.8}`
- `R3_dt1_k20_conservative` (dtk_router): actions=`['A0', 'A1']` params=`{'family': 'dtk_router', 'fallback': False, 'a1_scale': 0.9}`
- `R3_dt1_k20_fallback` (dtk_router): actions=`['A0', 'A3']` params=`{'family': 'dtk_router', 'fallback': True, 'a1_scale': 0.9}`
- `R5_logistic_q80_conservative` (logistic_router): actions=`['A0', 'A1']` params=`{'family': 'logistic_router', 'threshold_quantile': 0.8, 'threshold_score': 1.1506443749530533, 'action_if_risky': 'A1', 'logistic_fit': {'family': 'logistic_router', 'mean': [0.19435999002598692, 0.5002287188763148, 7.6484375, 0.9938476562499996, 0.150390625, 0.193359375], 'std': [0.11754283252443809, 0.8588168427466363, 6.707143863716787, 0.026075315988149283, 0.3574538920086189, 0.39493230698387966], 'weights': [1.4271651110268362, 3.2008289803571817, 0.3856459801148205, 0.3241259216305596, 2.014040731834648, -0.1871870881197589], 'bias': -1.760222114940175, 'train_auc': 0.9907185415529233, 'train_score_quantiles': {'0.80': 1.1506443749530533, '0.85': 3.1752298150261415}, 'train_rows_used': 512}}`

## Train-CV selection protocol

- folds: `scene01/seq01` and `scene01/seq02` leave-one-seq-out CV
- selection uses train-CV only; test is used once for the selected candidate only
- logistic threshold and all quantiles are derived from fold-train rows only
- hard gates require safe path ratio, non-worsened mean CV ATE/drift, bounded worst-fold drift, contract-compatible loading, and no forbidden feature leakage

## CV results table

| candidate | family | cv_mean_ATE | cv_mean_drift | cv_mean_path | pair_pos_err | high-error recall | worst-bucket err | clean_gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R1_predq90_conservative | pred_q90_router | 6.635615 | 1.363504 | 0.973481 | 0.433706 | 0.250172 | 2.951693 | False |
| R2_predq90_q95_upper_tail | pred_upper_tail_router | 6.635615 | 1.363504 | 0.973481 | 0.433706 | 0.250172 | 2.951693 | False |
| R4_combined_pred_dt_k | combined_router | 6.635615 | 1.363504 | 0.973481 | 0.433990 | 0.438367 | 2.959442 | False |
| R3_dt1_k20_conservative | dtk_router | 6.647316 | 1.363353 | 0.994282 | 0.433608 | 0.188195 | 2.959442 | False |
| R3_dt1_k20_fallback | dtk_router | 6.647316 | 1.363353 | 0.994282 | 0.433324 | 0.188195 | 2.951693 | False |
| R5_logistic_q80_conservative | logistic_router | 6.650133 | 1.364060 | 0.991920 | 0.434990 | 0.853271 | 2.959442 | False |

## Hard-gate audit

- `R1_predq90_conservative`: mean_ATE `6.635615` vs `6.645021`, mean_drift `1.363504` vs `1.363763`, mean_path `0.973481`, worst_fold_drift `1.605026` vs baseline worst `1.607115`, worst_bucket_delta `-0.000000`, gate=`False`
- `R2_predq90_q95_upper_tail`: mean_ATE `6.635615` vs `6.645021`, mean_drift `1.363504` vs `1.363763`, mean_path `0.973481`, worst_fold_drift `1.605026` vs baseline worst `1.607115`, worst_bucket_delta `-0.000000`, gate=`False`
- `R4_combined_pred_dt_k`: mean_ATE `6.635615` vs `6.645021`, mean_drift `1.363504` vs `1.363763`, mean_path `0.973481`, worst_fold_drift `1.605026` vs baseline worst `1.607115`, worst_bucket_delta `0.007749`, gate=`False`
- `R3_dt1_k20_conservative`: mean_ATE `6.647316` vs `6.645021`, mean_drift `1.363353` vs `1.363763`, mean_path `0.994282`, worst_fold_drift `1.606352` vs baseline worst `1.607115`, worst_bucket_delta `0.007749`, gate=`False`
- `R3_dt1_k20_fallback`: mean_ATE `6.647316` vs `6.645021`, mean_drift `1.363353` vs `1.363763`, mean_path `0.994282`, worst_fold_drift `1.606352` vs baseline worst `1.607115`, worst_bucket_delta `-0.000000`, gate=`False`
- `R5_logistic_q80_conservative`: mean_ATE `6.650133` vs `6.645021`, mean_drift `1.364060` vs `1.363763`, mean_path `0.991920`, worst_fold_drift `1.607766` vs baseline worst `1.607115`, worst_bucket_delta `0.007749`, gate=`False`

## Final test result, if selected

- no final test was run because no candidate passed train-CV clean selection.

## Regime-level before/after

- not available because no final test candidate was run.

## Scene/chain-level before/after

- not available because no final test candidate was run.

## Leakage audit

- passed: `True`
- token_features_used: `False`
- gt_tmag_as_inference_feature: `False`
- test_used_for_selection: `False`
- train_cv_selection_only: `True`
- forbidden_feature_used: `False`
- current_arch_contract_match: `True`

## Final classification

- `NO-STABLE-REGIME-ROUTER-GAIN`
- S5 remains final clean candidate: `True`
