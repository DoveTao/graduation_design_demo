# S10 Chain-Level Path-Ratio Preserving Smoother Report

## Executive summary

- final classification: `NO-STABLE-CHAIN-SMOOTHER-GAIN`
- S10 replaces S5: `False`
- selected candidate: `None`
- official odom scope selected_k: `1`

## Motivation

- S8 ruled out added reliability value from token features.
- S9 suggested that local fallback-like behaviors can show diagnostic signal but tend to disturb global path ratio.
- S10 therefore moves from pair-level routing to chain-level tmag smoothing with explicit chain-sum preservation.

## Baseline gate using S8b contract

- gate passed: `True`
- contract: `/home/dovetao/graduation_design_demo/checkpoints/S8b_reproduction_contract.json`
- locked S5 metrics preserved: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Inference-visible inputs

| feature | inference-visible | allowed |
| --- | --- | --- |
| R_i from S5 | True | True |
| tdir_i from S5 | True | True |
| tmag_i from S5 | True | True |
| pred_tmag_i | True | True |
| dt_i | True | True |
| k_i | True | True |
| S5 bucket id | True | True |
| fine token / spherical token | True | False |
| gt_tmag / gt pose / pair_pos_err | False | False |

## Candidate families

- local log-tmag rolling median blend
- rolling MAD spike clipping
- regime-aware spike clipping
- conservative hybrid blend+clip
- renorm mode includes `preserve_chain_sum` and a few `none` controls
- note: official odometry clean scope uses `selected_k=1`, so `k==20` trigger paths remain diagnostic-only and are effectively inactive under final odom evaluation.

## Train-CV selection protocol

- folds: `scene01/seq01` and `scene01/seq02`
- no model training; only chain-level inference-time post-processing of S5 pair outputs
- train-only quantiles are used for regime-aware triggers
- final test is allowed only for a train-CV clean-gate candidate

## CV results table

| candidate | family | cv_mean_ATE | cv_mean_drift | cv_mean_path | pair_pos_err | changed_pct | max_renorm | clean_gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hybrid_w7_predq90ordt1_a0p50_preserve | hybrid_risk_blend | 6.496720 | 1.349646 | 0.994282 | 0.081223 | 0.734035 | 1.109929 | False |
| spike_w7_m2p5_c1p0_preserve | spike_clip | 6.528695 | 1.355933 | 0.994282 | 0.086067 | 0.777957 | 1.145792 | False |
| spike_w7_m3p0_c1p0_preserve | spike_clip | 6.528695 | 1.355933 | 0.994282 | 0.086067 | 0.777957 | 1.145792 | False |
| spike_w7_m3p5_c1p0_preserve | spike_clip | 6.528695 | 1.355929 | 0.994282 | 0.086067 | 0.752151 | 1.145792 | False |
| hybrid_w5_predq90ordt1_a0p50_preserve | hybrid_risk_blend | 6.530454 | 1.351198 | 0.994282 | 0.077986 | 0.734035 | 1.084811 | False |
| median_w7_a0p75_preserve | median_blend | 6.546204 | 1.356099 | 0.994282 | 0.087702 | 0.945605 | 1.057378 | False |
| regime_spike_pred_q90_preserve | regime_spike_clip | 6.549676 | 1.351510 | 0.994282 | 0.076132 | 0.524357 | 1.065670 | False |
| regime_spike_pred_q90_or_dt_ge_1_preserve | regime_spike_clip | 6.549676 | 1.351510 | 0.994282 | 0.076132 | 0.524357 | 1.065670 | False |
| spike_w7_m2p5_c0p75_preserve | spike_clip | 6.549707 | 1.357138 | 0.994282 | 0.082400 | 0.777957 | 1.113704 | False |
| spike_w7_m3p0_c0p75_preserve | spike_clip | 6.549707 | 1.357138 | 0.994282 | 0.082400 | 0.777957 | 1.113704 | False |
| spike_w7_m3p5_c0p75_preserve | spike_clip | 6.549707 | 1.357135 | 0.994282 | 0.082400 | 0.752151 | 1.113704 | False |
| regime_spike_predq90_none | regime_spike_clip | 6.554213 | 1.352074 | 0.957562 | 0.076305 | 0.035835 | 1.000000 | False |
| median_w7_a0p5_preserve | median_blend | 6.570591 | 1.357766 | 0.994282 | 0.080526 | 0.945605 | 1.045979 | False |
| spike_w7_m2p5_c0p5_preserve | spike_clip | 6.575784 | 1.358722 | 0.994282 | 0.078510 | 0.777957 | 1.078823 | False |
| spike_w7_m3p0_c0p5_preserve | spike_clip | 6.575784 | 1.358722 | 0.994282 | 0.078510 | 0.777957 | 1.078823 | False |
| spike_w7_m3p5_c0p5_preserve | spike_clip | 6.575784 | 1.358720 | 0.994282 | 0.078510 | 0.752151 | 1.078823 | False |
| median_w7_a0p25_preserve | median_blend | 6.604034 | 1.360163 | 0.994282 | 0.075167 | 0.945605 | 1.026722 | False |
| spike_w5_m3p5_c1p0_preserve | spike_clip | 6.605648 | 1.358558 | 0.994282 | 0.080569 | 0.759841 | 1.105469 | False |
| spike_w5_m2p5_c1p0_preserve | spike_clip | 6.605665 | 1.358512 | 0.994282 | 0.080560 | 0.759841 | 1.105469 | False |
| spike_w5_m3p0_c1p0_preserve | spike_clip | 6.605665 | 1.358512 | 0.994282 | 0.080560 | 0.759841 | 1.105469 | False |
| spike_w5_m3p5_c0p75_preserve | spike_clip | 6.614016 | 1.359638 | 0.994282 | 0.078300 | 0.759841 | 1.083486 | False |
| spike_w5_m2p5_c0p75_preserve | spike_clip | 6.614026 | 1.359603 | 0.994282 | 0.078293 | 0.759841 | 1.083486 | False |
| spike_w5_m3p0_c0p75_preserve | spike_clip | 6.614026 | 1.359603 | 0.994282 | 0.078293 | 0.759841 | 1.083486 | False |
| spike_w5_m3p0_c0p75_none | spike_clip | 6.616433 | 1.362857 | 0.935196 | 0.079089 | 0.082188 | 1.000000 | False |
| median_w5_a0p75_preserve | median_blend | 6.618034 | 1.364653 | 0.994282 | 0.081447 | 0.945605 | 1.059656 | False |
| spike_w5_m3p5_c0p5_preserve | spike_clip | 6.623619 | 1.360796 | 0.994282 | 0.075837 | 0.759841 | 1.058807 | False |
| spike_w5_m2p5_c0p5_preserve | spike_clip | 6.623623 | 1.360773 | 0.994282 | 0.075832 | 0.759841 | 1.058807 | False |
| spike_w5_m3p0_c0p5_preserve | spike_clip | 6.623623 | 1.360773 | 0.994282 | 0.075832 | 0.759841 | 1.058807 | False |
| median_w5_a0p5_preserve | median_blend | 6.624435 | 1.363890 | 0.994282 | 0.077429 | 0.945605 | 1.045750 | False |
| median_w5_a0p50_none | median_blend | 6.625606 | 1.365641 | 0.960643 | 0.078167 | 0.738102 | 1.000000 | False |
| median_w5_a0p25_preserve | median_blend | 6.634176 | 1.363462 | 0.994282 | 0.074190 | 0.945605 | 1.025868 | False |
| regime_spike_dt_ge_1_preserve | regime_spike_clip | 6.647316 | 1.363353 | 0.994282 | 0.073168 | 0.000000 | 1.000000 | False |
| regime_spike_pred_q90_and_dt_ge_1_preserve | regime_spike_clip | 6.647316 | 1.363353 | 0.994282 | 0.073168 | 0.000000 | 1.000000 | False |
| median_w3_a0p25_preserve | median_blend | 6.662233 | 1.367278 | 0.994282 | 0.076025 | 0.939154 | 1.027802 | False |
| median_w3_a0p5_preserve | median_blend | 6.681804 | 1.370907 | 0.994282 | 0.081447 | 0.939154 | 1.047074 | False |
| median_w3_a0p75_preserve | median_blend | 6.705627 | 1.374225 | 0.994282 | 0.089188 | 0.945605 | 1.057892 | False |

## Hard-gate audit

- `hybrid_w7_predq90ordt1_a0p50_preserve`: mean_ATE `6.496720` vs baseline `6.647316`, mean_drift `1.349646` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.734035`, max_renorm `1.109929`, gate=`False`
- `spike_w7_m2p5_c1p0_preserve`: mean_ATE `6.528695` vs baseline `6.647316`, mean_drift `1.355933` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.777957`, max_renorm `1.145792`, gate=`False`
- `spike_w7_m3p0_c1p0_preserve`: mean_ATE `6.528695` vs baseline `6.647316`, mean_drift `1.355933` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.777957`, max_renorm `1.145792`, gate=`False`
- `spike_w7_m3p5_c1p0_preserve`: mean_ATE `6.528695` vs baseline `6.647316`, mean_drift `1.355929` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.752151`, max_renorm `1.145792`, gate=`False`
- `hybrid_w5_predq90ordt1_a0p50_preserve`: mean_ATE `6.530454` vs baseline `6.647316`, mean_drift `1.351198` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.734035`, max_renorm `1.084811`, gate=`False`
- `median_w7_a0p75_preserve`: mean_ATE `6.546204` vs baseline `6.647316`, mean_drift `1.356099` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.945605`, max_renorm `1.057378`, gate=`False`
- `regime_spike_pred_q90_preserve`: mean_ATE `6.549676` vs baseline `6.647316`, mean_drift `1.351510` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.524357`, max_renorm `1.065670`, gate=`False`
- `regime_spike_pred_q90_or_dt_ge_1_preserve`: mean_ATE `6.549676` vs baseline `6.647316`, mean_drift `1.351510` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.524357`, max_renorm `1.065670`, gate=`False`
- `spike_w7_m2p5_c0p75_preserve`: mean_ATE `6.549707` vs baseline `6.647316`, mean_drift `1.357138` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.777957`, max_renorm `1.113704`, gate=`False`
- `spike_w7_m3p0_c0p75_preserve`: mean_ATE `6.549707` vs baseline `6.647316`, mean_drift `1.357138` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.777957`, max_renorm `1.113704`, gate=`False`
- `spike_w7_m3p5_c0p75_preserve`: mean_ATE `6.549707` vs baseline `6.647316`, mean_drift `1.357135` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.752151`, max_renorm `1.113704`, gate=`False`
- `regime_spike_predq90_none`: mean_ATE `6.554213` vs baseline `6.647316`, mean_drift `1.352074` vs baseline `1.363353`, mean_path `0.957562`, dist_to_locked_s5 `0.025183`, changed_pct `0.035835`, max_renorm `1.000000`, gate=`False`
- `median_w7_a0p5_preserve`: mean_ATE `6.570591` vs baseline `6.647316`, mean_drift `1.357766` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.945605`, max_renorm `1.045979`, gate=`False`
- `spike_w7_m2p5_c0p5_preserve`: mean_ATE `6.575784` vs baseline `6.647316`, mean_drift `1.358722` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.777957`, max_renorm `1.078823`, gate=`False`
- `spike_w7_m3p0_c0p5_preserve`: mean_ATE `6.575784` vs baseline `6.647316`, mean_drift `1.358722` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.777957`, max_renorm `1.078823`, gate=`False`
- `spike_w7_m3p5_c0p5_preserve`: mean_ATE `6.575784` vs baseline `6.647316`, mean_drift `1.358720` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.752151`, max_renorm `1.078823`, gate=`False`
- `median_w7_a0p25_preserve`: mean_ATE `6.604034` vs baseline `6.647316`, mean_drift `1.360163` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.945605`, max_renorm `1.026722`, gate=`False`
- `spike_w5_m3p5_c1p0_preserve`: mean_ATE `6.605648` vs baseline `6.647316`, mean_drift `1.358558` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.759841`, max_renorm `1.105469`, gate=`False`
- `spike_w5_m2p5_c1p0_preserve`: mean_ATE `6.605665` vs baseline `6.647316`, mean_drift `1.358512` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.759841`, max_renorm `1.105469`, gate=`False`
- `spike_w5_m3p0_c1p0_preserve`: mean_ATE `6.605665` vs baseline `6.647316`, mean_drift `1.358512` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.759841`, max_renorm `1.105469`, gate=`False`
- `spike_w5_m3p5_c0p75_preserve`: mean_ATE `6.614016` vs baseline `6.647316`, mean_drift `1.359638` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.759841`, max_renorm `1.083486`, gate=`False`
- `spike_w5_m2p5_c0p75_preserve`: mean_ATE `6.614026` vs baseline `6.647316`, mean_drift `1.359603` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.759841`, max_renorm `1.083486`, gate=`False`
- `spike_w5_m3p0_c0p75_preserve`: mean_ATE `6.614026` vs baseline `6.647316`, mean_drift `1.359603` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.759841`, max_renorm `1.083486`, gate=`False`
- `spike_w5_m3p0_c0p75_none`: mean_ATE `6.616433` vs baseline `6.647316`, mean_drift `1.362857` vs baseline `1.363353`, mean_path `0.935196`, dist_to_locked_s5 `0.002817`, changed_pct `0.082188`, max_renorm `1.000000`, gate=`False`
- `median_w5_a0p75_preserve`: mean_ATE `6.618034` vs baseline `6.647316`, mean_drift `1.364653` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.945605`, max_renorm `1.059656`, gate=`False`
- `spike_w5_m3p5_c0p5_preserve`: mean_ATE `6.623619` vs baseline `6.647316`, mean_drift `1.360796` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.759841`, max_renorm `1.058807`, gate=`False`
- `spike_w5_m2p5_c0p5_preserve`: mean_ATE `6.623623` vs baseline `6.647316`, mean_drift `1.360773` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.759841`, max_renorm `1.058807`, gate=`False`
- `spike_w5_m3p0_c0p5_preserve`: mean_ATE `6.623623` vs baseline `6.647316`, mean_drift `1.360773` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.759841`, max_renorm `1.058807`, gate=`False`
- `median_w5_a0p5_preserve`: mean_ATE `6.624435` vs baseline `6.647316`, mean_drift `1.363890` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.945605`, max_renorm `1.045750`, gate=`False`
- `median_w5_a0p50_none`: mean_ATE `6.625606` vs baseline `6.647316`, mean_drift `1.365641` vs baseline `1.363353`, mean_path `0.960643`, dist_to_locked_s5 `0.028264`, changed_pct `0.738102`, max_renorm `1.000000`, gate=`False`
- `median_w5_a0p25_preserve`: mean_ATE `6.634176` vs baseline `6.647316`, mean_drift `1.363462` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.945605`, max_renorm `1.025868`, gate=`False`
- `regime_spike_dt_ge_1_preserve`: mean_ATE `6.647316` vs baseline `6.647316`, mean_drift `1.363353` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.000000`, max_renorm `1.000000`, gate=`False`
- `regime_spike_pred_q90_and_dt_ge_1_preserve`: mean_ATE `6.647316` vs baseline `6.647316`, mean_drift `1.363353` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.000000`, max_renorm `1.000000`, gate=`False`
- `median_w3_a0p25_preserve`: mean_ATE `6.662233` vs baseline `6.647316`, mean_drift `1.367278` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.939154`, max_renorm `1.027802`, gate=`False`
- `median_w3_a0p5_preserve`: mean_ATE `6.681804` vs baseline `6.647316`, mean_drift `1.370907` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.939154`, max_renorm `1.047074`, gate=`False`
- `median_w3_a0p75_preserve`: mean_ATE `6.705627` vs baseline `6.647316`, mean_drift `1.374225` vs baseline `1.363353`, mean_path `0.994282`, dist_to_locked_s5 `0.061903`, changed_pct `0.945605`, max_renorm `1.057892`, gate=`False`

## Final test result, if selected

- no final test was run because no candidate passed the train-CV clean gate.

## Regime-level before/after

- not available because no final test candidate was run.

## Scene/chain-level before/after

- not available because no final test candidate was run.

## Leakage audit

- passed: `True`
- token_features_used: `False`
- gt_tmag_as_inference_feature: `False`
- gt_pose_as_inference_feature: `False`
- test_used_for_selection: `False`
- train_cv_selection_only: `True`
- forbidden_feature_used: `False`
- current_arch_contract_match: `True`

## Final classification

- `NO-STABLE-CHAIN-SMOOTHER-GAIN`
- S5 remains final clean candidate: `True`
