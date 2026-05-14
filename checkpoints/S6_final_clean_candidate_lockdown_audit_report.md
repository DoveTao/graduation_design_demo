# S6 final clean candidate lockdown audit

- final classification: `S5-LOCKED-FINAL-CLEAN-CANDIDATE`
- note: gain is clean but marginal.

## Baseline reproduction audit

- S2b reproduced: drift=`1.327402`, ATE=`7.352371`, path_ratio=`0.934984`
- S5 reproduced: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`
- expected S2b: drift=`1.327402`, ATE=`7.352371`, path_ratio=`0.934984`
- expected S5: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`
- checkpoint load: missing=`14` unexpected=`0`

## Leakage audit

- thresholds_train_only: `True`
- candidate_train_cv_selected: `True`
- final_test_selected_only: `True`
- test_stats_used_for_policy: `False`
- gt_tmag_in_inference_policy: `False`
- hidden_oracle_path: `False`

## Regime-level before/after audit

- `pred_tmag` buckets:
  - pred<q90: pair_pos_err `7.391773` -> `7.391773`, high_error_mass `0.923077` -> `0.923077`, n `39`
  - q90<=pred<q95: pair_pos_err `0.481115` -> `0.481115`, high_error_mass `0.000000` -> `0.000000`, n `1`
- `gt_tmag` buckets (diagnostic only):
  - gt<q90: pair_pos_err `7.498919` -> `7.498919`, high_error_mass `0.916667` -> `0.916667`, n `36`
  - gt>=q95: pair_pos_err `2.543711` -> `2.543711`, high_error_mass `0.500000` -> `0.500000`, n `2`
  - q90<=gt<q95: pair_pos_err `6.855878` -> `6.855878`, high_error_mass `1.000000` -> `1.000000`, n `2`
- `dt` buckets:
  - <0.1: pair_pos_err `9.781988` -> `9.781988`, high_error_mass `0.923077` -> `0.923077`, n `13`
  - [0.1,0.3): pair_pos_err `6.362122` -> `6.362122`, high_error_mass `0.833333` -> `0.833333`, n `12`
  - [0.3,0.5): pair_pos_err `5.943031` -> `5.943031`, high_error_mass `1.000000` -> `1.000000`, n `9`
  - [0.5,1.0): pair_pos_err `5.293614` -> `5.293614`, high_error_mass `0.833333` -> `0.833333`, n `6`
- `k` buckets:
  - k=1: pair_pos_err `7.219007` -> `7.219007`, high_error_mass `0.900000` -> `0.900000`, n `40`
- `dt × k` buckets:
  - <0.1|k=1: pair_pos_err `9.781988` -> `9.781988`, high_error_mass `0.923077` -> `0.923077`, n `13`
  - [0.1,0.3)|k=1: pair_pos_err `6.362122` -> `6.362122`, high_error_mass `0.833333` -> `0.833333`, n `12`
  - [0.3,0.5)|k=1: pair_pos_err `5.943031` -> `5.943031`, high_error_mass `1.000000` -> `1.000000`, n `9`
  - [0.5,1.0)|k=1: pair_pos_err `5.293614` -> `5.293614`, high_error_mass `0.833333` -> `0.833333`, n `6`
- worst bucket `>=1.0|k=20`:

## Scene / chain-level audit

- per-scene (aggregated from scene_seq chains) ATE/drift proxies and per-chain path_ratio/pos_err are audited from `odom_trajectory_debug_latest.json`.
- worst chain before: `('scene01', 'seq03')` drift=`10.142222`
- worst chain after: `('scene01', 'seq03')` drift=`10.141109`
- hard warning: none triggered by drift>0.03 criterion.

## Sensitivity diagnostic (train-only intent)

- proxy metric: CV mean pair magnitude absolute error on train/CV folds only; used only as stability diagnostic, not for candidate reselection.
- q=0.85, mid=1.00, high=0.85, cv_proxy_err=0.364814
- q=0.90, mid=1.00, high=0.85, cv_proxy_err=0.364814
- q=0.95, mid=0.90, high=0.85, cv_proxy_err=0.364814
- q=0.95, mid=0.95, high=0.85, cv_proxy_err=0.364814
- q=0.95, mid=1.00, high=0.85, cv_proxy_err=0.364814
- q=0.85, mid=1.00, high=0.80, cv_proxy_err=0.365619
- q=0.90, mid=1.00, high=0.80, cv_proxy_err=0.365619
- q=0.95, mid=0.90, high=0.80, cv_proxy_err=0.365619

## Final decision

- final classification: `S5-LOCKED-FINAL-CLEAN-CANDIDATE`
- decision note: S5 provides a clean but marginal inference-time calibration gain.
