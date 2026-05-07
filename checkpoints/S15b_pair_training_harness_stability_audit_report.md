# S15b Pair Training Harness Stability Audit Report

## Executive summary
- final classification: `POLICY-WRAP-MISMATCH, TRAIN-EVAL-MODE-DRIFT`
- root cause: S15 pair-training harness does not apply the S5/S2b clean wrapper, so it is not training/evaluating the same policy family as the locked final candidate
- S15 can continue without fixes: `False`
- S5 remains final clean candidate: `yes`

## Baseline context
- baseline gate: `passed`
- locked S5 metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Zero-update reload audit
- missing / unexpected: `0 / 0`
- R max abs diff: `0.000000`
- tvec max abs diff: `0.000000`
- tmag abs diff: `0.000000`
- interpretation: plain save/reload is stable; there is no standalone checkpoint serialization bug in this minimal path.

## One-update audit
- lr: `5e-06`
- pair_pose before step: `2.150188`
- pair_tmag before step: `0.153439`
- total before step: `2.165532`
- output delta R_l2: `0.000000`
- output delta tvec_l2: `0.001126`
- output delta tmag_abs: `0.001126`
- coarse delta_l2: `0.002872`
- fine delta_l2: `0.000000`
- interpretation: a direct one-step update in the intended tmag-head-only regime is not catastrophically destructive by itself.

## LR sensitivity audit
- lr=`5e-06`: tmag_abs_delta=`0.001126`, tvec_l2_delta=`0.001126`, pred_tmag_max_before/after=`0.131643`/`0.130363`
- lr=`5.000000000000001e-07`: tmag_abs_delta=`0.000113`, tvec_l2_delta=`0.000113`, pred_tmag_max_before/after=`0.131643`/`0.131514`
- lr=`5.0000000000000004e-08`: tmag_abs_delta=`0.000011`, tvec_l2_delta=`0.000011`, pred_tmag_max_before/after=`0.131643`/`0.131630`
- interpretation: lower LR reduces the already small one-step output drift, but there is no evidence that the current default LR alone explains the catastrophic smoke result.

## Optimizer / freeze audit
- trainable_param_count: `794760`
- trainable_groups: `{'log_tmag_bias': 1, 'coarse': 563334, 'fine': 231425}`
- optimizer_param_count: `794760`
- coupled_pose_head included: `False`
- coarse backbone included: `False`
- fine rotation path included under tmag_head_only: `False`
- tdir path included under tmag_head_only: `False`
- interpretation: the freeze mask is behaving correctly; only the magnitude heads are trainable.

## Pair-loss target / convention audit
- pred_t_frame: `A`
- gt direction norm: `1.000000`
- pose loss value: `1.923778`
- tmag loss value: `0.153740`
- pair-loss target mismatch found: `False`

## Train / eval mode audit
- module counts: `{'BatchNorm': 0, 'Dropout': 17, 'LayerNorm': 37}`
- eval repeat tmag diff: `0.000000`
- train repeat tmag diff: `0.000046`
- eval vs train tmag diff: `0.000022`
- eval vs train tvec diff: `0.038063`
- eval vs train R diff: `0.390395`
- interpretation: the model has dropout and `model.train()` materially changes raw outputs even before any optimizer step, so train/eval mode drift is a real secondary harness issue.

## Policy wrapping audit
- S5 policy path: `/home/dovetao/graduation_design_demo/checkpoints/S5_clean_tmag_calibration_policy.json`
- S5 base policy path: `/home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json`
- training policy enabled: `False`
- training restore enabled: `False`
- training dt-anchor apply: `False`
- mismatch found: `True`
- interpretation: this is the primary bug. The S15 training harness is using the raw base checkpoint path, while the locked final candidate is the S5 wrapper with S2b/S5 policy factors.

## Output scale audit
- before: rot_mean=`21.174194`, tdir_abs_mean=`15.516412`, tmag_log_mean=`1.319835`, pred_tmag_max=`0.131643`
- after current-lr one-step: rot_mean=`21.174194`, tdir_abs_mean=`15.516409`, tmag_log_mean=`1.318083`, pred_tmag_max=`0.130363`
- interpretation: no standalone tmag explosion was reproduced in the clean one-step direct audit.

## Decision
- final classification: `POLICY-WRAP-MISMATCH, TRAIN-EVAL-MODE-DRIFT`
- root cause: S15 pair-training harness does not apply the S5/S2b clean wrapper, so it is not training/evaluating the same policy family as the locked final candidate
- recommendation: do not continue S15 training until the harness is fixed to use the same S5/S2b wrapper path and to control train/eval-mode stochastic drift.
- S5 remains final clean candidate: `yes`
