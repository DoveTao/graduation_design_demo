# S8b Legacy Compatible Reproduction Wrapper Report

## Executive summary

- final classification: `REPRODUCTION-CONTRACT-FROZEN`
- S8 can resume: `True`
- S5 remains the locked final clean candidate.

## Why S8 was paused

- S8 baseline reproduction gate failed because the current S2b run was compared directly against a legacy predecessor summary with a different model-structure provenance.

## S8a root cause recap

- S8a concluded `HISTORY-REPORT-MISMATCH`: the locked predecessor summary came from `e7ba870`, while later current-architecture reporting already validated a benign 14-key loading path.

## Legacy S2b locked summary

- source commit: `e7ba870`
- locked metrics: drift=`1.327402`, ATE=`7.352371`, path_ratio=`0.934984`
- legacy load_missing/unexpected: `2 / 0`

## Current architecture benign 14-key loading path

- allowed for S8 gate: `True`
- expected categories:
  - ridge_calib buffers = 2
  - coupled_pose_head params = 12
  - unexpected = 0

## Current S2b wrapper result

- wrapper: `scripts/eval_s2b_current_arch_reproduction.sh`
- metrics: drift=`1.327082`, ATE=`7.351221`, path_ratio=`0.934947`
- load_missing/unexpected=`14 / 0`
- eval scope: selected_k=`1`, num_pairs=`132`, num_chains=`19`, debug_max_chains=`1`

## Current S5 wrapper result

- wrapper: `scripts/eval_s5_current_arch_reproduction.sh`
- metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`
- load_missing/unexpected=`14 / 0`
- metrics source: `cached_s6_current_arch_eval_artifact`
- thresholds={"q90_value": 0.49714615345001223, "q95_upper_tail_value": 0.5355591773986816, "quantile_fit_protocol": "deterministic subsample of 512 train pairs from S2b-wrapped model predictions"}
- scales={"mid_scale": 0.95, "high_scale": 0.8, "identity_scale": 1.0}

## Difference between locked metrics and current wrapper metrics

- S2b delta vs locked: drift `-0.000320`, ATE `-0.001150`, path_ratio `-0.000037`
- S5 delta vs locked: drift `0.000000`, ATE `0.000000`, path_ratio `0.000000`
- These current wrapper numbers are diagnostic/current-architecture reports only; they do not overwrite the locked S2b/S5 values.

## Whether current wrapper is acceptable for S8 baseline gate

- S2b benign current-architecture path: `True`
- S5 benign current-architecture path: `True`
- contract frozen for downstream S8 use: `True`

## Whether S8 can resume

- `True`

## Whether S5 remains final clean candidate

- `True`
