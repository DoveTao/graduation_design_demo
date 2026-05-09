# S5 External Export Compatibility Audit

## Scope
This report audits whether the current S5 TUM export is compatible with the authoritative locked clean evaluator.
It does not change S5 final-candidate status.

## Locked S5 Reference
- ATE = `7.352288`
- drift = `1.327343`
- path_ratio = `0.932379`

## External Export Summary
- num S5 poses = `41`
- num GT poses = `454`
- matched timestamps = `41`
- tracking_success_rate = `0.090308`
- external none ATE/drift/path_ratio = `8.150968` / `0.490792` / `1.319695`
- external se3 ATE/drift/path_ratio = `1.872819` / `0.352925` / `1.319695`
- external sim3 ATE/drift/path_ratio = `1.497803` / `0.487979` / `1.319695`

## Mismatch Analysis
- path_ratio mismatch: locked = `0.932379` vs external = `1.319695`
- coverage mismatch: external matched poses = `41` vs GT poses = `454`
- coverage ratio = `0.090308`
- GT path length on matched subset = `10.200689`
- GT path length on full exported sequence = `25.833640`
- predicted path length from S5 export = `13.461797`
- path_ratio on matched subset = `1.319695`

## Root Cause Investigation
- export pose count is only `41` because the current exporter follows the official odometry-chain construction over `selected_k=1` pairs, which yielded `40` pairs rather than a full 454-frame stream.
- export contains only chain endpoints / selected pairs / sparse frames: `True`
- timestamp ordering monotonic: `True`
- quaternion ordering used in export: `qx qy qz qw`
- convention uses accumulated camera pose from relative pair predictions: `Relative pair predictions are accumulated in the same odometry-chain convention used by the official evaluator debug path.`
- full trajectory export to 454 timestamps currently available from official clean evaluator: `False`

## Conclusion
The current S5 TUM export is diagnostic-only and should not be used as a full-coverage alignment-consistent comparison against Pano-ORB-VO.
The authoritative S5 result remains the locked clean evaluator metrics.

## Thesis Policy
For thesis reporting, S5 locked metrics and Pano-ORB-VO external trajectory metrics should be reported in clearly separated blocks unless a full-coverage S5 TUM export equivalent to the official evaluator is available.
