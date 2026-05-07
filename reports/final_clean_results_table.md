# Final Clean Results Table

## Consolidated table

| baseline_or_candidate | drift | ATE | path_ratio | status | clean | notes |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Raw T57b multiscale (no clean policy) | 1.494000 | 9.644000 | 0.496000 | historical raw baseline | no | path_ratio collapse reference point |
| S1d5 clean dt-anchor policy | 1.396358 | 7.632463 | 0.934982 | scale/path_ratio repaired | yes | first stable clean mainline |
| S2b clean fine-rot policy | 1.327402 | 7.352371 | 0.934984 | major clean refinement | yes | train-CV selected fine_rot=0.45 |
| S5 final clean tmag calibration | 1.327343 | 7.352288 | 0.932379 | final clean candidate | yes | clean but marginal gain over S2b |
| S3a residual-head line | N/A | N/A | N/A | negative (not selected) | no | no legal train-CV selected replacement |
| S3b fine-token diagnostic | N/A | N/A | N/A | negative/diagnostic | no | no stable residual signal for replacement |
| S8 fine/spherical reliability router | N/A | N/A | N/A | diagnostic negative | no | `TOKEN-NO-ADDED-VALUE`; token-only and token+regime did not beat regime-only |
| S9 regime-only reliability router | N/A | N/A | N/A | diagnostic negative | no | `NO-STABLE-REGIME-ROUTER-GAIN`; no clean gate pass, final test not run |
| S10 chain-level path-ratio smoother | N/A | N/A | N/A | diagnostic negative | no | `NO-STABLE-CHAIN-SMOOTHER-GAIN`; no stable clean gain over S5 |
| S11 tmag scale consistency training | N/A | N/A | N/A | diagnostic negative | no | `NO-STABLE-TMAG-CONSISTENCY-GAIN`; weak proxy gain but no stable path_ratio-supported clean evidence |
| S12 regime-balanced sampling | N/A | N/A | N/A | diagnostic negative | no | `NO-STABLE-REGIME-SAMPLING-GAIN`; improved some high-risk proxies but no clean-eligible path_ratio evidence |
| S14 local-window pose graph diagnostic | N/A | N/A | N/A | diagnostic negative | no | `NO-STABLE-POSE-GRAPH-GAIN`; final test run, selected `D_joint_w7_s1` worsened ATE/drift/path_ratio despite redundant graph constraints |
| Oracle-only tmag diagnostic | 1.262237 | 7.214477 | 0.898234 | upper-bound diagnostic only | no | not deployable, uses oracle-style gt_tmag path |

## Reading guide
- S1d5 is the decisive path-ratio repair stage.
- S2b provides the main clean performance improvement.
- S5 is the final deployable candidate, but improvement over S2b is intentionally described as marginal.
- S8/S9/S10/S11/S12/S14 are post-lockdown or post-S5 negative diagnostics and must not be presented as deployable replacements.
- Oracle numbers are for diagnostic ceiling only and must not be reported as a deployable method.
