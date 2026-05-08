# Final Ablation and Baseline Comparison

## Scope
This is an evaluation-only comparison built from existing locked or historical results. It does not run new training, does not search for a new policy, and does not change the final project conclusion.

## Final candidate reminder
- final candidate: `S5_clean_tmag_calibration_policy`
- ATE = `7.352288`
- drift = `1.327343`
- path_ratio = `0.932379`

## Main clean baseline comparison
| method | role | ATE | drift | path_ratio | delta_ATE_vs_S5 | delta_drift_vs_S5 | delta_path_ratio_vs_S5 | final_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1d5_clean_dt_anchor_policy | scale/path_ratio collapse repaired clean baseline | 7.632463 | 1.396358 | 0.934982 | 0.280175 | 0.069015 | 0.002603 | False | first stable clean mainline after scale repair |
| S2b_clean_fine_rot_policy | major clean fine-rotation improvement | 7.352371 | 1.327402 | 0.934984 | 0.000083 | 0.000059 | 0.002605 | False | train-CV selected fine_rot=0.45 policy |
| S5_clean_tmag_calibration_policy | final locked clean candidate | 7.352288 | 1.327343 | 0.932379 | 0.000000 | 0.000000 | 0.000000 | True | clean but marginal gain over S2b |

## Component ablation interpretation
| transition | component_added | delta_ATE | delta_drift | delta_path_ratio | interpretation |
| --- | --- | --- | --- | --- | --- |
| S1d5 -> S2b | fine_rot clean policy | -0.280092 | -0.068956 | 0.000001 | adding fine_rot produced the major clean ATE/drift improvement while preserving safe path_ratio |
| S2b -> S5 | pred_tmag regime-aware calibration | -0.000083 | -0.000059 | -0.002605 | tmag calibration produced only a very small clean gain, but passed train-CV, leakage audit, and lockdown and therefore became the final candidate |

## Oracle diagnostic comparison
| diagnostic | ATE | drift | path_ratio | interpretation |
| --- | --- | --- | --- | --- |
| official_current_S5 | 7.352288 | 1.327343 | 0.932379 | locked clean final candidate under the historical protocol |
| oracle_R | 10.590301 | 1.516962 | 0.934986 | rotation alone does not solve the practical gap and can even worsen global metrics |
| oracle_tdir | 7.658319 | 1.375470 | 0.934985 | translation direction alone does not solve the practical gap |
| oracle_R_tdir | 0.911024 | 0.304305 | 0.934986 | jointly fixing R and tdir collapses most of the gap, supporting the R-TDIR-COUPLED-LIMITED diagnosis |
| oracle_tmag | 7.214477 | 1.262237 | 0.898234 | tmag alone can improve some scalar metrics but path_ratio falls below the safe range, so tmag is not the main bottleneck |

## Negative post-S5 baselines
| experiment | category | final_classification | replaced_S5 | key_reason | thesis_usage |
| --- | --- | --- | --- | --- | --- |
| S8 | post-S5 reliability router | TOKEN-NO-ADDED-VALUE | False | fine/spherical token reliability router did not beat regime-only features or S5 | negative baseline showing token reliability probing is not enough |
| S9 | post-S5 regime router | NO-STABLE-REGIME-ROUTER-GAIN | False | diagnostic regime signal existed but no clean final candidate passed the gate | negative baseline for deployable routing |
| S10 | post-S5 chain smoother | NO-STABLE-CHAIN-SMOOTHER-GAIN | False | chain-level smoothing did not produce a clean train-CV winner | negative baseline for lightweight smoothing |
| S11 | post-S5 tmag consistency training | NO-STABLE-TMAG-CONSISTENCY-GAIN | False | slight proxy improvements but no stable path_ratio evidence | negative baseline for train-time tmag consistency |
| S12 | post-S5 data balancing | NO-STABLE-REGIME-SAMPLING-GAIN | False | regime-balanced sampling still lacked clean-eligible path_ratio evidence | negative baseline for data-centric rebalance |
| S14 | post-S5 pose graph | NO-STABLE-POSE-GRAPH-GAIN | False | lightweight local-window pose graph worsened ATE/drift/path_ratio | negative baseline for lightweight graph post-optimization |
| S15 | post-S5 trajectory training | TRAINING-STILL-UNSTABLE | False | trajectory objective route remained unstable under the current harness | negative baseline for tiny trajectory-level training |
| S16b | post-S5 backbone comparison | NO-STABLE-BACKBONE-FEATURE-GAIN | False | frozen ImageNet ResNet50 features were worse than current task-specific features | negative baseline against generic pretrained backbone features |
| S19 | post-S5 geometry pretraining | NO-STABLE-GEOMETRY-PRETRAINING-GAIN | False | shallow geometry pretraining showed partial signal but no stable coupled R/tdir gain | negative baseline against probe-head-only geometry pretraining |

## Thesis-ready interpretation
- S1d5 fixed the scale/path_ratio collapse and established the first clean mainline baseline.
- S2b provided the major clean gain through fine rotation policy selection.
- S5 provided only a very small but clean, leakage-audited, and locked tmag calibration gain over S2b.
- Oracle diagnostics show that R and tdir must be improved jointly; neither one alone closes the practical gap.
- Post-S5 lightweight modifications and exploratory baselines from S8 to S19 did not replace S5.
- S5 remains the best clean candidate under the historical protocol, but it is not practical-ready.

## Caveats
- evaluation protocol remains the small-sequence historical protocol
- representativeness caveat from S17/S18 still applies
- S5 is not practical-ready
- no deployment-readiness claim is made here
- main bottleneck remains `R-TDIR-COUPLED-LIMITED`

