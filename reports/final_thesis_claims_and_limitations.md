# Final Thesis Claims and Limitations

## Strong claims
- S1d5 clean dt-anchor policy fixed the critical scale/path_ratio collapse under corrected evaluation scope.
- S2b train-CV selected `fine_rot=0.45` delivered the main clean refinement over S1d5.
- S4/S5 analysis supports magnitude-regime as an important explanatory axis for remaining error behavior.
- S5 achieved a clean, reproducible, inference-time calibration gain and passed S6 lockdown leakage audit.

## Weak claims
- Current residual-head direction (S3a line) has no stable clean gain in the tested setup.
- Fine token representation (S3b diagnostic) shows limited stable support for immediate residual-pose gain.
- dt/k correlates with error, but observed effects are more consistent with tmag-regime-dominant interpretation in the finalized line.
- Fine/spherical token features showed neither stable residual pose signal nor stable added reliability signal in the completed S3b/S8 diagnostic lines.
- Regime-only routing showed some diagnostic signal in S9, but did not remain stable under the clean CV gate because path_ratio moved outside the safe deployable range.
- Post-S5 router/smoother/training attempts did not produce a new clean deployable candidate beyond `S5_clean_tmag_calibration_policy`.
- Training-time tmag consistency in S11 showed weak proxy improvement and slight fold ATE/drift improvement, but lacked path_ratio-supported clean evidence for promotion.
- Regime-balanced sampling in S12 improved some high-risk proxy behavior, but still did not produce stable clean-eligible path_ratio evidence.
- Lightweight pose-graph post-optimization in S14 was insufficient in the current formulation: even with redundant multi-edge constraints, the selected candidate worsened ATE/drift/path_ratio on the final diagnostic.
- Trajectory-level training in S15 is conceptually promising, but the current implementation remained unstable: after the harness parity fix, even the pair-only tiny baseline still collapsed in odometry, so no S15 trajectory result should be claimed as an improvement over S5.
- If practical-ready performance is required, the next step likely needs a stronger trajectory-level formulation, stronger backbone, or data/supervision redesign rather than another lightweight local-window tweak.

## Do-not-claim
- Do not claim S5 is a large performance breakthrough.
- Do not claim residual head is universally ineffective across all possible settings.
- Do not claim fine token is always useless.
- Do not present oracle diagnostic as deployable method.
- Do not claim token reliability router is effective.
- Do not claim regime-only fallback routing is deployable.
- Do not present `R3_dt1_k20_fallback` diagnostic CV numbers as final results.
- Do not claim S9 replaced S5.
- Do not claim S10 chain smoothing is deployable.
- Do not claim S11 tmag consistency training replaced S5.
- Do not present S11 lightweight CV proxy improvements as final clean results.
- Do not claim S12 regime-balanced sampling replaced S5.
- Do not claim S14 lightweight pose graph is deployable.
- Do not present S14 diagnostic graph availability as evidence of final practical readiness.
- Do not claim S15 trajectory-level training improves over S5.
- Do not present S15 proxy improvement alone as evidence of practical trajectory improvement when odometry still worsened.

## Recommended thesis wording
- Preferred: "S5 provides a clean but marginal inference-time calibration gain over S2b."
- Preferred: "Main contribution is a reproducible clean pipeline from scale repair to locked final candidate."
- Preferred: "Later post-lockdown token-reliability, routing, smoothing, sampling, training-time tmag-consistency, lightweight pose-graph, and trajectory-level training attempts were informative diagnostics, but did not yield a new clean deployable replacement."
- Preferred: "Future work on trajectory-level training should redesign the training harness and supervision more fundamentally rather than continuing the current tiny-update route."
- Avoid: any phrasing that implies dramatic end-to-end model redesign gains.
