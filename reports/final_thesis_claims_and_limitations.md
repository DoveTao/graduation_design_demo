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

## Do-not-claim
- Do not claim S5 is a large performance breakthrough.
- Do not claim residual head is universally ineffective across all possible settings.
- Do not claim fine token is always useless.
- Do not present oracle diagnostic as deployable method.

## Recommended thesis wording
- Preferred: "S5 provides a clean but marginal inference-time calibration gain over S2b."
- Preferred: "Main contribution is a reproducible clean pipeline from scale repair to locked final candidate."
- Avoid: any phrasing that implies dramatic end-to-end model redesign gains.
