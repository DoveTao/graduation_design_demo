# Final S5 clean candidate summary

- mainline: S1d5 -> S2b -> S4 -> S5.
- S3a/S3b were not continued because residual-head directions did not convert to stable clean gains under the no-retrain constraint.
- S4 pointed to magnitude-regime error concentration (high tmag / long-step regime), motivating S5 inference-time calibration.
- S5 validated magnitude-regime-aware calibration cleanly: train-CV selected candidate, train-only quantiles, no test-time policy construction.
- current final clean candidate: `highpred_q90_mid0p95_high0p80` frozen at `checkpoints/S5_clean_tmag_calibration_policy.json`.
- official comparison vs S2b: drift `1.327402` -> `1.327343`, ATE `7.352371` -> `7.352288`, path_ratio `0.934984` -> `0.932379`.
- paper claim recommendation: S5 provides a clean but marginal inference-time calibration gain.
- avoid claiming large model-level improvement.
