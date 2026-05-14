# Final S12 Regime Balanced Sampling Summary

- Final classification: `NO-STABLE-REGIME-SAMPLING-GAIN`
- Run mode: `cv`
- Device: `cuda`
- Odometry eval run: `True`
- Best diagnostic sampler: `D_dt_k_balanced_sampling`
- S12 replaces S5: `False`
- S12b full clean CV recommended: `False`
- S5 remains final clean candidate at this stage

This stage upgrades S12 from audit to a lightweight two-fold CV comparison over A/D/E/F. It reuses the train-only manifests generated in audit mode, keeps the model structure fixed, and asks whether sampler choice alone produces a stable high-risk regime benefit without unsafe path-ratio or odometry drift.

Outcome: `sampler improved high-risk proxy but odometry/path safety was not stable enough for promotion`. At this stage S5 still remains the locked final clean candidate.
