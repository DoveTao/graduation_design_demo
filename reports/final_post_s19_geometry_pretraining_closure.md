# Final Post-S19 Geometry Pretraining Closure

## Why S19 Stops
S19 was scoped as a lightweight feasibility diagnostic, not a final candidate replacement line. The goal was to test whether geometry-aware supervision on top of frozen current features could produce a cleaner R/tdir/joint probe signal than the current task-specific representation. That condition was not met, so the line stops here and does not proceed to S19b.

## Current vs ResNet50 vs Geometry Pretraining
- Current task-specific baseline in S19: `rot_R2=-16.342585`, `tdir_R2=-28.474333`, `joint_AUC=0.581828`
- Frozen ImageNet ResNet50 reference from S16b: `rot_R2=-258.785889`, `tdir_R2=-5.509770`, `joint_AUC=0.383036`
- Best geometry-aware pretraining result in S19: `geometry_multitask_small` with `rot_R2=-24.773435`, `tdir_R2=-17.617115`, `joint_AUC=0.513306`

The comparison shows that shallow geometry-aware pretraining did not produce a stable coupled gain over the current task-specific baseline. It also did not clearly establish a new representation that is uniformly better than the frozen ImageNet reference on the metrics that matter for the R/tdir bottleneck.

## Partial Rot/Tdir Trade-off
S19 did expose partial component-level signal:
- `rot_tdir_pretrain_small` improved `rot_R2` to `-7.008641`, but its `tdir_R2` and `joint_AUC` still failed to beat the current baseline.
- `geometry_multitask_small` improved `tdir_R2` to `-17.617115`, but its `rot_R2` and `joint_AUC` degraded relative to the current baseline.

This trade-off is the core reason S19 stops. The bottleneck identified in S13 is coupled `R-TDIR-COUPLED-LIMITED` behavior, not a single isolated scalar or component target. A method that helps only one side while hurting the coupled proxy does not justify deeper integration.

## Final Candidate Status
The final candidate does not change.

- final candidate: `S5_clean_tmag_calibration_policy`
- locked metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`
- S19 classification: `NO-STABLE-GEOMETRY-PRETRAINING-GAIN`
- S19b full integration recommended: `False`

## Future Work If Continued Later
If this direction is revisited in future work, the next step should not be another shallow probe-head-only pass. The more credible directions are:
- deeper multi-frame architecture rather than pairwise frozen-feature probing
- better data coverage and broader sequence diversity
- geometry-aware pretraining integrated into the backbone/representation path itself rather than a shallow adapter trained only for probe readability

Those directions remain future work only. They do not alter the current final clean result, and they do not reopen S19 under the present closure.
