# S15a Trajectory Training Smoke Failure Attribution Summary

- final classification: `TRAINING-HARNESS-ISSUE`
- main failure cause: pair-anchor-only tiny run already collapses, so the failure appears before trajectory loss tuning
- accumulation convention matched eval: `True`
- no-train baseline proxy: val_ate_proxy=`5.554382`, val_path_proxy=`1.907361`
- pair-anchor-only tiny: ATE=`37.949599`, drift=`59.054601`
- gradient finding: trajectory/pair grad ratio=`0.421711`
- parameter drift finding: smoke final checkpoint is not available locally, so direct before/after parameter delta for fine_pose_light could not be recomputed in this audit
- S15b full clean CV recommended: `no`
- S5 remains final clean candidate: `yes`
