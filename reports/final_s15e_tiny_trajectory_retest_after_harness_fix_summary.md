# S15e Tiny Trajectory Retest After Harness Fix Summary

- final classification: `TRAINING-STILL-UNSTABLE`
- candidates run: `A_pair_only_baseline_fixed_harness, D_full_light_traj_fixed_harness`
- training budget: `W=3, updates=20, batch_size=1`
- trainable mode: `tmag_head_only`
- A result: val_ate_proxy=`5.457961`, odom_ATE=`21.710394`, drift=`35.126991`
- D result: val_ate_proxy=`5.528377`, odom_ATE=`21.995400`, drift=`35.549201`
- pre-fix D comparison: old_ate_proxy=`15.708141` -> new=`5.528377`, old_drift=`24.674397` -> new=`35.549201`
- S15f lightweight CV recommended: `False`
- S5 remains final clean candidate: `yes`
