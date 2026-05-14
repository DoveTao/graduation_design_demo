# BASE360D component metric alignment

## 1. Executive summary
- `BASE360D alignment status = success`
- `BASE360D` remains the kept external baseline label for HKUST official `360DVO`
- current usage: official trajectory baseline plus trajectory-derived pair/component metrics aligned to the DSET2C canonical manifest
- main caveat: component metrics are derived from sequence trajectory post-processing rather than native image-pair forward output

## 2. Current visible role in the cleaned repo
- external baseline name: `BASE360D`
- paired current mainline comparison targets:
  - `FINAL360I` pair-level main model
  - `TRAIN360E` trajectory export of `FINAL360I`
  - `SEQ360B` retained sequence-scale variant
  - `T57b` legacy recovered external reference
- this report is kept because it explains how the official external baseline enters the current comparison set after MAINT13 cleanup

## 3. Current test component metrics
- signed translation-direction mean: `128.402578`
- anti-parallel rate: `0.814895`
- translation-magnitude median ratio: `0.039910`
- pair-component path ratio: `0.043542`

## 4. Current test trajectory metrics
- ATE none RMSE: `103.256252`
- ATE SE3 RMSE: `79.295214`
- ATE Sim3 RMSE: `2.974466`
- trajectory path ratio: `0.043618`

## 5. Interpretation
- `BASE360D` is stronger than the kept learned mainline on trajectory-shape metrics such as Sim3 ATE.
- `BASE360D` is much weaker than the kept learned mainline on pair/component translation metrics.
- this asymmetry is expected because `BASE360D` is an official sequence pipeline, while `FINAL360I` is a pair-level model and `TRAIN360E` composes pair predictions into trajectories.
- the official public demo required a `0.5x` image adapter to avoid full-resolution OOM; that caveat remains part of the interpretation.

## 6. Caveats
- derived from trajectory output, not direct image-pair forward output
- official public demo used a `0.5x` image adapter to avoid full-resolution OOM
- pair-component path ratio and trajectory path ratio are distinct and should not be conflated
- this report no longer treats older in-repo baselines as part of the visible main comparison set after MAINT15 cleanup

## 7. Compliance note
- no new inference, training, or checkpoint modification was executed during MAINT15
- no metric values were changed during label cleanup
