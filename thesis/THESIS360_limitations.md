# THESIS360 Limitations

## 1. Pair-level success does not imply full VO pipeline superiority
The strongest retained result is pair-level relative pose estimation. This does not by itself establish superiority over a mature full sequence VO pipeline such as official 360DVO.

## 2. Direct sequential composition accumulates drift
`TRAIN360E` shows that direct adjacent-pair composition can recover full trajectories, but also shows that drift accumulates without an additional sequence-level correction mechanism.

## 3. ATE Sim3 remains difficult
The trajectory-level Sim3 metric remains hard to improve. Even after scale-oriented correction in `SEQ360B`, Sim3 ATE does not improve in a meaningful way.

## 4. SEQ360B improves scale/path but not shape
The retained variant demonstrates that path-length and SE3 behavior can be improved, but also shows that the core trajectory-shape error remains unresolved.

## 5. BASE360D remains a mature sequence VO pipeline
The official external baseline still has strong trajectory-level behavior and cannot be dismissed by pair-level comparisons alone.

## 6. No explicit matching, RANSAC, PnP, or BA
The present work deliberately avoids explicit matching and classical geometric post-processing. This keeps the method aligned with the match-free panoramic relative pose objective, but it also limits the current trajectory-level correction capacity.

## 7. Future work
Future work should focus on sequence-level refinement rather than only stronger isolated pair prediction. Promising directions include sequence-level pose graph refinement, temporal memory, learned global scale calibration, lightweight trajectory optimization, and explicit rotation/translation consistency over longer clips.

