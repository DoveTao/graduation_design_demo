# THESIS361 Claims and Caveats Checklist

## Allowed Claims
- `FINAL360I` improves pair-level translation component metrics over `T57b`.
- `FINAL360I` outperforms the trajectory-derived `BASE360D` component baseline on component-level translation metrics, with the trajectory-derived caveat clearly stated.
- `FINAL360I` can be composed into full trajectories through direct adjacent-pair composition.
- `SEQ360B` improves path-length behavior and ATE SE3 relative to `TRAIN360E`.
- `ATE Sim3` remains challenging and indicates unresolved trajectory-shape error.
- The main contribution of the work lies in match-free panoramic pair-level relative pose estimation.

## Forbidden Claims
- `FINAL360I` fully outperforms HKUST official 360DVO as a complete VO pipeline.
- `SEQ360B` solves trajectory drift.
- `STRUCT360C` improves the final retained results.
- `BASE360D` component metrics and `FINAL360I` pair-level metrics are fully identical in protocol.
- Pair-level success automatically implies trajectory-level superiority.
- `SEQ360A` provides a successful sequence-consistency improvement.
