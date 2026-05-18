# CURRENT_MAINLINE FINAL360M Thesis Narrative Refresh

## Final Narrative Position

- `FINAL360M_fulltrain_struct360b_thesis_main_guarded` is the final pair-level thesis main model
- the promotion is justified by `protocol qualification`, not by blanket metric superiority
- old `FINAL360I` must be described as a `subset-trained candidate only`

## Pair-level Thesis Wording

- recommended wording:
  - `FINAL360M completed the full-train protocol and is therefore adopted as the final thesis main model.`
  - `Compared with the earlier FINAL360I subset-trained candidate, the result is mixed rather than uniformly better.`
  - `The final decision is based on training-protocol validity and checkpoint lineage, not on a claim that every metric improved simultaneously.`

## Trajectory-level Thesis Wording

- recommended wording:
  - `Direct composition of FINAL360M predictions still shows substantial sequential drift.`
  - `A lightweight trajectory-fusion backend improves FINAL360M direct composition on SE3 ATE and path ratio.`
  - `However, the refreshed FINAL360M-based backend remains weaker than the earlier subset-model-based ODOM360A reference, so the trajectory backend result should be presented as a partial improvement rather than a definitive sequence-level breakthrough.`

## Core Scientific Takeaway

- the full-train pair model is now complete and thesis-ready
- `tdir` remains the dominant bottleneck
- backend fusion helps but does not fully remove trajectory-shape error
- this means the thesis can claim:
  - `full-train pair-level main model completed`
  - `trajectory-level backend improvement is real but limited`
  - `further trajectory-shape stabilization remains future work`
