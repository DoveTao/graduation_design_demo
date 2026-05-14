# S14 Local Window Pose Graph Optimization Report

## Executive summary
- final classification: `NO-STABLE-POSE-GRAPH-GAIN`
- baseline gate passed: `True`
- graph availability sufficient: `True`
- best train-CV candidate: `D_joint_w7_s1`
- S14b full clean CV recommended: `False`
- S5 remains final clean candidate: `True`

## Motivation from S13
- S13 identified `R-TDIR-COUPLED-LIMITED` as the primary bottleneck and `CHAIN-ACCUMULATION-LIMITED` as a secondary bottleneck.
- S14 therefore tests multi-edge local-window consistency instead of further pairwise post-processing.

## Baseline gate
- load_missing / load_unexpected: `14 / 0`
- missing categories: ridge_calib=`2`, coupled_pose_head=`12`, other=`0`
- locked metrics citation: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Graph availability audit
- `scene01/seq01`: frames=`324`, edges=`1507`, k_dist=`{'1': 155, '2': 218, '3': 252, '5': 299, '10': 314, '20': 269}`, non_adjacent=`True`, cycle_or_redundant=`True`
  window avg edges: w5=`5.512500`, w7=`11.248428`, w9=`16.971519`
- `scene01/seq02`: frames=`464`, edges=`1779`, k_dist=`{'1': 138, '2': 211, '3': 253, '5': 317, '10': 454, '20': 406}`, non_adjacent=`True`, cycle_or_redundant=`True`
  window avg edges: w5=`3.652174`, w7=`7.613537`, w9=`11.548246`
- `scene01/seq03`: frames=`454`, edges=`1695`, k_dist=`{'1': 132, '2': 191, '3': 237, '5': 305, '10': 430, '20': 400}`, non_adjacent=`True`, cycle_or_redundant=`True`
  window avg edges: w5=`3.455556`, w7=`7.234375`, w9=`10.982063`

## Measurement construction
- Every edge uses S5-derived `R_ij_pred`, `tdir_ij_pred`, `tmag_ij_pred`, with `dt` and `k` retained as inference-visible metadata.
- Ground truth is used only for evaluation and oracle-style diagnostics, not for optimization-time features.

## Optimization formulation
- Variables: per-frame local-window/global pose states in `SO(3)` + translation, first frame fixed per connected component.
- Residuals: rotation consistency, translation direction consistency, translation magnitude log residual, and path-length prior on k=1 chains.
- Robust candidates use Huber loss and downweight train-selected high-risk edges.

## Candidate definitions
- `A_identity_baseline`: family=`A`, window=`0`, stride=`0`, wR=`0.0`, wTdir=`0.0`, wTmag=`0.0`, wPath=`0.0`, robust=`none`, high_risk_downweight=`False`
- `B_rot_only_w7_s1`: family=`B`, window=`7`, stride=`1`, wR=`1.0`, wTdir=`0.0`, wTmag=`0.0`, wPath=`0.0`, robust=`none`, high_risk_downweight=`False`
- `C_pos_only_w7_s1`: family=`C`, window=`7`, stride=`1`, wR=`0.0`, wTdir=`1.0`, wTmag=`0.5`, wPath=`1.0`, robust=`none`, high_risk_downweight=`False`
- `D_joint_w7_s1`: family=`D`, window=`7`, stride=`1`, wR=`1.0`, wTdir=`1.0`, wTmag=`0.5`, wPath=`1.0`, robust=`none`, high_risk_downweight=`False`
- `D_joint_w5_s3`: family=`D`, window=`5`, stride=`3`, wR=`1.0`, wTdir=`1.0`, wTmag=`0.5`, wPath=`1.0`, robust=`none`, high_risk_downweight=`False`
- `D_joint_w9_s3`: family=`D`, window=`9`, stride=`3`, wR=`1.0`, wTdir=`1.0`, wTmag=`0.5`, wPath=`1.0`, robust=`none`, high_risk_downweight=`False`
- `E_robust_joint_w7_s1`: family=`E`, window=`7`, stride=`1`, wR=`1.0`, wTdir=`1.0`, wTmag=`0.5`, wPath=`1.0`, robust=`huber`, high_risk_downweight=`True`
- `F_path_strong_joint_w7_s1`: family=`F`, window=`7`, stride=`1`, wR=`1.0`, wTdir=`1.0`, wTmag=`0.5`, wPath=`2.0`, robust=`huber`, high_risk_downweight=`True`

## Train-CV selection protocol
- two folds on the train split: hold out `scene01/seq01` and `scene01/seq02` in turn
- high-risk q90 is derived from the opposite train fold only
- candidate ranking prioritizes safe path ratio, then CV ATE, then CV drift

## CV results table
| candidate | cv ATE | cv drift | cv path_ratio | rot residual delta | tdir residual delta | tmag residual delta | path safe | eligible for S14b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A_identity_baseline | 2.534051 | 1.418660 | 4.891592 | 0.000000 | 0.000000 | 0.000000 | False | False |
| B_rot_only_w7_s1 | 2.568383 | 1.536427 | 4.891592 | -2.427327 | -6.190266 | 0.126484 | False | False |
| C_pos_only_w7_s1 | 2.571007 | 1.393586 | 6.051862 | -0.000000 | 10.074213 | -0.109720 | False | False |
| D_joint_w7_s1 | 2.514520 | 1.349712 | 5.356321 | -6.469677 | 13.243660 | -0.195054 | False | False |
| D_joint_w5_s3 | 2.617682 | 1.543044 | 8.557513 | -6.366587 | -2.737825 | -0.231520 | False | False |
| D_joint_w9_s3 | 2.604198 | 1.501447 | 8.580319 | -6.352931 | -5.864515 | -0.161647 | False | False |
| E_robust_joint_w7_s1 | 2.523838 | 1.329883 | 4.313186 | -4.807069 | 14.334799 | -0.173681 | False | False |
| F_path_strong_joint_w7_s1 | 2.523298 | 1.329980 | 4.281464 | -4.818003 | 14.579612 | -0.172130 | False | False |

## Hard-gate audit
- no forbidden feature leakage: `True`
- leakage audit passed: `True`
- path-ratio safety for practical diagnostic uses `[0.90, 1.05]`
- clean replacement still requires S5 locked baseline comparison and a single selected-candidate final test

## Final test result
- selected candidate: `D_joint_w7_s1` on `scene01/seq03`
- baseline diagnostic: ATE=`2.635339`, drift=`1.326834`, path_ratio=`4.944008`
- selected diagnostic: ATE=`2.785907`, drift=`1.495600`, path_ratio=`7.277363`
- practical gap shrunk: `False`

## Practical threshold before/after
- `lenient` pass rate: baseline=`0.000000`, S14=`0.000000`
- `moderate` pass rate: baseline=`0.000000`, S14=`0.000000`
- `strict` pass rate: baseline=`0.000000`, S14=`0.000000`

## Component residual before/after
- best-CV fold summary: rot `12.922238 -> 20.408368`, tdir `57.668786 -> 37.012682`, tmag `0.540293 -> 0.748483`

## Regime / chain analysis
- high-risk evaluation focuses on `pred_tmag` upper tail and `dt>=1.0,k=20` edges.
- S14 is intended to reduce coupled R/tdir accumulation without relying on oracle substitutions.

## Failure cases
- If path ratio moves outside the practical-safe band, the candidate is treated as non-clean even if ATE improves.
- Sparse k=1 chains remain the reporting path, so graph gains must survive projection back to odometry accumulation.

## Leakage audit
- Optimization uses only S5 predictions plus inference-visible `dt`, `k`, and train-derived high-risk thresholds.
- Test labels are not used for candidate selection.

## Final classification
- `NO-STABLE-POSE-GRAPH-GAIN`
