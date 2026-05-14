# Thesis Three-Axis Ablation and Baseline Comparison

## Scope
This report reorganizes existing locked or historical results around the three thesis innovation axes. It is reporting-only, introduces no new experiment line, selects no new candidate, and leaves S5 as the final clean candidate.

## Final Candidate
- `S5_clean_tmag_calibration_policy`
- ATE = `7.352288`
- drift = `1.327343`
- path_ratio = `0.932379`

## Axis 1: Spherical Formulation
- A strict non-spherical retrained baseline is not available in the locked historical protocol.
- Current direction diagnostics still show the direction branch is weak:
  - rot mean / median / p90 / max = `20.715353` / `20.433789` / `21.371863` / `21.717051` deg
  - tdir mean / median / p90 / max = `72.349483` / `102.517940` / `110.918120` / `141.610521` deg
  - tdir mean cosine similarity = `0.226300`
- Interpretation: The spherical/coarse-fine pipeline provides a structured direction representation, but current results still show that the translation-direction branch is a major limitation. This report does not claim that spherical formulation alone solved tdir prediction.

## Axis 2: Coarse-to-Fine Refinement
| method | role | ATE | drift | path_ratio | delta_ATE_vs_S5 | delta_drift_vs_S5 | delta_path_ratio_vs_S5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1d5_clean_dt_anchor_policy | scale/path_ratio collapse repaired clean baseline | 7.632463 | 1.396358 | 0.934982 | 0.280175 | 0.069015 | 0.002603 |
| S2b_clean_fine_rot_policy | major clean fine-rotation improvement | 7.352371 | 1.327402 | 0.934984 | 0.000083 | 0.000059 | 0.002605 |
| S5_clean_tmag_calibration_policy | final locked clean candidate | 7.352288 | 1.327343 | 0.932379 | 0.000000 | 0.000000 | 0.000000 |

| transition | component_added | delta_ATE | delta_drift | delta_path_ratio | interpretation |
| --- | --- | --- | --- | --- | --- |
| S1d5 -> S2b | fine_rot clean policy | -0.280092 | -0.068956 | 0.000001 | adding fine_rot produced the major clean ATE/drift improvement while preserving safe path_ratio |
| S2b -> S5 | pred_tmag regime-aware calibration | -0.000083 | -0.000059 | -0.002605 | tmag calibration produced only a very small clean gain, but passed train-CV, leakage audit, and lockdown and therefore became the final candidate |

- Interpretation: Coarse-to-fine refinement, especially the fine rotation policy from S1d5 to S2b, accounts for the largest clean improvement. The final S5 tmag calibration contributes only a marginal but reproducible gain.

## Axis 3: Geometric Constraints
| method | constraint | ATE | drift | path_ratio | interpretation |
| --- | --- | --- | --- | --- | --- |
| S1d5_clean_dt_anchor_policy | dt-anchor / scale-path-ratio correction | 7.632463 | 1.396358 | 0.934982 | repaired the historical path_ratio collapse and stabilized the clean mainline |
| S5_clean_tmag_calibration_policy | tmag regime-aware calibration | 7.352288 | 1.327343 | 0.932379 | final clean candidate with marginal but locked gain |
| S14_local_window_pose_graph_selected | local-window pose graph optimization | 2.785907 | 1.495600 | 7.277363 | naive stronger geometry constraints worsened the diagnostic instead of helping |

- S14 diagnostic baseline on `scene01/seq03`: ATE=`2.635339`, drift=`1.326834`, path_ratio=`4.944008`
- S14 selected diagnostic: ATE=`2.785907`, drift=`1.495600`, path_ratio=`7.277363`, classification=`NO-STABLE-POSE-GRAPH-GAIN`
- Interpretation: Lightweight scale/tmag constraints help stabilize path behavior, but naive local-window pose graph optimization did not improve results. Geometry-constraint claims in the thesis should therefore remain cautious.

## Oracle Diagnostic
| diagnostic | ATE | drift | path_ratio | interpretation |
| --- | --- | --- | --- | --- |
| official_current_S5 | 7.352288 | 1.327343 | 0.932379 | locked clean final candidate under the historical protocol |
| oracle_R | 10.590301 | 1.516962 | 0.934986 | rotation alone does not solve the practical gap and can even worsen global metrics |
| oracle_tdir | 7.658319 | 1.375470 | 0.934985 | translation direction alone does not solve the practical gap |
| oracle_R_tdir | 0.911024 | 0.304305 | 0.934986 | jointly fixing R and tdir collapses most of the gap, supporting the R-TDIR-COUPLED-LIMITED diagnosis |
| oracle_tmag | 7.214477 | 1.262237 | 0.898234 | tmag alone can improve some scalar metrics but path_ratio falls below the safe range, so tmag is not the main bottleneck |

- Fixing R alone does not solve the problem and can worsen global metrics.
- Fixing tdir alone does not solve the problem either.
- Fixing R and tdir jointly produces a dramatic gain, supporting the R-TDIR-COUPLED-LIMITED conclusion.
- Fixing tmag alone gives some scalar gain but path_ratio leaves the safe range, so tmag is not the main bottleneck.

## Thesis-Ready Interpretation
- spherical formulation provides a structured direction representation, but current direction prediction remains weak.
- coarse-to-fine refinement contributes the largest observed clean improvement.
- geometric scale/tmag constraints stabilize the system but only marginally improve final S5.
- naive stronger geometry constraints do not necessarily help, as shown by S14.
- S5 remains the best clean candidate, but not practical-ready.

## Caveats
- small sequence protocol
- representativeness caveat from S17/S18
- S5 not practical-ready
- no deployment readiness claim
- missing strict non-spherical baseline in the locked historical protocol

