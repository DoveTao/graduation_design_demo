# Thesis Experiment Section Draft

## Experimental Setup
The experiments are conducted on the panoramic relative-pose dataset used throughout the project, with the historical protocol retained as the authoritative evaluation contract. Under this protocol, the final clean test target is the single held-out sequence `scene01/seq03`, while model and policy selection are restricted to train/CV evidence with leakage audit and final-test lockdown. The final clean candidate remains `S5_clean_tmag_calibration_policy`, whose locked metrics are `ATE = 7.352288`, `drift = 1.327343`, and `path_ratio = 0.932379`.

The evaluation uses both trajectory-level and component-level diagnostics. Trajectory-level metrics include `ATE`, `drift`, and `path_ratio`, where `path_ratio` is retained as a scale-sensitive safety indicator rather than a cosmetic auxiliary score. Component-level diagnostics summarize relative rotation, translation direction, and translation magnitude behavior. This separation is important because later results show that some methods can achieve favorable aligned trajectory-shape scores while still exhibiting unstable pairwise direction or scale behavior.

The entire experimental workflow follows a clean-evaluation discipline: internal candidate selection is constrained by train/CV only, test labels are not used for policy tuning, and the final clean result is locked after explicit reproduction and leakage audits. Accordingly, the purpose of the analyses below is not to replace the final candidate, but to explain why `S5` was retained and where its remaining weaknesses lie.

## Internal Clean Baseline Comparison
Table 1 summarizes the main internal clean baselines. These results correspond to the authoritative historical protocol and are therefore the primary quantitative comparison for final-candidate selection.

| Method | Role | ATE | drift | path_ratio |
| --- | --- | --- | --- | --- |
| S1d5 | scale/path-ratio collapse repaired clean baseline | 7.632463 | 1.396358 | 0.934982 |
| S2b | major clean fine-rotation improvement | 7.352371 | 1.327402 | 0.934984 |
| S5 | final locked clean candidate | 7.352288 | 1.327343 | 0.932379 |

These results support a simple mainline narrative. First, `S1d5` repaired the earlier scale/path-ratio collapse and established a usable clean baseline. Second, `S2b` delivered the major clean improvement by strengthening the rotation side of the pipeline. Finally, `S5` only improved `ATE` and `drift` by a very small margin, but that gain was reproducible, leakage-clean, and preserved under lockdown, which is why it remained the final candidate.

## Three-Axis Ablation
The three-axis ablation reorganizes the historical experiments around the thesis innovations: spherical formulation, coarse-to-fine refinement, and geometric constraints.

### Axis 1: Spherical Formulation
A strict non-spherical retrained baseline is not available under the locked historical protocol, so this axis cannot be presented as a direct spherical-versus-non-spherical retraining table. The fairest statement is therefore methodological rather than triumphalist: spherical formulation provides a structured direction representation for panoramic input, but it does not remove the direction-estimation bottleneck by itself. This interpretation is consistent with later diagnostics showing that translation direction remains difficult even when the overall pipeline is carefully engineered.

### Axis 2: Coarse-to-Fine Refinement
The clean improvement from `S1d5` to `S2b` is the most important internal gain:

| Transition | delta_ATE | delta_drift | Interpretation |
| --- | --- | --- | --- |
| S1d5 -> S2b | -0.280092 | -0.068956 | Fine rotation / coarse-to-fine refinement provides the major clean gain. |

This is the strongest quantitative evidence among the thesis innovations. In practice, it means the largest verified progress did not come from post-hoc scale handling, but from improving the relative rotation/refinement pathway.

### Axis 3: Geometric Constraints
The `S2b -> S5` transition is much smaller:

| Transition | delta_ATE | delta_drift | delta_path_ratio | Interpretation |
| --- | --- | --- | --- | --- |
| S2b -> S5 | -0.000083 | -0.000059 | -0.002605 | Tmag calibration gives only a marginal but clean final gain. |

This result matters precisely because it is small. It shows that lightweight geometric scale calibration can still matter enough to decide the final clean candidate, but it is not the dominant source of improvement. This caution is reinforced by the negative `S14` result, where a stronger local-window pose-graph style geometric constraint did not produce a stable gain. Therefore, the thesis should not claim that simply adding more geometry always helps; rather, it should state that carefully constrained lightweight calibration can help, while naive stronger geometry can fail.

## Strong Classical Baseline: Pano-ORB-VO
To complement the internal learning-based baselines, a protocol-compatible classical baseline was implemented as `Pano-ORB-VO`. This baseline does not use original fisheye camera streams and is not ORB-SLAM3. Instead, it starts from the available stitched equirectangular panoramas, extracts panorama-derived virtual pinhole views, performs ORB feature matching, estimates an essential matrix, recovers relative pose, and composes a trajectory using a train-only non-test-tuned scale policy.

### Trajectory-Level Comparison
Because `SB2b` shows that the current S5 external TUM export is sparse diagnostic only, `S5` locked metrics and `Pano-ORB-VO` external-trajectory metrics should be reported in separate blocks rather than as a falsely alignment-consistent full-coverage table.

| Method | Evaluator block | ATE | drift | path_ratio | Notes |
| --- | --- | --- | --- | --- | --- |
| S5 | locked clean evaluator | 7.352288 | 1.327343 | 0.932379 | authoritative final result |
| Pano-ORB-VO (none) | external trajectory evaluator | 10.194695 | 0.172360 | 0.811334 | full coverage |
| Pano-ORB-VO (se3) | external trajectory evaluator | 4.137858 | 0.173159 | 0.811334 | alignment-corrected shape |
| Pano-ORB-VO (sim3) | external trajectory evaluator | 4.133322 | 0.233378 | 0.811334 | alignment-corrected shape with scale alignment |

The key observation is that `Pano-ORB-VO` is competitive in alignment-corrected trajectory-shape metrics, but its `path_ratio` is clearly worse than `S5`. Therefore, `Pano-ORB-VO` is a meaningful strong baseline, yet it does not replace the locked clean candidate. At the same time, these results also prevent any claim that `S5` is uniformly superior to classical geometry.

### Component-Level Comparison
Pairwise component diagnostics reveal why the trajectory story is mixed.

| Method | Source | num_valid_pairs | rot_mean_deg | rot_median_deg | rot_p90_deg | rot_max_deg | tdir_mean_deg | tdir_median_deg | tdir_p90_deg | tdir_max_deg | tdir_mean_cosine | tmag_mean_log_error | tmag_median_log_error | tmag_p90_log_error | tmag_max_log_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pano-ORB-VO | same-evaluator full-coverage pairwise diagnostic | 453 | 21.171064 | 0.705887 | 179.468812 | 179.998624 | 93.422068 | 94.202303 | 143.112335 | 172.996726 | -0.058091 | 1.513278 | 1.195670 | 3.236833 | 5.617485 |
| S5 reference | S13 diagnostics reference only, not same external full-coverage evaluator | N/A | 20.715353 | 20.433789 | 21.371863 | 21.717051 | 72.349483 | 102.517940 | 110.918120 | 141.610521 | 0.226300 | 0.899289 | 0.793141 | 1.742507 | 1.985574 |

`Pano-ORB-VO` shows an unusual error profile. Its rotation median is very low, which indicates that many local ORB-based relative estimates are nearly correct. However, its rotation distribution also contains an extreme long tail, with `p90` and maximum errors near `180 deg`, suggesting catastrophic flip-like failures on a nontrivial subset of pairs. Its translation-direction and translation-magnitude diagnostics are also worse than the `S5` reference row. Thus, the classical geometry baseline is not simply “good” or “bad”; rather, it is competitive in some trajectory-shape metrics while remaining fragile and heavy-tailed at the pairwise component level.

Feature-tracking diagnostics add useful context: `Pano-ORB-VO` achieves full coverage (`454` poses, `tracking_success_rate = 1.0`) with `mean_inliers = 191.67` and `median_inliers = 88.0`, and it uses all four virtual yaw sectors rather than collapsing onto a single view direction. This makes it a credible strong baseline rather than a toy failure case.

## Oracle Diagnostic and Practical Gap
The oracle experiments from `S13` are essential for understanding why the remaining gap persists.

| Diagnostic | ATE | drift | path_ratio | Interpretation |
| --- | --- | --- | --- | --- |
| official current / S5 | 7.352288 | 1.327343 | 0.932379 | locked final clean candidate |
| oracle_R | 10.590301 | 1.516962 | N/A | fixing rotation alone does not solve the practical gap |
| oracle_tdir | 7.658319 | 1.375470 | N/A | fixing translation direction alone does not solve the practical gap |
| oracle_R_tdir | 0.911024 | 0.304305 | N/A | jointly fixing rotation and direction collapses most of the gap |
| oracle_tmag | 7.214477 | 1.262237 | 0.898234 | scalar improvement exists, but path behavior degrades |

These results reject simplistic bottleneck stories. Rotation alone is not sufficient. Translation direction alone is not sufficient. Magnitude alone can improve some scalar metrics, but it does not preserve safe path behavior. The strong oracle gain only appears when rotation and translation direction are corrected together. Therefore, the main remaining bottleneck is best described as `R-TDIR-COUPLED-LIMITED`.

## External Literature Baselines
The broader external baseline section should remain disciplined. `ORB-SLAM2`, `ORB-SLAM3`, `DSO`, and `DROID-SLAM` are included as qualitative literature-context baselines only, because no protocol-compatible quantitative runs were completed on the exact dataset and input contract. Published numbers from other datasets must not be compared directly against `S5`.

This limitation is especially important here because the project does not retain raw fisheye streams as the working experimental input. The available input is stitched equirectangular panorama imagery. For that reason, `Pano-ORB-VO` should be described as a repo-local protocol-compatible strong baseline, not as a published external baseline and not as an original fisheye-camera baseline.

## Limitations
Several limitations should be stated explicitly.

First, `S5` is not practical-ready. It is a clean and reproducible final candidate under the locked historical protocol, but that is not the same as deployment readiness. Second, the protocol is based on a very small number of sequences, and `S17/S18` already showed the representativeness caveat associated with the historical single-sequence final test. Third, a strict non-spherical retrained baseline is unavailable, so the spherical-formulation contribution cannot be isolated by a fully symmetric retraining comparison.

Fourth, the current `S5` external TUM export is sparse diagnostic only, as shown by `SB2b`, and therefore cannot be used as a full-coverage alignment-consistent comparison row against `Pano-ORB-VO`. Fifth, `Pano-ORB-VO` itself is panorama-derived rather than a direct original fisheye baseline. Finally, there is no direct quantitative `ORB-SLAM3` or `DROID-SLAM` comparison under the same protocol, because protocol-compatible external runs are not available.

## Thesis-Ready Conclusion
Under the locked historical protocol, `S5_clean_tmag_calibration_policy` remains the final clean candidate. The largest verified clean improvement comes from coarse-to-fine refinement, especially the fine-rotation improvement from `S1d5` to `S2b`. The final `S5` step contributes only a marginal but reproducible gain through lightweight tmag-aware calibration. The protocol-compatible `Pano-ORB-VO` strong baseline shows that classical geometry remains competitive in alignment-corrected trajectory shape, yet its pairwise diagnostics reveal severe long-tail failures and weaker direction/magnitude behavior. The oracle analyses further show that the dominant remaining bottleneck is coupled rotation-direction error rather than isolated scalar calibration. Overall, the system is clean, reproducible, and experimentally well-audited, but it is not practical-ready.
