# THESIS360 Eval-only Matching Geometry Debug Export

- task name: `THESIS360_eval_only_matching_geometry_debug_export`
- debug hook added: `true`
- default model behavior changed: `false`
- training executed: `false`
- checkpoint modified: `false`
- metrics modified: `false`
- eval-only forward executed: `true`
- sample split: `val`

## Real Exported Quantities
- cross-image token relation heatmaps before and after fine residual refinement
- cross-attention context response energy
- fine gate and residual gate scalar responses
- applied delta rotation / delta tdir / delta log_tmag
- coarse pose vs final pose diagnostics against GT
- spherical token layout in ERP coordinates

## Unavailable Targets
- explicit matching / keypoint correspondence: The retained FINAL360I / STRUCT360B mainline is match-free and does not emit explicit keypoint matches.
- epipolar allowed mask / epipolar residual / epipolar bias: These tensors are not part of the retained STRUCT360B forward path and were not fabricated.

## Figure Usage
- `thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample01_normal_mountains.png` -> `defense_ppt`: Eval-only token-level relation heatmaps derived from real fine-stage tokens; this is not explicit keypoint matching.
- `thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample02_normal_downhill.png` -> `thesis_appendix`: Eval-only token-level relation heatmaps derived from real fine-stage tokens; this is not explicit keypoint matching.
- `thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample03_scale_path_k5.png` -> `thesis_appendix`: Eval-only token-level relation heatmaps derived from real fine-stage tokens; this is not explicit keypoint matching.
- `thesis/final_assets/figures/matching_geometry/thesis360_residual_gate_diagnostic.png` -> `thesis_appendix`: Shows real fine-gate and residual-gate responses plus applied residual update magnitudes.
- `thesis/final_assets/figures/matching_geometry/thesis360_coarse_final_pose_diagnostic.png` -> `thesis_appendix`: Compares coarse and final pose errors against GT on fixed eval-only samples.
- `thesis/final_assets/figures/matching_geometry/thesis360_token_layout_visualization.png` -> `defense_ppt`: Visualizes actual coarse and fine spherical token locations in ERP coordinates.

## Naming Caveat
- these figures visualize cross-image interaction / token relation / residual refinement
- they are not explicit keypoint matching figures
