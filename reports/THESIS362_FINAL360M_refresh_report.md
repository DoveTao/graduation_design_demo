# THESIS362 FINAL360M refresh report

- checkpoint used: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- training executed: `false`
- eval-only trajectory refresh executed: `true`
- refreshed pair table now promotes `FINAL360M` and demotes `FINAL360I` to subset-trained candidate
- refreshed trajectory table now includes FINAL360M direct / ODOM360A / ODOM360B alongside old backends and BASE360D
- updated figure set includes pair-level bar, trajectory metric comparison, and refreshed ridge_to_lake overlay
- recommended FINAL360M trajectory backend for thesis narrative: `lightweight_trajectory_fusion`
