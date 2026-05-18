# FINAL360M trajectory-level evaluation

## Pair prediction export
- checkpoint path: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- val manifest: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`
- test manifest: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- val pair count: `5342`
- test pair count: `5062`
- val export path: `external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/cache/val_all_pair_predictions.jsonl`
- test export path: `external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/cache/test_all_pair_predictions.jsonl`

## Trajectory methods
- direct composition test ATE none / SE3 / Sim3: `251.214513` / `119.934128` / `29.423059`
- direct composition test path ratio: `1.867511`
- lightweight trajectory fusion test ATE none / SE3 / Sim3: `132.075060` / `53.831131` / `29.394188`
- lightweight trajectory fusion test path ratio: `1.186821`
- local pose graph test ATE none / SE3 / Sim3: `132.076321` / `53.833900` / `29.395214`
- local pose graph test path ratio: `1.221642`

## Interpretation
- pair-level scale/path improvement transferred to trajectory-level: `false`
- worse signed_tdir likely hurts final shape: `true`
- old ODOM360A conclusion still holds unchanged: `false`
- update backend main result to FINAL360M-based version: `true`
