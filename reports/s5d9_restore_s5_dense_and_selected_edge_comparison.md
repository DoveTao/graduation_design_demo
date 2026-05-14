# S5D9 Restore S5 Dense and Selected Edge Comparison

## Executive summary
- final classification: `S5D9_DENSE_RESTORED_COMPARISON_COMPLETE`

## S5D8 blocker recap
- missing existing S5 dense TUM path caused selected-edge dense comparison not computed.

## Dense TUM restore/regeneration result
- {'target_path': 'external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt', 'restored': True, 'source': 'found_existing', 'source_path': 'git:experiment/orbslam3-fisheye-strong-baseline:external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt', 'num_poses': 454, 'timestamp_status': 'aligned_exact', 'gt_leakage_check_passed': True, 'metrics_consistent_with_s5d': True}

## Dense TUM validation
- num poses: 454

## Existing dense on selected edges
- {'rot_mean_deg': 20.820702786908246, 'tdir_mean_deg': 85.8020142769065, 'tdir_abs_mean_deg': 51.49108111896134, 'tmag_median_ratio': 1.310038384597929, 'tmag_mean_ratio': 2.313284854791453}

## Selected replay vs existing dense comparison
- {'relative_rot_diff_mean_deg': 1.6960582806780404, 'relative_rot_diff_median_deg': 1.67937251362133, 'relative_rot_diff_p90_deg': 2.186814202274997, 'relative_tdir_diff_mean_deg': 96.66290826370484, 'relative_tdir_diff_median_deg': 99.3853017620607, 'relative_tdir_diff_p90_deg': 163.0420208099238, 'relative_tmag_ratio_mean': 0.7488460630322377, 'relative_tmag_ratio_median': 0.767765189129267, 'path_length_ratio_replay_over_existing': 0.674259051441856}

## Gap diagnosis
- {'selected_pairwise_tdir_issue_confirmed': True, 'selected_pairwise_tmag_good': True, 'dense_export_or_integration_mismatch_suspected': False, 'replay_dense_pipeline_mismatch_suspected': True, 'most_likely_gap_source': 'selected_pairwise_tdir', 'evidence': ['pairwise_tdir_mean=99.54956140857581', 'pairwise_tmag_median=0.9819414718338633', 'dense_selected_tmag_median=1.310038384597929', 'common_edges=67'], 'recommended_next_actions': ['If needed, replay all graph components instead of only longest chain for broader edge overlap.', 'Keep selected_k1 sparse protocol caveat in all cross-metric conclusions.', 'Use identical edge protocol when comparing pairwise/replay/dense to avoid attribution bias.']}

## Caveats
- diagnostic only
- does not replace official S5 locked result
- S5 locked metrics/policy unchanged
- selected_k1 sparse protocol caveat
