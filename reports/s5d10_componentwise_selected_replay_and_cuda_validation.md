# S5D10 Componentwise Selected Replay and CUDA Validation

## Executive summary
- final classification: `S5D10_COMPLETE_WITH_CUDA_BLOCKER`

## CUDA validation recovery
- validation: {'verify_final_candidate': {'passed': False, 'log_path': 'logs/s5d10_verify_final_candidate.log'}, 'project_health_check': {'passed': False, 'log_path': 'logs/s5d10_project_health_check.log'}, 's6_eval_only': {'passed': False, 'log_path': 'logs/s5d10_s6_eval_only.log'}, 'unittest': {'passed': True, 'test_count': 115, 'log_path': 'logs/s5d10_unittest.log'}, 'validation_clean': False, 'cuda_oom_status': 'intermittent'}

## Component graph audit
- graph: {'num_pairs': 132, 'num_components': 19, 'longest_component_edges': 67, 'num_replayable_components': 19, 'component_graph_path': 'external_baselines/results/s5_pairwise_replay_s5d10/component_graph_audit.json'}

## Componentwise replay protocol
- declared/inverse variants are diagnostic only.

## Declared vs inverse convention comparison
- {'declared': {'num_components_evaluated': 19, 'relative_rot_diff_mean_deg': 1.7989036726528747, 'relative_tdir_diff_mean_deg': 104.45627113717318, 'relative_tmag_ratio_median': 0.7677981613872882}, 'inverse': {'num_components_evaluated': 19, 'relative_rot_diff_mean_deg': 41.49708918510672, 'relative_tdir_diff_mean_deg': 75.79944390669957, 'relative_tmag_ratio_median': 0.7677947715549021}, 'best_variant': 'declared'}

## Existing dense selected vs non-selected edge contributions
- {'selected_edges': {'num_edges': 132, 'rot_mean_deg': 20.820702786908246, 'tdir_mean_deg': 85.8020142769065, 'tdir_abs_mean_deg': 51.49108111896134, 'tmag_median_ratio': 1.310038384597929, 'path_length': 28.24870735259957, 'path_length_fraction': 0.39372704448922863}, 'nonselected_edges': {'num_edges': 321, 'rot_mean_deg': 20.82698038561777, 'tdir_mean_deg': 93.69127653978485, 'tdir_abs_mean_deg': 59.50699260891793, 'tmag_median_ratio': 29.375292370072714, 'path_length': 43.498224304700855, 'path_length_fraction': 0.6062729555107714}}

## Gap diagnosis
- {'selected_pairwise_tdir_issue_confirmed': True, 'convention_issue_suspected': False, 'dense_pathratio_driven_by_nonselected_edges': True, 'full_dense_tmag_issue_not_explained_by_selected_edges': True, 'most_likely_gap_source': 'nonselected_dense_edges', 'evidence': ['selected_path_fraction=0.39372704448922863', 'nonselected_path_fraction=0.6062729555107714', 'declared_score=106.25517480982604', 'inverse_score=117.29653309180628'], 'recommended_next_actions': ['If CUDA OOM persists, run validator in isolated GPU window (single process) and keep PYTORCH_CUDA_ALLOC_CONF set.', 'Prioritize non-selected edge quality improvements since they dominate path length contribution.', 'Use best diagnostic convention only for analysis; do not alter official evaluation pipeline.', 'Keep selected_k1 sparse/disconnected caveat in all cross-stage comparisons.']}

## Caveats
- diagnostic only
- selected_k1 sparse/disconnected protocol
- no GT used for prediction
- does not replace official S5 locked result
- S5 locked metrics/policy unchanged
