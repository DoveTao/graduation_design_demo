# S5D8 Selected Pairwise Replay Comparison

## Executive summary
- final classification: `S5D8_REPLAY_GRAPH_DISCONNECTED`

## S5D7 artifact recap
- pairs: 132 (`selected_k1` sparse protocol)

## Artifact graph audit
- connected: `False`
- components: `19`
- replayable edges: `67`

## Pairwise component diagnostics
- {'rot_mean_deg': 20.835182098696297, 'rot_median_deg': 20.75245218725162, 'rot_p90_deg': 21.448424566634742, 'tdir_mean_deg': 99.54956140857578, 'tdir_median_deg': 57.48547914456783, 'tdir_p90_deg': 160.73854903392154, 'tdir_abs_mean_deg': 41.117886773689875, 'tdir_abs_median_deg': 47.69726053649836, 'tdir_abs_p90_deg': 55.4396494071546, 'tmag_mean_ratio': 2.086847196715845, 'tmag_median_ratio': 0.9819414718338607}

## Replay protocol and coverage
- selected variant: `model_convention_as_declared`
- num poses: `68` / 454

## Existing dense restricted to selected edges
- {'computed': False, 'num_edges': 0, 'rot_mean_deg': nan, 'rot_median_deg': nan, 'rot_p90_deg': nan, 'tdir_mean_deg': nan, 'tdir_median_deg': nan, 'tdir_p90_deg': nan, 'tdir_abs_mean_deg': nan, 'tdir_abs_median_deg': nan, 'tdir_abs_p90_deg': nan, 'tmag_mean_ratio': nan, 'tmag_median_ratio': nan}

## Replay vs existing dense comparison
- {'compared': False, 'num_common_edges': 0, 'relative_rot_diff_mean_deg': nan, 'relative_tdir_diff_mean_deg': nan, 'relative_tmag_ratio_median': nan, 'path_length_ratio_replay_over_existing': nan, 'dense_export_consistent_with_selected_pairwise': None, 'dense_export_or_integration_mismatch_suspected': None}

## Gap diagnosis
- {'selected_pairwise_tdir_issue_confirmed': True, 'selected_pairwise_tmag_good': True, 'dense_tmag_mismatch_on_selected_edges': False, 'integration_gap_confirmed': None, 'most_likely_gap_source': 'selected_pairwise_tdir', 'evidence': ['pairwise tdir mean=99.54956140857578', 'pairwise tmag median ratio=0.9819414718338607', 'dense(selected) tmag median ratio=nan', 'components=19'], 'recommended_next_actions': ['Run S5D8 inverse-variant replay diagnostics side-by-side in downstream analysis.', 'Add explicit selected_k1 edge list export to official diagnostic path for reproducible replay.', 'For dense conclusions, compare on identical edge protocol before path-level claims.', 'Keep selected_k1 sparse protocol caveat in all tdir/tmag reporting.']}

## Caveats
- diagnostic only
- selected_k1 sparse protocol
- does not replace official S5 locked result
- S5 locked metrics/policy unchanged
