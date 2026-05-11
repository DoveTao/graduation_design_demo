# S5D4 Reconcile Old tdir With Dense tdir

## Executive summary

Final classification: `S5D4_OLD_TDIR_SOURCE_NOT_FOUND`
old_tdir_20_source_status: `UNKNOWN_SOURCE`

## The question

User remembered old tdir around 20 deg, while current dense external trajectory-derived tdir is around 90-100 deg.
This audit reconciles metric provenance and convention differences only.

## Search results

- keyword log: `logs/s5d4_tdir_keyword_search.log`
- twenty-degree log: `logs/s5d4_tdir_20deg_search.log`
- candidates: `80`

## Candidate provenance table

| candidate_id | file | metric | value | method | sequence | classification | explanation |
|---|---|---|---:|---|---|---|---|
| candidate_001 | checkpoints/S16_stronger_visual_backbone_feasibility_candidates.json | "tdir_angular_error_val" | 119.24411418633139 | other | unknown | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_002 | checkpoints/S16_stronger_visual_backbone_feasibility_candidates.json | "current_model_mean_tdir_err" | 19.006343561378138 | other | unknown | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_003 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir" | 19.243048667907715 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_004 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_abs" | 19.243048667907715 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_005 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_local_A" | 20.01670551300049 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_006 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_local_A_abs" | 20.01670551300049 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_007 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir" | 19.09162425994873 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_008 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_abs" | 19.09162425994873 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_009 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_local_A" | 19.828197479248047 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_010 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_local_A_abs" | 19.828197479248047 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_011 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir" | 19.167336463928223 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_012 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_abs" | 19.167336463928223 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_013 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_local_A" | 19.922451496124268 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_014 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_local_A_abs" | 19.922451496124268 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_015 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir" | 19.243048667907715 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_016 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_abs" | 19.243048667907715 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_017 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_local_A" | 20.01670551300049 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_018 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_local_A_abs" | 20.01670551300049 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_019 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir" | 19.09162425994873 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |
| candidate_020 | checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json | "tdir_abs" | 19.09162425994873 | other | scene01_seq03 | OLD_TDIR_PAIRWISE_LOCAL_FRAME | tdir under different protocol |

## Current dense tdir reproduction

- dense_Twc_adjacent tdir mean/median/p90: `91.392419` / `95.679859` / `139.551260`
- matches S5D2: `True`

## Convention variant results

- dense_Twc_adjacent: `{'num_pairs': 453, 'num_valid_pairs': 453, 'rot_mean_deg': 20.82515115155671, 'tdir_mean_deg': 91.39241866186, 'tdir_median_deg': 95.67985920477633, 'tdir_p90_deg': 139.55126037861987, 'tdir_mean_cosine': -0.028522221625`
- dense_Tcw_adjacent: `{'num_pairs': 453, 'num_valid_pairs': 453, 'rot_mean_deg': 20.82515115155671, 'tdir_mean_deg': 91.39241866186217, 'tdir_median_deg': 95.67985920479926, 'tdir_p90_deg': 139.5512603786417, 'tdir_mean_cosine': -0.0285222216`
- world_delta_adjacent: `{'num_pairs': 453, 'num_valid_pairs': 453, 'rot_mean_deg': 0.0, 'tdir_mean_deg': 85.63331890482517, 'tdir_median_deg': 85.32171669386678, 'tdir_p90_deg': 137.98472601483408, 'tdir_mean_cosine': 0.06162735756169902, 'tmag`
- axis_flip_sweep: `{'best': {'sign': [1.0, -1.0, 1.0], 'num_pairs': 453, 'num_valid_pairs': 453, 'rot_mean_deg': 20.82515115155671, 'tdir_mean_deg': 77.66299027098084, 'tdir_median_deg': 76.58262207693912, 'tdir_p90_deg': 128.3007244696135`
- axis_permutation_and_sign_sweep: `{'best': {'perm': [1, 2, 0], 'sign': [1.0, -1.0, -1.0], 'num_pairs': 453, 'num_valid_pairs': 453, 'rot_mean_deg': 20.82515115155671, 'tdir_mean_deg': 65.80605778539095, 'tdir_median_deg': 62.75409957376085, 'tdir_p90_deg`
- k_step_pairs: `{'k1': {'num_pairs': 453, 'num_valid_pairs': 453, 'rot_mean_deg': 20.82515115155671, 'tdir_mean_deg': 91.39241866186, 'tdir_median_deg': 95.67985920477633, 'tdir_p90_deg': 139.55126037861987, 'tdir_mean_cosine': -0.02852`
- selected_sparse_chain_if_available: `{'available': False, 'result': {}, 'notes': 'unavailable in current branch artifacts'}`
- component_prediction_if_available: `{'available': False, 'result': {}, 'notes': 'no direct pairwise prediction artifact located'}`

## Interpretation

- Most 20deg candidates map to rot_mean_deg or historical pairwise/local tdir_abs, not current dense external trajectory-derived tdir.
- dense_Twc_adjacent reproduces S5D2 high tdir, consistent with current external-dense protocol.

## Caveats

- S5D4 does not modify predictions.
- S5D4 does not replace official S5 locked result.
- S5 locked metrics/policy were not changed.
- Any axis/sign/permutation sweep is diagnostic only and must not be applied without explicit convention proof.
