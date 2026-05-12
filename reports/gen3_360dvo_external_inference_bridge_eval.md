# GEN3 360DVO external inference bridge eval

## 1. 为什么做 GEN3
GEN2B 已经证明 360DVO field one-sequence 的数据接入、pose convention 和 motion audit 都可以成立，因此 GEN3 的目标是确认 S5E15-like 路径能否真实对外部 ERP pair 做推理。

## 2. GEN2B 数据和 motion audit 摘要
{
  "num_pairs": 100,
  "gt_step_median": 1.1221688852924325,
  "gt_step_mean": 1.1227872974709823,
  "gt_step_p10": 0.3778633395147681,
  "gt_step_p25": 0.7444670891145597,
  "gt_step_p50": 1.1221688852924325,
  "gt_step_p75": 1.8753056274114026,
  "gt_step_p90": 1.8908075712689083,
  "gt_step_p95": 2.262509924036204,
  "small_motion_fraction": 0.0,
  "very_small_motion_fraction": 0.0,
  "path_length": 112.27872974709823,
  "rotation_step_mean": 0.632855073162311,
  "rotation_step_p90": 1.40334445167258,
  "train_eval_sequence_split_feasibility": false,
  "compared_to_data2_seq03": {
    "data2_seq03_gt_step_median": 0.00750599760191659,
    "data2_seq03_small_motion_fraction": 0.38852097130242824,
    "data2_seq03_path_length": 25.833639521294945,
    "gt_step_median_ratio_over_data2_seq03": 149.50296347095778,
    "small_motion_fraction_delta_vs_data2_seq03": -0.38852097130242824,
    "path_length_ratio_over_data2_seq03": 4.346221896242907
  },
  "vs_data2_seq03": {
    "data2_seq03_gt_step_median": 0.00750599760191659,
    "data2_seq03_small_motion_fraction": 0.38852097130242824,
    "data2_seq03_path_length": 25.833639521294945,
    "gt_step_median_ratio_over_data2_seq03": 149.50296347095778,
    "small_motion_fraction_delta_vs_data2_seq03": -0.38852097130242824,
    "path_length_ratio_over_data2_seq03": 4.346221896242907
  },
  "larger_or_more_stable_than_data2_seq03": true,
  "suitable_for_tdir_generalization_eval": false,
  "small_motion_risk_like_data2": false,
  "blockers": []
}

## 3. S5E15-like inference availability audit
{
  "s5e15_checkpoint_found": true,
  "s5e15_model_weights_found": false,
  "image_pair_forward_available": false,
  "external_erp_input_supported": false,
  "depends_on_scene01_artifacts": true,
  "uses_eval_gt_for_prediction": false,
  "can_predict_R_tdir_tmag": true,
  "inference_bridge_ready": false,
  "bridge_classification": "SCENE_SPECIFIC_EXPORT_ONLY",
  "evidence": {
    "source_candidate": "S5E15_scale_deunderfit_antiparallel_candidate",
    "candidate_json": "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json",
    "candidate_dir": "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate",
    "training_status_exists": true,
    "export_tool": "tools/export_s5e15_adjacent_dense_predictions.py",
    "adjacent_dense_export_available": true,
    "training_status_classification": "S5E15_TRAIN_PRIOR_SCALE_ONLY"
  },
  "blockers": [
    "MODEL_WEIGHTS_MISSING",
    "SCENE_SPECIFIC_EXPORT_ONLY",
    "IMAGE_PAIR_FORWARD_MISSING",
    "INPUT_PROTOCOL_UNSUPPORTED"
  ]
}

## 4. 是否能真实预测 360DVO pair
{
  "model_eval_available": false,
  "num_pairs_predicted": 0,
  "num_pairs_total": 100,
  "pair_coverage": 0.0,
  "adjacent_pair_coverage": 0.0,
  "kstep_pair_coverage": 0.0,
  "eval_blocker": "SCENE_SPECIFIC_EXPORT_ONLY"
}

## 5. 如果 blocked，blocker 是什么
{
  "bridge_classification": "SCENE_SPECIFIC_EXPORT_ONLY",
  "eval_blocker": "SCENE_SPECIFIC_EXPORT_ONLY"
}

## 6. 如果成功，component metrics
{}

## 7. 如果成功，trajectory / path_ratio / ATE
{
  "t_path": "external_baselines/results/gen3_360dvo_external_eval/scene_field_s5e15_like_est_tum.txt",
  "none": null,
  "se3": null,
  "sim3": null,
  "path_ratio": null
}

## 8. 与 DATA2 scene01/seq03 的解释性对比
{
  "gen2b_motion_summary": {
    "num_pairs": 100,
    "gt_step_median": 1.1221688852924325,
    "gt_step_mean": 1.1227872974709823,
    "gt_step_p10": 0.3778633395147681,
    "gt_step_p25": 0.7444670891145597,
    "gt_step_p50": 1.1221688852924325,
    "gt_step_p75": 1.8753056274114026,
    "gt_step_p90": 1.8908075712689083,
    "gt_step_p95": 2.262509924036204,
    "small_motion_fraction": 0.0,
    "very_small_motion_fraction": 0.0,
    "path_length": 112.27872974709823,
    "rotation_step_mean": 0.632855073162311,
    "rotation_step_p90": 1.40334445167258,
    "train_eval_sequence_split_feasibility": false,
    "compared_to_data2_seq03": {
      "data2_seq03_gt_step_median": 0.00750599760191659,
      "data2_seq03_small_motion_fraction": 0.38852097130242824,
      "data2_seq03_path_length": 25.833639521294945,
      "gt_step_median_ratio_over_data2_seq03": 149.50296347095778,
      "small_motion_fraction_delta_vs_data2_seq03": -0.38852097130242824,
      "path_length_ratio_over_data2_seq03": 4.346221896242907
    },
    "vs_data2_seq03": {
      "data2_seq03_gt_step_median": 0.00750599760191659,
      "data2_seq03_small_motion_fraction": 0.38852097130242824,
      "data2_seq03_path_length": 25.833639521294945,
      "gt_step_median_ratio_over_data2_seq03": 149.50296347095778,
      "small_motion_fraction_delta_vs_data2_seq03": -0.38852097130242824,
      "path_length_ratio_over_data2_seq03": 4.346221896242907
    },
    "larger_or_more_stable_than_data2_seq03": true,
    "suitable_for_tdir_generalization_eval": false,
    "small_motion_risk_like_data2": false,
    "blockers": []
  },
  "avoids_small_motion_risk": true,
  "s5e15_scene01_seq03_reference": {
    "count": 453,
    "rot_mean_deg": 0.9134395040767528,
    "rot_median_deg": 0.797895569319923,
    "rot_p90_deg": 1.3218302317546538,
    "signed_tdir_mean_deg": 50.353041365033235,
    "signed_tdir_median_deg": 41.2294804093569,
    "signed_tdir_p90_deg": 101.804771450087,
    "tdir_abs_mean_deg": 43.16064629037026,
    "tdir_abs_median_deg": 39.87455596995841,
    "tdir_abs_p90_deg": 78.632504335116,
    "tdir_mean_cosine": 0.5591010842073055,
    "anti_parallel_rate": 0.1479028697571744,
    "severe_wrong_sign_rate": 0.05518763796909492,
    "direction_abs_good_but_signed_bad_rate": 0.024282560706401765,
    "tmag_median_ratio": 1.268517401538299,
    "tmag_mean_ratio": 1.6886134029934055,
    "tmag_p90_ratio": 3.852362321212163,
    "tmag_p95_ratio": 4.66860108610276,
    "tmag_p99_ratio": 6.775825973441673,
    "tmag_max_ratio": 19.819619392350763,
    "path_ratio": 0.1571668166639846
  }
}

## 9. 是否进入 GEN4
{
  "keep_s5e15_as_best_candidate": true,
  "do_gen4_external_expansion": true,
  "do_external_training": false,
  "main_next_step": "build_true_image_pair_external_bridge_or_reuse_lower_level_npz_models"
}

## 10. caveats
- 本轮不训练、不 fine-tune、不生成新训练候选。
- 不使用 360DVO GT 做 calibration。
- 如果 inference bridge 不可用，绝不伪造 prediction。
