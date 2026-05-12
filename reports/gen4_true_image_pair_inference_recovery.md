# GEN4 真正 image-pair inference 路径恢复审计

## 1. 执行摘要
- final_classification = GEN4_TRUE_IMAGE_PAIR_INFERENCE_READY

## 2. S5E15 是否只是 derived export candidate
{
  "candidate_json_found": true,
  "training_status_found": true,
  "dedicated_model_weights_found": false,
  "derived_export_tool_found": true,
  "depends_on_scene01_artifacts": true,
  "uses_eval_gt_for_prediction": false,
  "can_predict_R_tdir_tmag": true,
  "conclusion": "S5E15 is a valid derived export candidate but not fully reusable external inference model."
}

## 3. 是否存在低层级真实模型 checkpoint
{
  "generic_model_code_found": true,
  "train_mvp_checkpoint_load_path_found": true,
  "model_py_forward_found": true,
  "dataset_py_found": false,
  "candidate_inventory": {
    "s5e2_npz": "/home/dovetao/graduation_design_demo/checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz",
    "s5e3_npz": "/home/dovetao/graduation_design_demo/checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz",
    "s5e5_pt": "/home/dovetao/graduation_design_demo/checkpoints/S5E5_temporal_visual_backbone_geometry_candidate/s5e5_temporal_visual_best.pt",
    "s5e13_pt": "/home/dovetao/graduation_design_demo/checkpoints/S5E13_real_correspondence_signed_direction_candidate/s5e13_best.pt",
    "s5e15_candidate_json": "/home/dovetao/graduation_design_demo/checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json",
    "s5e15_candidate_dir": "/home/dovetao/graduation_design_demo/checkpoints/S5E15_scale_deunderfit_antiparallel_candidate",
    "t57b_final_pt": "/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt"
  },
  "selected_recovered_candidate": "T57b_no_dt_multiscale_tmag_head_400/final.pt",
  "t57b_forward_probe": {
    "checkpoint_path": "/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt",
    "cfg_H": 1024,
    "cfg_W": 2048,
    "cfg_in_ch": 3,
    "load_missing_keys": 14,
    "load_unexpected_keys": 0,
    "forward_success": true,
    "output_has_R": true,
    "output_has_tdir": true,
    "output_has_tmag": true,
    "output_has_tvec": true,
    "translation_local_frame": "A",
    "translation_output_frame": "B",
    "stage": "coarse_only",
    "sample_R_pred_BA": [
      [
        0.9998498558998108,
        -0.01713799312710762,
        0.002159301657229662
      ],
      [
        0.017193851992487907,
        0.9994333386421204,
        -0.028899317607283592
      ],
      [
        -0.0016627887962386012,
        0.028932157903909683,
        0.9995779395103455
      ]
    ],
    "sample_tdir_pred": [
      0.5801939964294434,
      0.6550771594047546,
      -0.48399266600608826
    ],
    "sample_tmag_pred": 0.12955550849437714,
    "sample_tvec_pred": [
      0.07356623560190201,
      0.08792537450790405,
      -0.0603470616042614
    ],
    "uses_gt_for_prediction": false,
    "uses_360dvo_gt_for_calibration": false
  }
}

## 4. train_mvp / model.py 的通用 forward path
- `PanoramaRelPoseModel` 在 `model.py` 中提供 `IA, IB -> (R_pred, t_pred, aux)` 的通用 forward。
- `train_mvp.py` 提供从 `.pt` checkpoint 恢复 `cfg` 和 `model state_dict` 的通用加载路径。

## 5. 是否能对 360DVO pair 做真实 inference
{
  "checkpoint_path": "/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt",
  "cfg_H": 1024,
  "cfg_W": 2048,
  "cfg_in_ch": 3,
  "load_missing_keys": 14,
  "load_unexpected_keys": 0,
  "forward_success": true,
  "output_has_R": true,
  "output_has_tdir": true,
  "output_has_tmag": true,
  "output_has_tvec": true,
  "translation_local_frame": "A",
  "translation_output_frame": "B",
  "stage": "coarse_only",
  "sample_R_pred_BA": [
    [
      0.9998498558998108,
      -0.01713799312710762,
      0.002159301657229662
    ],
    [
      0.017193851992487907,
      0.9994333386421204,
      -0.028899317607283592
    ],
    [
      -0.0016627887962386012,
      0.028932157903909683,
      0.9995779395103455
    ]
  ],
  "sample_tdir_pred": [
    0.5801939964294434,
    0.6550771594047546,
    -0.48399266600608826
  ],
  "sample_tmag_pred": 0.12955550849437714,
  "sample_tvec_pred": [
    0.07356623560190201,
    0.08792537450790405,
    -0.0603470616042614
  ],
  "uses_gt_for_prediction": false,
  "uses_360dvo_gt_for_calibration": false
}

## 6. 如果不能，blocker 是什么；如果能，下一步是什么
{
  "do_gen5_360dvo_true_external_eval": true,
  "keep_s5e15_as_best_candidate": true,
  "main_next_step": "GEN5_360DVO_true_external_eval"
}

## 7. 论文 caveat
当前 S5E15 是 valid derived candidate，但它本身不是 fully reusable external inference model；真正可复用的通用推理入口来自更底层的 checkpoint / model code path。

## 8. caveats
- 本轮不训练，不 fine-tune。
- no fine-tune, no external retraining, no 360DVO GT calibration.
- 不使用 360DVO GT calibration。
- 不伪造 external prediction。
