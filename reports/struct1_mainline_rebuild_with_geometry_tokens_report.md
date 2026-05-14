# STRUCT1 主结构重构报告

## 1. 为什么从 ARCH2P 进入 STRUCT1
ARCH2P 已经验证参数能够改变 delta 和 modified fraction，但没有任何 valid run 同时超过 S5E15 的 tdir / anti_parallel / path_ratio / sim3，因此需要进入更高层级的主结构重构。

## 2. STRUCT1 三段式主结构
STRUCT1 由球面 token backbone、soft correspondence geometry layer、geometry token pose solver 组成，并保留 coarse-to-fine。

## 3. 球面 token backbone 改造
新增 local tangent x/y、bearing xyz、lat/lon sincos 通道，保留 bearing metadata。

## 4. soft correspondence geometry layer
W_ab / W_ba 不再只是中间 attention，而是 geometry token 的直接来源，并显式输出 confidence / entropy / cycle / epipolar。

## 5. geometry token pose solver
tdir head 只从 geometry token summary 输出，避免 pooled appearance only tdir。

## 6. coarse-to-fine 细化
coarse_tokens=64, fine_tokens=24

## 7. 几何一致性辅助任务
保留 supervised pose anchor，同时加入 cycle consistency、epipolar residual、inverse consistency、true k-step composition、path length consistency 和 confidence entropy regularization。

## 8. training status
{
  "real_training_executed": true,
  "optimizer_step_count": 297,
  "learned_weights_saved": true,
  "smoke_policy_only": false,
  "uses_eval_gt_for_training": false,
  "uses_orbslam3_teacher": false,
  "uses_geometry_tokens": true,
  "W_ab_used_in_pose_solver": true,
  "tdir_from_geometry_tokens": true,
  "fine_stage_used": true,
  "coarse_token_count": 64,
  "fine_token_count": 24,
  "softcorr_temperature": 0.05,
  "classification": "STRUCT1_SHORT_REAL_TRAINING"
}

## 9. integrity audit
{
  "experiment": "STRUCT1_mainline_rebuild_with_geometry_tokens",
  "spherical_token_backbone_used": true,
  "tangent_geometric_channels_used": true,
  "W_ab_W_ba_produced": true,
  "W_ab_used_in_pose_solver": true,
  "geometry_tokens_nonzero": true,
  "tdir_from_geometry_tokens": true,
  "no_fallback_to_S5E15": true,
  "no_old_metrics_reuse": true,
  "pose_convention_B_frame_consistent": true,
  "fine_stage_used": true,
  "eval_GT_not_used_for_prediction": true,
  "ORB_teacher_not_used": true,
  "audit_details": {
    "confidence_mean": 0.6398040622012241,
    "softcorr_entropy_mean": 1.1447220895990367,
    "cycle_error_mean": 0.050072431243610695,
    "coverage": {
      "num_poses": 454,
      "num_edges": 453,
      "all_edges_traceable": true,
      "coverage": 1.0
    }
  },
  "integrity_pass": true
}

## 10. component metrics
{
  "rot_mean_deg": 0.9314774626005936,
  "rot_median_deg": 0.8000763487741889,
  "rot_p90_deg": 1.3749724459172628,
  "signed_tdir_mean_deg": 76.64506725187314,
  "signed_tdir_median_deg": 73.84564500230695,
  "signed_tdir_p90_deg": 116.61011699246903,
  "tdir_abs_mean_deg": 62.17215430000165,
  "tdir_abs_median_deg": 64.35891353139347,
  "tdir_abs_p90_deg": 85.16363498686458,
  "anti_parallel_rate": 0.33774834437086093,
  "tmag_median_ratio": 23.345353231395972,
  "tmag_p95_ratio": 188.60878004568698,
  "path_ratio": 12.970194691802755,
  "geometry_token_confidence_mean": 0.6398040622012241,
  "softcorr_entropy_mean": 1.1447220895990367,
  "cycle_consistency_mean": 0.050072431243610695,
  "epipolar_residual_mean": 0.03245164000246238
}

## 11. external evaluator
{
  "none": {
    "ate": 168.78684542237974,
    "drift": 3.8572867188824542,
    "path_ratio": 12.97020608133707
  },
  "se3": {
    "ate": 108.893584987358,
    "drift": 3.857733482014568,
    "path_ratio": 12.97020608133707
  },
  "sim3": {
    "ate": 4.125350624694253,
    "drift": 0.13106401982646956,
    "path_ratio": 12.97020608133707
  }
}

## 12. 与 S5E15 / ARCH2P / ARCH2 / ORB-SLAM3 对比
{
  "signed_tdir_improved": false,
  "anti_parallel_reduced": false,
  "path_ratio_improved": true,
  "sim3_ate_improved": false,
  "high_confidence_subset_tdir_improved": false
}

## 13. 是否 promote STRUCT1
STRUCT1_REAL_TRAINING_NO_IMPROVEMENT

## 14. caveats
- 仍属于实验候选，不替代 official locked S5。
- 不使用 eval GT calibration。
- 不使用 ORB-SLAM3 作为 teacher。
