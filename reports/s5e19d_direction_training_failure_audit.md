# S5E19D direction training failure audit

## 执行摘要
- final_classification = `S5E19D_MIXED_FAILURE_MODES`

## 为什么要做 S5E19D
S5E19B 已经排除 smoke/no-op/fallback 之外的实现污染后，S5E19C 证明 direction head 确实在输出，但结果反而伤害了 S5E15，需要解释根因。

## S5E19C 已经排除 no-op/fallback 的证据
{
  "delta_tdir_zero_count": 0,
  "delta_tdir_norm_mean": 0.6029416022053354,
  "sign_score_unique_count": 453,
  "final_equals_s5e15_direction_count": 0,
  "fallback_to_s5e15_direction": false,
  "learned_weights_loaded": true,
  "export_used_learned_weights": true,
  "evaluator_hardcoded_s5e15_metrics": false
}

## S5E19C 相对 S5E15 变差的指标
{
  "s5e15_reference": {
    "tdir_mean_deg": 50.353,
    "anti_parallel_rate": 0.1479,
    "path_ratio": 0.1572,
    "sim3_ate": 3.9682
  },
  "s5e19c_result": {
    "tdir_mean_deg": 56.17528018993624,
    "anti_parallel_rate": 0.16556291390728478,
    "path_ratio": 0.12240164100086463,
    "sim3_ate": 3.9958080586973814
  }
}

## training path audit
{
  "real_training_executed": true,
  "optimizer_step_count": 180,
  "learned_weights_saved": true,
  "losses_present": {
    "signed_tdir": true,
    "tdir_abs_aux": true,
    "anti_parallel_hard_negative": true,
    "sign_score_bce": true,
    "robust_tmag_log": true,
    "multiframe_composition": true,
    "path_length_consistency": true
  },
  "losses_enter_total": {
    "signed_tdir": true,
    "tdir_abs_aux": true,
    "anti_parallel_hard_negative": true,
    "sign_score_bce": true,
    "robust_tmag_log": true,
    "multiframe_composition": true,
    "path_length_consistency": true
  },
  "loss_weight_risks": [
    "sign_score_bce 以 coarse-vs-gt sign 作为目标，但只占 0.2 权重，且不直接驱动 final_tdir 翻转。",
    "multiframe_composition 权重较小，但实现方式可能并非真实 k-step 几何。",
    "scale/path loss 权重合计高于 sign/anti_parallel 单项，可能压过 direction 修正。"
  ],
  "detach_or_no_grad_risks": [
    "脚本存在 detach/no_grad，需要逐项确认是否影响 direction 主干。"
  ],
  "split_integrity_ok": true,
  "smoke_policy_only": false
}

## delta_tdir audit
{
  "delta_tdir_zero_count": 0,
  "delta_tdir_norm_mean": 0.6029416024169616,
  "delta_tdir_norm_median": 0.6060211101436472,
  "delta_tdir_norm_p90": 0.6061971391137733,
  "delta_tdir_norm_p95": 0.606199093638023,
  "delta_tdir_norm_max": 0.6062014266923509,
  "delta_tdir_too_aggressive": true,
  "mean_angle_delta_to_ideal_correction": 105.55529237985583,
  "delta_moves_toward_gt_fraction": 0.23841059602649006,
  "delta_moves_away_from_gt_fraction": 0.7615894039735099,
  "diagnosis": "delta_too_large"
}

## sign_score audit
{
  "sign_score_unique_count": 453,
  "sign_score_min": 0.35212433338165283,
  "sign_score_max": 3.562387228012085,
  "sign_score_mean": 0.7424892446326368,
  "sign_score_std": 0.4134010834792655,
  "correlation_with_antiparallel": -0.15218990644413732,
  "auc_if_computable": null,
  "high_score_tdir_mean": 43.04593670882843,
  "low_score_tdir_mean": 59.24602043041803,
  "correct_sign_score_mean": 0.7705140233670593,
  "wrong_sign_score_mean": 0.6012443598111471,
  "sign_head_informative": false
}

## train/eval generalization audit
{
  "train_metrics_available": false,
  "train_tdir_improved": null,
  "eval_tdir_degraded": true,
  "generalization_gap_suspected": null,
  "evidence": [
    "history_tail val signed_tdir_mean range = 55.333..57.977",
    "history_tail val anti_parallel_rate range = 0.177..0.215",
    "缺少 train split 预测/metrics，无法证明 train 改善后 eval 变差，只能确认 val/eval 没学好。"
  ]
}

## frame convention audit
{
  "tdir_pred_frame": "camera",
  "tdir_gt_frame": "camera",
  "twc_tcw_mismatch_suspected": false,
  "quaternion_order_risk": false,
  "composition_order_risk": false,
  "normalize_logic_risk": false,
  "frame_convention_suspected": false,
  "evidence": [
    "训练 GT 使用 relative_pose_A_to_B_in_B，属于 B/local camera frame。",
    "导出和 TUM 积分也使用 B/local 相对位姿再写回 world trajectory。",
    "quaternion 使用 xyzw 顺序，和项目公共工具一致。"
  ]
}

## observability weighting audit
{
  "weights_available": false,
  "weight_distribution": {
    "proxy_confidence_mean": 0.7932038511188587,
    "proxy_confidence_std": 0.18028860377564696,
    "proxy_q25": 0.600126562116202,
    "proxy_q75": 1.0
  },
  "high_weight_tdir_mean": 52.94794809175225,
  "low_weight_tdir_mean": 55.90970679125939,
  "high_weight_antiparallel_rate": 0.12804878048780488,
  "weighting_helpful": true,
  "weighting_bad": false,
  "notes": [
    "S5E19C 本身没有 observability_weight；这里仅用 S5E15 direction_confidence 作为代理。"
  ]
}

## multiframe loss audit
{
  "k1_tdir": 50.35304148245505,
  "k2_tdir": 48.066943689366354,
  "k3_tdir": 47.774063258873134,
  "k5_tdir": 47.298629760079265,
  "kstep_improves_direction": true,
  "composition_loss_conflict_suspected": true,
  "path_length_consistency_helpful": null,
  "diagnosis": "conflicts_with_k1"
}

## root cause diagnosis
把 multiframe loss 改成真实 k-step 组合监督，并移除随机 batch 相邻样本惩罚。

## 是否继续 S5E20
{
  "continue_to_s5e20": true,
  "recommended_next_stage": "S5E20_multiframe_supervision_only",
  "keep_s5e15_as_best_candidate": true,
  "s5e19c_should_replace_s5e15": false,
  "main_fix": "把 multiframe loss 改成真实 k-step 组合监督，并移除随机 batch 相邻样本惩罚。"
}

## caveats
- 本任务不训练模型。
- 不刷新指标。
- 不替代 S5E15。
- S5 locked metrics/policy unchanged。

## validation
- static_test: `PASS`
- run_serial_validation_guard: `HANG_WITH_PASS_SUBLOGS`

## git
- commits_created: `830a73d`, `7764b53`
- pushed_to_remote: `true`
- working_tree_clean: `true`
