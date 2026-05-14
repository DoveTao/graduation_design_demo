# STRUCT1B scale guard 修复报告

## 执行摘要
final_classification=STRUCT1B_SCALE_GUARD_NOT_EFFECTIVE

## 前提说明
由于原始 `STRUCT1` 训练权重 `struct1_model.pt` 已不在当前工作区，本轮 `STRUCT1B` 采用与 STRUCT1 相同的 geometry-token 主干和相同 tdir head 结构重新训练，只修复 scale guard 与 scale/path 权重，不重设计 tdir head。

## scale repair 结果
{
  "rot_mean_deg": 0.9153562920646506,
  "signed_tdir_mean_deg": 79.71209808275077,
  "anti_parallel_rate": 0.3554083885209713,
  "tmag_median_ratio": 11.866910847001154,
  "tmag_p95_ratio": 94.01508919525963,
  "path_ratio": 6.39102545028478
}

## external evaluator
{
  "none": {
    "ate": 98.36928358217783,
    "drift": 1.998011855528178,
    "path_ratio": 6.391023106917223
  },
  "se3": {
    "ate": 59.83886749016649,
    "drift": 1.9986010749372427,
    "path_ratio": 6.391023106917223
  },
  "sim3": {
    "ate": 4.116319654825962,
    "drift": 0.13120271110228535,
    "path_ratio": 6.391023106917223
  }
}

## recommendation
{
  "keep_s5e15_as_best_candidate": true,
  "promote_struct1": false,
  "continue_to_struct2": false,
  "recommended_next_stage": "stop_STRUCT1_and_STRUCT2"
}
