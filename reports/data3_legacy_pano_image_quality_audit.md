# DATA3 legacy 全景图像质量审计

## 1. 审计动机
DATA2 已显示旧数据存在明显 motion shift 风险，因此需要进一步确认 legacy 图像质量是否也是 tdir / tmag / path 退化的重要来源。

## 2. 图像质量指标定义
{
  "metric_definitions": [
    "low_texture_fraction",
    "blur_score_laplacian",
    "seam_score_vertical",
    "seam_score_horizontal",
    "exposure_discontinuity",
    "gradient_feature_density",
    "bright_low_texture_area_proxy",
    "pair_photometric_change",
    "image_quality_score"
  ]
}

## 3. 每个旧序列质量统计
{
  "scene01/seq01": {
    "num_frames": 324,
    "sampled_frames": 200,
    "mean_metrics": {
      "low_texture_fraction": 0.7246161270141601,
      "blur_score_laplacian": 0.0030507232423406094,
      "seam_score_vertical": 0.00890476955100894,
      "seam_score_horizontal": 0.020563891413621603,
      "exposure_discontinuity": 0.07397740125656128,
      "gradient_feature_density": 0.10864307403564454,
      "bright_low_texture_area_proxy": 0.08342113494873046,
      "image_quality_score": 0.7514627014577855
    },
    "pair_photometric_change_mean": 0.01360034072387688
  },
  "scene01/seq02": {
    "num_frames": 464,
    "sampled_frames": 200,
    "mean_metrics": {
      "low_texture_fraction": 0.731297607421875,
      "blur_score_laplacian": 0.0031069623318035157,
      "seam_score_vertical": 0.00706750460434705,
      "seam_score_horizontal": 0.028410066636279225,
      "exposure_discontinuity": 0.12128484457731246,
      "gradient_feature_density": 0.11216636657714844,
      "bright_low_texture_area_proxy": 0.07530345916748046,
      "image_quality_score": 0.7307403955436312
    },
    "pair_photometric_change_mean": 0.013451382394013668
  },
  "scene01/seq03": {
    "num_frames": 454,
    "sampled_frames": 200,
    "mean_metrics": {
      "low_texture_fraction": 0.7170674133300782,
      "blur_score_laplacian": 0.0029834799095988275,
      "seam_score_vertical": 0.0072671828116290275,
      "seam_score_horizontal": 0.022516519245691598,
      "exposure_discontinuity": 0.06381387740373612,
      "gradient_feature_density": 0.11978961944580079,
      "bright_low_texture_area_proxy": 0.05954460144042969,
      "image_quality_score": 0.7666832457239506
    },
    "pair_photometric_change_mean": 0.013377628528003734
  }
}

## 4. 低质量图像与 tdir/tmag/path error 的关系
{
  "quality_vs_s5e15_signed_tdir": -0.29002125942291923,
  "quality_vs_s5e15_tmag_ratio": -0.40133852984397295,
  "quality_vs_struct1b_signed_tdir": -0.09522221840956516,
  "quality_vs_struct1b_tmag_ratio": 0.02123656770571913,
  "quality_vs_struct1b_path_contribution": 0.04490230379405749
}

## 5. 是否确认 legacy 数据质量风险
- final_classification = DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED

## 6. 对训练失败的解释
如果 seq03 同时表现出高 low-texture / bright-low-texture 区域与较差方向/尺度误差相关性，则 legacy 数据质量很可能是 DATA2/STRUCT1B 失败的重要背景因素。

## 7. 建议
{
  "downgrade_legacy_dataset_to_diagnostic": true,
  "use_360dvo_as_main_dataset": true
}
