# DSET1 360DVO 主数据集迁移报告

## 1. 为什么更换数据集
旧数据集在 DATA2 中已经显示出明显 motion distribution shift，而 DATA3 进一步审计图像质量风险；因此需要评估 360DVO 是否可以升级为新主数据集。

## 2. 旧数据集问题证据
{
  "data2_motion_shift": {
    "gt_step_median_ratio_eval_over_train": 0.6303086823480436,
    "small_motion_fraction_delta": 0.12770672193856053,
    "path_length_ratio_eval_over_train": 0.49415517458410535,
    "distribution_shift_suspected": true
  },
  "data3_quality": "DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED"
}

## 3. 360DVO 下载/挂载状态
{
  "dataset_repo_reachable": true,
  "local_root": "data/360DVO",
  "selected_sequences": [
    "bridge_night",
    "field",
    "wingsuit"
  ],
  "download_attempted": true,
  "download_complete": true,
  "total_bytes_downloaded": 200577677,
  "access_blocked": false,
  "dependency_blocked": false,
  "blockers": [],
  "downloaded_sequences": [
    {
      "seq_id": "bridge_night",
      "num_images": 80,
      "groundtruth": "data/360DVO/GroundTruth/bridge_night.txt",
      "timestamps": "data/360DVO/Timestamps/bridge_night_timestamps.txt"
    },
    {
      "seq_id": "field",
      "num_images": 178,
      "groundtruth": "data/360DVO/GroundTruth/field.txt",
      "timestamps": "data/360DVO/Timestamps/field_timestamps.txt"
    },
    {
      "seq_id": "wingsuit",
      "num_images": 47,
      "groundtruth": "data/360DVO/GroundTruth/wingsuit.txt",
      "timestamps": "data/360DVO/Timestamps/wingsuit_timestamps.txt"
    }
  ]
}

## 4. 360DVO 多序列 manifest
{
  "dataset": "360DVO",
  "local_root": "data/360DVO",
  "selected_sequences": [
    "bridge_night",
    "field",
    "wingsuit"
  ],
  "num_sequences": 3,
  "split": {
    "train": [
      "bridge_night"
    ],
    "val": [
      "field"
    ],
    "test": [
      "wingsuit"
    ]
  },
  "split_by_sequence": true,
  "forbid_random_pair_split": true,
  "num_pairs_all": 767,
  "num_pairs_train": 309,
  "num_pairs_val": 309,
  "num_pairs_test": 149,
  "sequence_summaries": [
    {
      "seq_id": "bridge_night",
      "split": "train",
      "num_images": 80,
      "num_pose_rows": 80,
      "pairs_built": 309
    },
    {
      "seq_id": "field",
      "split": "val",
      "num_images": 178,
      "num_pose_rows": 80,
      "pairs_built": 309
    },
    {
      "seq_id": "wingsuit",
      "split": "test",
      "num_images": 47,
      "num_pose_rows": 40,
      "pairs_built": 149
    }
  ],
  "output_convention": {
    "R_BA": true,
    "t_BA_B": true,
    "tdir_frame": "B"
  },
  "insufficient_sequences": false
}

## 5. 360DVO motion distribution
{
  "dataset": "360DVO",
  "summary": {
    "dataset": "360DVO",
    "local_root": "data/360DVO",
    "selected_sequences": [
      "bridge_night",
      "field",
      "wingsuit"
    ],
    "num_sequences": 3,
    "split": {
      "train": [
        "bridge_night"
      ],
      "val": [
        "field"
      ],
      "test": [
        "wingsuit"
      ]
    },
    "split_by_sequence": true,
    "forbid_random_pair_split": true,
    "num_pairs_all": 767,
    "num_pairs_train": 309,
    "num_pairs_val": 309,
    "num_pairs_test": 149,
    "sequence_summaries": [
      {
        "seq_id": "bridge_night",
        "split": "train",
        "num_images": 80,
        "num_pose_rows": 80,
        "pairs_built": 309
      },
      {
        "seq_id": "field",
        "split": "val",
        "num_images": 178,
        "num_pose_rows": 80,
        "pairs_built": 309
      },
      {
        "seq_id": "wingsuit",
        "split": "test",
        "num_images": 47,
        "num_pose_rows": 40,
        "pairs_built": 149
      }
    ],
    "output_convention": {
      "R_BA": true,
      "t_BA_B": true,
      "tdir_frame": "B"
    },
    "insufficient_sequences": false
  },
  "split_stats": {
    "all": {
      "num_pairs": 767,
      "num_adjacent_pairs": 197,
      "num_kstep_pairs": 570,
      "gt_step_median": 0.5875152951884429,
      "gt_step_mean": 0.7183185870508929,
      "gt_step_p10": 0.15462984071308622,
      "gt_step_p25": 0.31124378849372525,
      "gt_step_p50": 0.5875152951884429,
      "gt_step_p75": 1.025730370204235,
      "gt_step_p90": 1.5767040274328026,
      "gt_step_p95": 1.8682793054867302,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 550.9503562680349,
      "rotation_step_mean": 1.4646962522980946,
      "rotation_step_p90": 3.326797734426328,
      "tmag_distribution_by_k": {
        "1": {
          "count": 197,
          "median": 0.27294072965292315,
          "p90": 0.3805517466733489
        },
        "2": {
          "count": 194,
          "median": 0.5450461361397051,
          "p90": 0.7587443990446582
        },
        "3": {
          "count": 191,
          "median": 0.8057910534617815,
          "p90": 1.1372208090612763
        },
        "5": {
          "count": 185,
          "median": 1.3383839974168457,
          "p90": 1.8887779917204544
        }
      }
    },
    "train": {
      "num_pairs": 309,
      "num_adjacent_pairs": 79,
      "num_kstep_pairs": 230,
      "gt_step_median": 0.44847684281290956,
      "gt_step_mean": 0.5045379315329813,
      "gt_step_p10": 0.15514421633035247,
      "gt_step_p25": 0.24393938872580345,
      "gt_step_p50": 0.44847684281290956,
      "gt_step_p75": 0.698223337705644,
      "gt_step_p90": 1.0593457396768438,
      "gt_step_p95": 1.1992362825660097,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 155.90222084369123,
      "rotation_step_mean": 1.8096238301637158,
      "rotation_step_p90": 3.338927861884563,
      "tmag_distribution_by_k": {
        "1": {
          "count": 79,
          "median": 0.18385370724778172,
          "p90": 0.26319480277288515
        },
        "2": {
          "count": 78,
          "median": 0.3670095400900337,
          "p90": 0.5251402778216229
        },
        "3": {
          "count": 77,
          "median": 0.546228721342646,
          "p90": 0.7862278052170971
        },
        "5": {
          "count": 75,
          "median": 0.9120412599418654,
          "p90": 1.3014492371735644
        }
      }
    },
    "val": {
      "num_pairs": 309,
      "num_adjacent_pairs": 79,
      "num_kstep_pairs": 230,
      "gt_step_median": 0.9132443618343159,
      "gt_step_mean": 0.9857362212703104,
      "gt_step_p10": 0.3470356421580293,
      "gt_step_p25": 0.60786589338786,
      "gt_step_p50": 0.9132443618343159,
      "gt_step_p75": 1.5211408160610185,
      "gt_step_p90": 1.8676750226842016,
      "gt_step_p95": 1.88126932575314,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 304.5924923725259,
      "rotation_step_mean": 1.4654995735162986,
      "rotation_step_p90": 3.196997418041944,
      "tmag_distribution_by_k": {
        "1": {
          "count": 79,
          "median": 0.3636184806660507,
          "p90": 0.38141965424381996
        },
        "2": {
          "count": 78,
          "median": 0.7288175082774854,
          "p90": 0.760394613637463
        },
        "3": {
          "count": 77,
          "median": 1.0925698077153978,
          "p90": 1.1396775826174022
        },
        "5": {
          "count": 75,
          "median": 1.8211759888154329,
          "p90": 1.8971975953217555
        }
      }
    },
    "test": {
      "num_pairs": 149,
      "num_adjacent_pairs": 39,
      "num_kstep_pairs": 110,
      "gt_step_median": 0.3878681347933658,
      "gt_step_mean": 0.6070848526967634,
      "gt_step_p10": 0.04248365533578011,
      "gt_step_p25": 0.11214171449584984,
      "gt_step_p50": 0.3878681347933658,
      "gt_step_p75": 0.9125458580736643,
      "gt_step_p90": 1.5321655857949266,
      "gt_step_p95": 1.9744281399708652,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 90.45564305181775,
      "rotation_step_mean": 0.7477106964799596,
      "rotation_step_p90": 1.2903874400451674,
      "tmag_distribution_by_k": {
        "1": {
          "count": 39,
          "median": 0.18725483538620805,
          "p90": 0.510932722855676
        },
        "2": {
          "count": 38,
          "median": 0.3785863057775181,
          "p90": 1.0066855899409664
        },
        "3": {
          "count": 37,
          "median": 0.5695992496807137,
          "p90": 1.4867695188624164
        },
        "5": {
          "count": 35,
          "median": 0.9629821231688878,
          "p90": 2.393132482181502
        }
      }
    }
  },
  "sequence_stats": {
    "bridge_night": {
      "num_pairs": 309,
      "num_adjacent_pairs": 79,
      "num_kstep_pairs": 230,
      "gt_step_median": 0.44847684281290956,
      "gt_step_mean": 0.5045379315329813,
      "gt_step_p10": 0.15514421633035247,
      "gt_step_p25": 0.24393938872580345,
      "gt_step_p50": 0.44847684281290956,
      "gt_step_p75": 0.698223337705644,
      "gt_step_p90": 1.0593457396768438,
      "gt_step_p95": 1.1992362825660097,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 155.90222084369123,
      "rotation_step_mean": 1.8096238301637158,
      "rotation_step_p90": 3.338927861884563,
      "tmag_distribution_by_k": {
        "1": {
          "count": 79,
          "median": 0.18385370724778172,
          "p90": 0.26319480277288515
        },
        "2": {
          "count": 78,
          "median": 0.3670095400900337,
          "p90": 0.5251402778216229
        },
        "3": {
          "count": 77,
          "median": 0.546228721342646,
          "p90": 0.7862278052170971
        },
        "5": {
          "count": 75,
          "median": 0.9120412599418654,
          "p90": 1.3014492371735644
        }
      }
    },
    "field": {
      "num_pairs": 309,
      "num_adjacent_pairs": 79,
      "num_kstep_pairs": 230,
      "gt_step_median": 0.9132443618343159,
      "gt_step_mean": 0.9857362212703104,
      "gt_step_p10": 0.3470356421580293,
      "gt_step_p25": 0.60786589338786,
      "gt_step_p50": 0.9132443618343159,
      "gt_step_p75": 1.5211408160610185,
      "gt_step_p90": 1.8676750226842016,
      "gt_step_p95": 1.88126932575314,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 304.5924923725259,
      "rotation_step_mean": 1.4654995735162986,
      "rotation_step_p90": 3.196997418041944,
      "tmag_distribution_by_k": {
        "1": {
          "count": 79,
          "median": 0.3636184806660507,
          "p90": 0.38141965424381996
        },
        "2": {
          "count": 78,
          "median": 0.7288175082774854,
          "p90": 0.760394613637463
        },
        "3": {
          "count": 77,
          "median": 1.0925698077153978,
          "p90": 1.1396775826174022
        },
        "5": {
          "count": 75,
          "median": 1.8211759888154329,
          "p90": 1.8971975953217555
        }
      }
    },
    "wingsuit": {
      "num_pairs": 149,
      "num_adjacent_pairs": 39,
      "num_kstep_pairs": 110,
      "gt_step_median": 0.3878681347933658,
      "gt_step_mean": 0.6070848526967634,
      "gt_step_p10": 0.04248365533578011,
      "gt_step_p25": 0.11214171449584984,
      "gt_step_p50": 0.3878681347933658,
      "gt_step_p75": 0.9125458580736643,
      "gt_step_p90": 1.5321655857949266,
      "gt_step_p95": 1.9744281399708652,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 90.45564305181775,
      "rotation_step_mean": 0.7477106964799596,
      "rotation_step_p90": 1.2903874400451674,
      "tmag_distribution_by_k": {
        "1": {
          "count": 39,
          "median": 0.18725483538620805,
          "p90": 0.510932722855676
        },
        "2": {
          "count": 38,
          "median": 0.3785863057775181,
          "p90": 1.0066855899409664
        },
        "3": {
          "count": 37,
          "median": 0.5695992496807137,
          "p90": 1.4867695188624164
        },
        "5": {
          "count": 35,
          "median": 0.9629821231688878,
          "p90": 2.393132482181502
        }
      }
    }
  },
  "compared_to_data2_seq03": {
    "gt_step_median_ratio": 78.2727794954832,
    "small_motion_fraction_delta": -0.38852097130242824,
    "path_length_ratio": 21.32685778997112
  },
  "answers": {
    "reduces_small_motion_risk": true,
    "suitable_as_main_train_eval_dataset": true,
    "multi_sequence_split_feasible": true,
    "enter_train360": true
  }
}

## 6. 360DVO vs legacy 对比
{
  "data2_train_eval_shift": {
    "gt_step_median_ratio_eval_over_train": 0.6303086823480436,
    "small_motion_fraction_delta": 0.12770672193856053,
    "path_length_ratio_eval_over_train": 0.49415517458410535,
    "distribution_shift_suspected": true
  },
  "legacy_quality_risk": "DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED",
  "360dvo_reduces_small_motion_risk": true
}

## 7. 新主数据集建议
{
  "use_360dvo_as_main_dataset": true,
  "downgrade_legacy_dataset_to_diagnostic": true,
  "do_train360_baseline": true,
  "do_gen5_t57b_external_eval": true,
  "keep_s5e15_as_legacy_best_candidate": true
}

## 8. 旧数据集如何保留为 diagnostic
legacy 数据更适合作为 diagnostic / low-quality stress test，而不是继续承担主训练/主评估职责。

## 9. 下一步 TRAIN360 计划
- do_train360_baseline = True

## 10. caveats
- 本轮不训练、不 fine-tune、不生成新 model candidate。
- 360DVO 原始数据保持 local-only，不纳入 Git。
