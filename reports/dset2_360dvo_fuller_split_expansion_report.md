# DSET2 360DVO fuller split expansion report

## 1. 为什么做 DSET2
DSET1 已经证明 360DVO 可以替代 legacy 数据成为主数据集候选，而 GEN5 说明 true image-pair 模型在外部数据上 translation generalization 仍弱，因此需要扩展数据子集，为 TRAIN360 新基线训练准备更完整的 sequence split。

## 2. 当前 DSET1 / GEN5 背景
{
  "dset1_final_classification": "DSET1_360DVO_MULTI_SEQUENCE_READY",
  "data3_final_classification": "DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED",
  "gen5_final_classification": "GEN5_360DVO_T57B_EXTERNAL_WEAK_GENERALIZATION"
}

## 3. 下载/挂载的新序列
{
  "dataset_repo_reachable": true,
  "local_root": "data/360DVO",
  "existing_sequences": [
    "bridge_night",
    "field",
    "wingsuit"
  ],
  "new_sequences_downloaded": [
    "canyon_line",
    "city_driving",
    "dragon_boat",
    "hongkong_central",
    "mountains"
  ],
  "selected_sequences_total": [
    "bridge_night",
    "canyon_line",
    "city_driving",
    "dragon_boat",
    "field",
    "hongkong_central",
    "mountains",
    "wingsuit"
  ],
  "num_sequences_total": 8,
  "download_attempted": true,
  "download_complete": true,
  "total_bytes_downloaded": 319279876,
  "max_download_bytes_respected": true,
  "access_blocked": false,
  "blockers": []
}

## 4. train/val/test sequence split
{
  "train_sequences": [
    "bridge_night",
    "canyon_line",
    "city_driving",
    "dragon_boat",
    "field",
    "hongkong_central"
  ],
  "val_sequences": [
    "mountains"
  ],
  "test_sequences": [
    "wingsuit"
  ],
  "sequences_all": [
    "bridge_night",
    "canyon_line",
    "city_driving",
    "dragon_boat",
    "field",
    "hongkong_central",
    "mountains",
    "wingsuit"
  ]
}

## 5. pair manifest 统计
{
  "num_sequences": 8,
  "sequences_all": [
    "bridge_night",
    "canyon_line",
    "city_driving",
    "dragon_boat",
    "field",
    "hongkong_central",
    "mountains",
    "wingsuit"
  ],
  "train_sequences": [
    "bridge_night",
    "canyon_line",
    "city_driving",
    "dragon_boat",
    "field",
    "hongkong_central"
  ],
  "val_sequences": [
    "mountains"
  ],
  "test_sequences": [
    "wingsuit"
  ],
  "num_pairs_all": 1473,
  "num_pairs_train": 1187,
  "num_pairs_val": 137,
  "num_pairs_test": 149,
  "num_adjacent_pairs": 382,
  "num_kstep_pairs": 1091,
  "split_by_sequence": true,
  "forbid_random_pair_split": true,
  "pose_convention": "R_BA_t_BA_B_tdir_B",
  "manifest_ready": true
}

## 6. motion distribution
{
  "manifest_summary": {
    "num_sequences": 8,
    "sequences_all": [
      "bridge_night",
      "canyon_line",
      "city_driving",
      "dragon_boat",
      "field",
      "hongkong_central",
      "mountains",
      "wingsuit"
    ],
    "train_sequences": [
      "bridge_night",
      "canyon_line",
      "city_driving",
      "dragon_boat",
      "field",
      "hongkong_central"
    ],
    "val_sequences": [
      "mountains"
    ],
    "test_sequences": [
      "wingsuit"
    ],
    "num_pairs_all": 1473,
    "num_pairs_train": 1187,
    "num_pairs_val": 137,
    "num_pairs_test": 149,
    "num_adjacent_pairs": 382,
    "num_kstep_pairs": 1091,
    "split_by_sequence": true,
    "forbid_random_pair_split": true,
    "pose_convention": "R_BA_t_BA_B_tdir_B",
    "manifest_ready": true
  },
  "split_stats": {
    "all": {
      "num_pairs": 1473,
      "num_adjacent_pairs": 382,
      "num_kstep_pairs": 1091,
      "gt_step_mean": 0.7732190113734588,
      "gt_step_median": 0.5378437102416956,
      "gt_step_p10": 0.1542392277056355,
      "gt_step_p25": 0.2799318462959662,
      "gt_step_p50": 0.5378437102416956,
      "gt_step_p75": 0.9928794029764628,
      "gt_step_p90": 1.7130872105688324,
      "gt_step_p95": 2.3303738699471843,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 1138.9516037531048,
      "rotation_step_mean": 1.3430684615103412,
      "rotation_step_p90": 3.3111189518899002,
      "tmag_distribution_by_k": {
        "1": {
          "count": 382,
          "median": 0.2188957654901838,
          "p90": 0.6535299252294351
        },
        "2": {
          "count": 374,
          "median": 0.4366874293095131,
          "p90": 1.3069498969828763
        },
        "3": {
          "count": 366,
          "median": 0.657883679195036,
          "p90": 1.9452665036787535
        },
        "5": {
          "count": 351,
          "median": 1.1031464739771926,
          "p90": 3.2739268633903396
        }
      }
    },
    "train": {
      "num_pairs": 1187,
      "num_adjacent_pairs": 307,
      "num_kstep_pairs": 880,
      "gt_step_mean": 0.8489729881378675,
      "gt_step_median": 0.6237927226652438,
      "gt_step_p10": 0.20514373486516918,
      "gt_step_p25": 0.32424967093946877,
      "gt_step_p50": 0.6237927226652438,
      "gt_step_p75": 1.0826914226441495,
      "gt_step_p90": 1.8580715041499487,
      "gt_step_p95": 2.464034492495168,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 1007.7309369196487,
      "rotation_step_mean": 1.4446680953167865,
      "rotation_step_p90": 3.340903789471745,
      "tmag_distribution_by_k": {
        "1": {
          "count": 307,
          "median": 0.23105739120074004,
          "p90": 0.7461693435759572
        },
        "2": {
          "count": 301,
          "median": 0.4646701485680924,
          "p90": 1.4455569435088953
        },
        "3": {
          "count": 295,
          "median": 0.6963957120163524,
          "p90": 2.196895709013572
        },
        "5": {
          "count": 284,
          "median": 1.1554256033344035,
          "p90": 3.655545651696654
        }
      }
    },
    "val": {
      "num_pairs": 137,
      "num_adjacent_pairs": 36,
      "num_kstep_pairs": 101,
      "gt_step_mean": 0.297554918114149,
      "gt_step_median": 0.25711727939278106,
      "gt_step_p10": 0.08570620544159678,
      "gt_step_p25": 0.17022156021721221,
      "gt_step_p50": 0.25711727939278106,
      "gt_step_p75": 0.41563105815296714,
      "gt_step_p90": 0.5714509688376576,
      "gt_step_p95": 0.6322688709002288,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 40.76502378163842,
      "rotation_step_mean": 1.110291393344476,
      "rotation_step_p90": 2.3413963872081838,
      "tmag_distribution_by_k": {
        "1": {
          "count": 36,
          "median": 0.10255550505414,
          "p90": 0.1722603414399725
        },
        "2": {
          "count": 35,
          "median": 0.2225387147666876,
          "p90": 0.26908463643650254
        },
        "3": {
          "count": 34,
          "median": 0.32786596888566255,
          "p90": 0.4278089324159565
        },
        "5": {
          "count": 32,
          "median": 0.5540198383596407,
          "p90": 0.6788773208785392
        }
      }
    },
    "test": {
      "num_pairs": 149,
      "num_adjacent_pairs": 39,
      "num_kstep_pairs": 110,
      "gt_step_mean": 0.6070848526967634,
      "gt_step_median": 0.3878681347933658,
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
      "gt_step_mean": 0.5045379315329813,
      "gt_step_median": 0.44847684281290956,
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
    "canyon_line": {
      "num_pairs": 6,
      "num_adjacent_pairs": 3,
      "num_kstep_pairs": 3,
      "gt_step_mean": 0.30268023220621293,
      "gt_step_median": 0.2690428806960401,
      "gt_step_p10": 0.16583598229869717,
      "gt_step_p25": 0.20335276551511217,
      "gt_step_p50": 0.2690428806960401,
      "gt_step_p75": 0.3892018894545666,
      "gt_step_p90": 0.4731618336239014,
      "gt_step_p95": 0.5055027719327985,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 1.8160813932372775,
      "rotation_step_mean": 0.07299141483597053,
      "rotation_step_p90": 0.11034586341101821,
      "tmag_distribution_by_k": {
        "1": {
          "count": 3,
          "median": 0.20223099582277104,
          "p90": 0.20582065883826267
        },
        "2": {
          "count": 2,
          "median": 0.3699238219030259,
          "p90": 0.4007687299854909
        },
        "3": {
          "count": 1,
          "median": 0.5378437102416956,
          "p90": 0.5378437102416956
        }
      }
    },
    "city_driving": {
      "num_pairs": 177,
      "num_adjacent_pairs": 46,
      "num_kstep_pairs": 131,
      "gt_step_mean": 2.05652638790393,
      "gt_step_median": 1.7133014424579613,
      "gt_step_p10": 0.7503765557076474,
      "gt_step_p25": 0.8587746469211968,
      "gt_step_p50": 1.7133014424579613,
      "gt_step_p75": 2.5527147684991034,
      "gt_step_p90": 4.015792274399551,
      "gt_step_p95": 4.164572300092224,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 364.0051706589956,
      "rotation_step_mean": 0.1525247550012845,
      "rotation_step_p90": 0.28273415343333635,
      "tmag_distribution_by_k": {
        "1": {
          "count": 46,
          "median": 0.7906049948561216,
          "p90": 0.8514982501504971
        },
        "2": {
          "count": 45,
          "median": 1.5744053826211717,
          "p90": 1.6899582656376988
        },
        "3": {
          "count": 44,
          "median": 2.3543073229784115,
          "p90": 2.5282513075727784
        },
        "5": {
          "count": 42,
          "median": 3.9639733922767935,
          "p90": 4.207617783939172
        }
      }
    },
    "dragon_boat": {
      "num_pairs": 189,
      "num_adjacent_pairs": 49,
      "num_kstep_pairs": 140,
      "gt_step_mean": 0.5739154479187679,
      "gt_step_median": 0.4722315716809981,
      "gt_step_p10": 0.20735138209955797,
      "gt_step_p25": 0.24203810366411807,
      "gt_step_p50": 0.4722315716809981,
      "gt_step_p75": 0.7028459900572073,
      "gt_step_p90": 1.05596269645669,
      "gt_step_p95": 1.1202400669588288,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 108.47001965664714,
      "rotation_step_mean": 3.554596298420539,
      "rotation_step_p90": 10.63047784470458,
      "tmag_distribution_by_k": {
        "1": {
          "count": 49,
          "median": 0.21002831603004327,
          "p90": 0.2319407357612144
        },
        "2": {
          "count": 48,
          "median": 0.4169408332089275,
          "p90": 0.4652306286321576
        },
        "3": {
          "count": 47,
          "median": 0.6255113116079676,
          "p90": 0.6951724672604291
        },
        "5": {
          "count": 45,
          "median": 1.0409779204788745,
          "p90": 1.1503674282990592
        }
      }
    },
    "field": {
      "num_pairs": 309,
      "num_adjacent_pairs": 79,
      "num_kstep_pairs": 230,
      "gt_step_mean": 0.9857362212703104,
      "gt_step_median": 0.9132443618343159,
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
    "hongkong_central": {
      "num_pairs": 197,
      "num_adjacent_pairs": 51,
      "num_kstep_pairs": 146,
      "gt_step_mean": 0.37027894413478013,
      "gt_step_median": 0.31716016812779196,
      "gt_step_p10": 0.1309871005952468,
      "gt_step_p25": 0.15956994639990355,
      "gt_step_p50": 0.31716016812779196,
      "gt_step_p75": 0.47647645295201596,
      "gt_step_p90": 0.7021135324419696,
      "gt_step_p95": 0.758553629506191,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 72.94495199455169,
      "rotation_step_mean": 0.01804247146282064,
      "rotation_step_p90": 0.031244323275426367,
      "tmag_distribution_by_k": {
        "1": {
          "count": 51,
          "median": 0.13682290654738824,
          "p90": 0.1566109357008045
        },
        "2": {
          "count": 50,
          "median": 0.27365345066525415,
          "p90": 0.31347220304290263
        },
        "3": {
          "count": 49,
          "median": 0.41048770456762446,
          "p90": 0.46843243702464904
        },
        "5": {
          "count": 47,
          "median": 0.6844480772020731,
          "p90": 0.7766776516720589
        }
      }
    },
    "mountains": {
      "num_pairs": 137,
      "num_adjacent_pairs": 36,
      "num_kstep_pairs": 101,
      "gt_step_mean": 0.297554918114149,
      "gt_step_median": 0.25711727939278106,
      "gt_step_p10": 0.08570620544159678,
      "gt_step_p25": 0.17022156021721221,
      "gt_step_p50": 0.25711727939278106,
      "gt_step_p75": 0.41563105815296714,
      "gt_step_p90": 0.5714509688376576,
      "gt_step_p95": 0.6322688709002288,
      "small_motion_fraction": 0.0,
      "very_small_motion_fraction": 0.0,
      "path_length": 40.76502378163842,
      "rotation_step_mean": 1.110291393344476,
      "rotation_step_p90": 2.3413963872081838,
      "tmag_distribution_by_k": {
        "1": {
          "count": 36,
          "median": 0.10255550505414,
          "p90": 0.1722603414399725
        },
        "2": {
          "count": 35,
          "median": 0.2225387147666876,
          "p90": 0.26908463643650254
        },
        "3": {
          "count": 34,
          "median": 0.32786596888566255,
          "p90": 0.4278089324159565
        },
        "5": {
          "count": 32,
          "median": 0.5540198383596407,
          "p90": 0.6788773208785392
        }
      }
    },
    "wingsuit": {
      "num_pairs": 149,
      "num_adjacent_pairs": 39,
      "num_kstep_pairs": 110,
      "gt_step_mean": 0.6070848526967634,
      "gt_step_median": 0.3878681347933658,
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
  "comparison": {
    "vs_dset1": {
      "available": true,
      "median_ratio": 0.915454822447107,
      "small_motion_fraction_delta": 0.0,
      "rotation_mean_ratio": 0.9169604000850547
    },
    "vs_data2_seq03": {
      "gt_step_median_ratio": 71.65519345547912,
      "small_motion_fraction_delta": -0.38852097130242824,
      "path_length_ratio": 44.08792662815686
    },
    "train_val_similarity": {
      "available": true,
      "median_ratio": 2.426101910141624,
      "small_motion_fraction_delta": 0.0,
      "rotation_mean_ratio": 1.301161212251753
    },
    "train_test_similarity": {
      "available": true,
      "median_ratio": 1.608259783953547,
      "small_motion_fraction_delta": 0.0,
      "rotation_mean_ratio": 1.932121744570371
    },
    "val_test_similarity": {
      "available": true,
      "median_ratio": 0.662898692437724,
      "small_motion_fraction_delta": 0.0,
      "rotation_mean_ratio": 1.4849211046083175
    }
  },
  "answers": {
    "continues_to_avoid_small_motion_risk": true,
    "train_val_test_motion_relatively_balanced": true,
    "meets_train360_readiness_threshold": false,
    "need_more_sequences": true
  }
}

## 7. 与 DSET1 和 legacy DATA2 的对比
{
  "vs_dset1": {
    "available": true,
    "median_ratio": 0.915454822447107,
    "small_motion_fraction_delta": 0.0,
    "rotation_mean_ratio": 0.9169604000850547
  },
  "vs_data2_seq03": {
    "gt_step_median_ratio": 71.65519345547912,
    "small_motion_fraction_delta": -0.38852097130242824,
    "path_length_ratio": 44.08792662815686
  },
  "train_val_similarity": {
    "available": true,
    "median_ratio": 2.426101910141624,
    "small_motion_fraction_delta": 0.0,
    "rotation_mean_ratio": 1.301161212251753
  },
  "train_test_similarity": {
    "available": true,
    "median_ratio": 1.608259783953547,
    "small_motion_fraction_delta": 0.0,
    "rotation_mean_ratio": 1.932121744570371
  },
  "val_test_similarity": {
    "available": true,
    "median_ratio": 0.662898692437724,
    "small_motion_fraction_delta": 0.0,
    "rotation_mean_ratio": 1.4849211046083175
  }
}

## 8. 是否 ready for TRAIN360
{
  "num_sequences_ok": true,
  "train_pairs_ok": false,
  "val_pairs_ok": false,
  "test_pairs_ok": false,
  "sequence_split_ok": true,
  "small_motion_risk_low": true,
  "train360_ready": false
}

## 9. caveats
- 本轮不训练、不 fine-tune、不使用 360DVO GT calibration。
- manifest jsonl 只包含 metadata/相对路径，不包含原始图像数据。
- data/360DVO 保持 local-only，不纳入 Git。
