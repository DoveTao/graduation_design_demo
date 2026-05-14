# DATA2 tdir / scale / observability / split shift 审计

## 1. 为什么做 DATA2
STRUCT1B 已证明 hard scale guard 能在 train-side 压住一部分 scale 爆炸，但 heldout seq03 仍明显失败，因此需要判断问题更像数据可观测性限制，还是模型本身仍有空间。

## 2. train/eval GT motion distribution
{
  "scene01_seq01": {
    "num_poses": 324,
    "num_adjacent_edges": 323,
    "gt_step_median": 0.018819936237936628,
    "gt_step_mean": 0.08032613773047229,
    "gt_step_p10": 0.0045279134051935685,
    "gt_step_p25": 0.007189890133912266,
    "gt_step_p50": 0.018819936237936628,
    "gt_step_p75": 0.07728853792715779,
    "gt_step_p90": 0.29077897968656197,
    "gt_step_p95": 0.3998299650325727,
    "very_small_motion_fraction": 0.015479876160990712,
    "small_motion_fraction": 0.11764705882352941,
    "medium_motion_fraction": 0.5201238390092879,
    "rotation_step_mean": 0.3615750741538758,
    "rotation_step_p90": 0.7133580087209269,
    "translation_rotation_ratio": 0.2221561813088029,
    "path_length": 25.94534248694255
  },
  "scene01_seq02": {
    "num_poses": 464,
    "num_adjacent_edges": 463,
    "gt_step_median": 0.008273451516748423,
    "gt_step_mean": 0.05687484393503026,
    "gt_step_p10": 0.0022565458592545165,
    "gt_step_p25": 0.003548148333901182,
    "gt_step_p50": 0.008273451516748423,
    "gt_step_p75": 0.029313208014529878,
    "gt_step_p90": 0.2329761444222035,
    "gt_step_p95": 0.32439058289649064,
    "very_small_motion_fraction": 0.07127429805615551,
    "small_motion_fraction": 0.36069114470842334,
    "medium_motion_fraction": 0.7019438444924406,
    "rotation_step_mean": 0.43454550933998387,
    "rotation_step_p90": 0.7177848167709806,
    "translation_rotation_ratio": 0.13088351556414768,
    "path_length": 26.333052741919012
  },
  "scene01_seq03": {
    "num_poses": 454,
    "num_adjacent_edges": 453,
    "gt_step_median": 0.00750599760191659,
    "gt_step_mean": 0.05702790181301312,
    "gt_step_p10": 0.0021940821954738654,
    "gt_step_p25": 0.0035185224171510503,
    "gt_step_p50": 0.00750599760191659,
    "gt_step_p75": 0.026876941790315374,
    "gt_step_p90": 0.2125134648635189,
    "gt_step_p95": 0.33083886993157713,
    "very_small_motion_fraction": 0.0684326710816777,
    "small_motion_fraction": 0.38852097130242824,
    "medium_motion_fraction": 0.7086092715231788,
    "rotation_step_mean": 0.46411830329116865,
    "rotation_step_p90": 0.7321424095755357,
    "translation_rotation_ratio": 0.12287363245236241,
    "path_length": 25.833639521294945
  },
  "train_vs_eval_shift": {
    "gt_step_median_ratio_eval_over_train": 0.6303086823480436,
    "small_motion_fraction_delta": 0.12770672193856053,
    "path_length_ratio_eval_over_train": 0.49415517458410535,
    "distribution_shift_suspected": true
  }
}

## 3. tdir label stability by gt step
{
  "bins": {
    "lt_0p002": {
      "s5e15": {
        "count": 31,
        "tdir_mean_deg": 55.69216989525706,
        "anti_parallel_rate": 0.03225806451612903,
        "tmag_median_ratio": 5.219757497942087,
        "path_contribution_fraction": 0.06484816962404416,
        "pred_step_mean": 0.008493417930982237,
        "gt_step_mean": 0.0015253877241384027
      },
      "struct1b": {
        "count": 31,
        "tdir_mean_deg": 87.640138531096,
        "anti_parallel_rate": 0.5161290322580645,
        "tmag_median_ratio": 32.076542818848516,
        "path_contribution_fraction": 0.014869219313465932,
        "pred_step_mean": 0.07919224039752275,
        "gt_step_mean": 0.0015253877241384027
      }
    },
    "0p002_0p005": {
      "s5e15": {
        "count": 145,
        "tdir_mean_deg": 51.32320784839749,
        "anti_parallel_rate": 0.09655172413793103,
        "tmag_median_ratio": 2.643640499657814,
        "path_contribution_fraction": 0.30605896086806883,
        "pred_step_mean": 0.008570054096451016,
        "gt_step_mean": 0.003332510350907565
      },
      "struct1b": {
        "count": 145,
        "tdir_mean_deg": 83.10905021136648,
        "anti_parallel_rate": 0.3931034482758621,
        "tmag_median_ratio": 17.832179979122973,
        "path_contribution_fraction": 0.4963452264954779,
        "pred_step_mean": 0.5651607640207285,
        "gt_step_mean": 0.003332510350907565
      }
    },
    "0p005_0p01": {
      "s5e15": {
        "count": 86,
        "tdir_mean_deg": 29.317974797668942,
        "anti_parallel_rate": 0.03488372093023256,
        "tmag_median_ratio": 1.3511250335595095,
        "path_contribution_fraction": 0.1940825390058193,
        "pred_step_mean": 0.009162932345109535,
        "gt_step_mean": 0.007105036914259122
      },
      "struct1b": {
        "count": 86,
        "tdir_mean_deg": 60.96503713620608,
        "anti_parallel_rate": 0.05813953488372093,
        "tmag_median_ratio": 15.467808310232037,
        "path_contribution_fraction": 0.300322062268931,
        "pred_step_mean": 0.5765605755563471,
        "gt_step_mean": 0.007105036914259122
      }
    },
    "0p01_0p02": {
      "s5e15": {
        "count": 59,
        "tdir_mean_deg": 33.312845649448874,
        "anti_parallel_rate": 0.06779661016949153,
        "tmag_median_ratio": 0.6502568828902572,
        "path_contribution_fraction": 0.13312744329649243,
        "pred_step_mean": 0.009161404229848584,
        "gt_step_mean": 0.014410541418594925
      },
      "struct1b": {
        "count": 59,
        "tdir_mean_deg": 76.23047207490487,
        "anti_parallel_rate": 0.3050847457627119,
        "tmag_median_ratio": 8.895209372252516,
        "path_contribution_fraction": 0.06653873045092777,
        "pred_step_mean": 0.1861995619801461,
        "gt_step_mean": 0.014410541418594925
      }
    },
    "gte_0p02": {
      "s5e15": {
        "count": 132,
        "tdir_mean_deg": 69.35455763388212,
        "anti_parallel_rate": 0.3409090909090909,
        "tmag_median_ratio": 0.05885911568653142,
        "path_contribution_fraction": 0.3018828872055753,
        "pred_step_mean": 0.009285622630372732,
        "gt_step_mean": 0.18062033501367789
      },
      "struct1b": {
        "count": 132,
        "tdir_mean_deg": 87.88888492606732,
        "anti_parallel_rate": 0.49242424242424243,
        "tmag_median_ratio": 0.7612664226900652,
        "path_contribution_fraction": 0.12192476147119737,
        "pred_step_mean": 0.1525015085393383,
        "gt_step_mean": 0.18062033501367789
      }
    }
  },
  "low_motion_tdir_unstable": false,
  "tdir_label_noise_suspected": false
}

## 4. S5E15 vs STRUCT1B scale/path error
{
  "s5e15": {
    "overall": {
      "rot_mean_deg": 0.9134395040767528,
      "rot_median_deg": 0.797895569319923,
      "rot_p90_deg": 1.3218302317546538,
      "tdir_mean_deg": 50.353041365033235,
      "tdir_median_deg": 41.2294804093569,
      "tdir_p90_deg": 101.804771450087,
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
    },
    "by_gt_step_bin": {
      "lt_0p002": {
        "count": 31,
        "tdir_mean_deg": 55.69216989525706,
        "anti_parallel_rate": 0.03225806451612903,
        "tmag_median_ratio": 5.219757497942087,
        "path_contribution_fraction": 0.06484816962404416,
        "pred_step_mean": 0.008493417930982237,
        "gt_step_mean": 0.0015253877241384027
      },
      "0p002_0p005": {
        "count": 145,
        "tdir_mean_deg": 51.32320784839749,
        "anti_parallel_rate": 0.09655172413793103,
        "tmag_median_ratio": 2.643640499657814,
        "path_contribution_fraction": 0.30605896086806883,
        "pred_step_mean": 0.008570054096451016,
        "gt_step_mean": 0.003332510350907565
      },
      "0p005_0p01": {
        "count": 86,
        "tdir_mean_deg": 29.317974797668942,
        "anti_parallel_rate": 0.03488372093023256,
        "tmag_median_ratio": 1.3511250335595095,
        "path_contribution_fraction": 0.1940825390058193,
        "pred_step_mean": 0.009162932345109535,
        "gt_step_mean": 0.007105036914259122
      },
      "0p01_0p02": {
        "count": 59,
        "tdir_mean_deg": 33.312845649448874,
        "anti_parallel_rate": 0.06779661016949153,
        "tmag_median_ratio": 0.6502568828902572,
        "path_contribution_fraction": 0.13312744329649243,
        "pred_step_mean": 0.009161404229848584,
        "gt_step_mean": 0.014410541418594925
      },
      "gte_0p02": {
        "count": 132,
        "tdir_mean_deg": 69.35455763388212,
        "anti_parallel_rate": 0.3409090909090909,
        "tmag_median_ratio": 0.05885911568653142,
        "path_contribution_fraction": 0.3018828872055753,
        "pred_step_mean": 0.009285622630372732,
        "gt_step_mean": 0.18062033501367789
      }
    },
    "top20_path_contributing_edges": [
      {
        "edge_index": 443,
        "pred_step_length": 0.010786400279570861,
        "gt_step_length": 0.021161772832167046,
        "tmag_ratio": 0.5097115617447204,
        "tdir_deg": 81.79615703024666,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 441,
        "pred_step_length": 0.010784067698207174,
        "gt_step_length": 0.01440829257065828,
        "tmag_ratio": 0.748462570795401,
        "tdir_deg": 81.38604172612891,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 440,
        "pred_step_length": 0.010782893884162254,
        "gt_step_length": 0.01983654510749983,
        "tmag_ratio": 0.5435872943462036,
        "tdir_deg": 72.523194916301,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 438,
        "pred_step_length": 0.010780560936563632,
        "gt_step_length": 0.008547514141552307,
        "tmag_ratio": 1.2612510208267151,
        "tdir_deg": 83.19262180356293,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 436,
        "pred_step_length": 0.010778217090542545,
        "gt_step_length": 0.008511687565275946,
        "tmag_ratio": 1.2662843893040756,
        "tdir_deg": 84.14092551713028,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 432,
        "pred_step_length": 0.010773537027595469,
        "gt_step_length": 0.0052392622763528995,
        "tmag_ratio": 2.0563080180622357,
        "tdir_deg": 91.8331588921724,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 430,
        "pred_step_length": 0.010771204659881004,
        "gt_step_length": 0.018426411674708272,
        "tmag_ratio": 0.5845524809730234,
        "tdir_deg": 96.17304003340102,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 428,
        "pred_step_length": 0.01076888338804187,
        "gt_step_length": 0.008191491585853155,
        "tmag_ratio": 1.314642550160207,
        "tdir_deg": 97.25345217295667,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 427,
        "pred_step_length": 0.010767704327356421,
        "gt_step_length": 0.0331549785645005,
        "tmag_ratio": 0.32476885202651146,
        "tdir_deg": 111.01802758606988,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 425,
        "pred_step_length": 0.010765372741116096,
        "gt_step_length": 0.04740041895595451,
        "tmag_ratio": 0.2271155609641238,
        "tdir_deg": 111.41956814636933,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 423,
        "pred_step_length": 0.01076305625450149,
        "gt_step_length": 0.04432089547155397,
        "tmag_ratio": 0.24284383562172007,
        "tdir_deg": 98.11159851022573,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 421,
        "pred_step_length": 0.010760730004110396,
        "gt_step_length": 0.07542928359861603,
        "tmag_ratio": 0.14265984629221418,
        "tdir_deg": 134.85030514958393,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 419,
        "pred_step_length": 0.01075840986744503,
        "gt_step_length": 0.0794944324088231,
        "tmag_ratio": 0.13533538817054253,
        "tdir_deg": 128.61868300883157,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 418,
        "pred_step_length": 0.010757240930019263,
        "gt_step_length": 0.06981180461211567,
        "tmag_ratio": 0.1540891399354024,
        "tdir_deg": 123.67476444750575,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 416,
        "pred_step_length": 0.010754937571732143,
        "gt_step_length": 0.09022922480660207,
        "tmag_ratio": 0.1191957217274597,
        "tdir_deg": 142.63298328172107,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 415,
        "pred_step_length": 0.010753773338162282,
        "gt_step_length": 0.12672160886526113,
        "tmag_ratio": 0.08486140157513634,
        "tdir_deg": 143.9758158282579,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 414,
        "pred_step_length": 0.010752576544493507,
        "gt_step_length": 0.08378961185254438,
        "tmag_ratio": 0.12832827729786161,
        "tdir_deg": 136.6850123007137,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 413,
        "pred_step_length": 0.010751441884708664,
        "gt_step_length": 0.11717073789474947,
        "tmag_ratio": 0.09175876228044516,
        "tdir_deg": 148.7084363766626,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 411,
        "pred_step_length": 0.010749113339335623,
        "gt_step_length": 0.1072246665837671,
        "tmag_ratio": 0.10024851260263044,
        "tdir_deg": 152.37571278013868,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 410,
        "pred_step_length": 0.010747935962757908,
        "gt_step_length": 0.18668414820439364,
        "tmag_ratio": 0.057572836612728294,
        "tdir_deg": 136.54363613795687,
        "anti_parallel_flag": true
      }
    ],
    "top20_tmag_ratio_edges": [
      {
        "edge_index": 289,
        "tmag_ratio": 19.819619392350763,
        "gt_step_length": 0.00042475181630888194,
        "tdir_deg": 55.42727484753299,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 337,
        "tmag_ratio": 10.972052983683184,
        "gt_step_length": 0.0007685294177460289,
        "tdir_deg": 67.13489291369787,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 313,
        "tmag_ratio": 10.14220081440763,
        "gt_step_length": 0.0008305148304635886,
        "tdir_deg": 27.914710154133804,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 346,
        "tmag_ratio": 7.939076548804769,
        "gt_step_length": 0.0010631146775815922,
        "tdir_deg": 104.83527860828148,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 286,
        "tmag_ratio": 6.943229390025936,
        "gt_step_length": 0.0012123797585922582,
        "tdir_deg": 60.26558827906898,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 322,
        "tmag_ratio": 6.621299742748497,
        "gt_step_length": 0.001272421145637757,
        "tdir_deg": 67.01494260051919,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 334,
        "tmag_ratio": 6.561896254346098,
        "gt_step_length": 0.0012846508582362206,
        "tdir_deg": 41.24623897054074,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 194,
        "tmag_ratio": 6.5387987940746894,
        "gt_step_length": 0.001625072697240159,
        "tdir_deg": 27.61759992390206,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 314,
        "tmag_ratio": 6.312519039121046,
        "gt_step_length": 0.0013344055102560964,
        "tdir_deg": 19.245666308840995,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 262,
        "tmag_ratio": 6.2860900694807516,
        "gt_step_length": 0.0013383505114954511,
        "tdir_deg": 80.20160850881773,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 291,
        "tmag_ratio": 6.25683976236417,
        "gt_step_length": 0.0013455424451967184,
        "tdir_deg": 54.18897195238659,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 339,
        "tmag_ratio": 5.713162853070296,
        "gt_step_length": 0.0014762549499631238,
        "tdir_deg": 55.79865239620348,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 285,
        "tmag_ratio": 5.598827763842057,
        "gt_step_length": 0.001503458352147214,
        "tdir_deg": 75.80114721431204,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 283,
        "tmag_ratio": 5.455690996704374,
        "gt_step_length": 0.001542836874527689,
        "tdir_deg": 74.14487603959226,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 264,
        "tmag_ratio": 5.454868011226163,
        "gt_step_length": 0.0015423642498529479,
        "tdir_deg": 10.416010105139748,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 287,
        "tmag_ratio": 5.219757497942087,
        "gt_step_length": 0.001612718596443666,
        "tdir_deg": 67.56216457813206,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 353,
        "tmag_ratio": 5.197568533498379,
        "gt_step_length": 0.001625012667400246,
        "tdir_deg": 68.33239073190789,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 344,
        "tmag_ratio": 5.193430274520374,
        "gt_step_length": 0.0016248288595645123,
        "tdir_deg": 49.44138394627353,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 263,
        "tmag_ratio": 5.1297099232164,
        "gt_step_length": 0.0016401035348222776,
        "tdir_deg": 81.61536345107079,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 267,
        "tmag_ratio": 5.02127520242653,
        "gt_step_length": 0.001675663832398578,
        "tdir_deg": 47.44612947877274,
        "anti_parallel_flag": false
      }
    ]
  },
  "struct1b": {
    "overall": {
      "rot_mean_deg": 0.9153562920646506,
      "signed_tdir_mean_deg": 79.71209808275077,
      "anti_parallel_rate": 0.3554083885209713,
      "tmag_median_ratio": 11.866910847001154,
      "tmag_p95_ratio": 94.01508919525963,
      "path_ratio": 6.39102545028478
    },
    "by_gt_step_bin": {
      "lt_0p002": {
        "count": 31,
        "tdir_mean_deg": 87.640138531096,
        "anti_parallel_rate": 0.5161290322580645,
        "tmag_median_ratio": 32.076542818848516,
        "path_contribution_fraction": 0.014869219313465932,
        "pred_step_mean": 0.07919224039752275,
        "gt_step_mean": 0.0015253877241384027
      },
      "0p002_0p005": {
        "count": 145,
        "tdir_mean_deg": 83.10905021136648,
        "anti_parallel_rate": 0.3931034482758621,
        "tmag_median_ratio": 17.832179979122973,
        "path_contribution_fraction": 0.4963452264954779,
        "pred_step_mean": 0.5651607640207285,
        "gt_step_mean": 0.003332510350907565
      },
      "0p005_0p01": {
        "count": 86,
        "tdir_mean_deg": 60.96503713620608,
        "anti_parallel_rate": 0.05813953488372093,
        "tmag_median_ratio": 15.467808310232037,
        "path_contribution_fraction": 0.300322062268931,
        "pred_step_mean": 0.5765605755563471,
        "gt_step_mean": 0.007105036914259122
      },
      "0p01_0p02": {
        "count": 59,
        "tdir_mean_deg": 76.23047207490487,
        "anti_parallel_rate": 0.3050847457627119,
        "tmag_median_ratio": 8.895209372252516,
        "path_contribution_fraction": 0.06653873045092777,
        "pred_step_mean": 0.1861995619801461,
        "gt_step_mean": 0.014410541418594925
      },
      "gte_0p02": {
        "count": 132,
        "tdir_mean_deg": 87.88888492606732,
        "anti_parallel_rate": 0.49242424242424243,
        "tmag_median_ratio": 0.7612664226900652,
        "path_contribution_fraction": 0.12192476147119737,
        "pred_step_mean": 0.1525015085393383,
        "gt_step_mean": 0.18062033501367789
      }
    },
    "top20_path_contributing_edges": [
      {
        "edge_index": 183,
        "pred_step_length": 23.666539237457414,
        "gt_step_length": 0.0036358983250567713,
        "tmag_ratio": 6509.131202696071,
        "tdir_deg": 50.00814201276764,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 184,
        "pred_step_length": 21.825970299763963,
        "gt_step_length": 0.0046069030977747396,
        "tmag_ratio": 4737.666461946314,
        "tdir_deg": 67.3072632435378,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 182,
        "pred_step_length": 16.29505390763636,
        "gt_step_length": 0.007141982570100127,
        "tmag_ratio": 2281.5869049940725,
        "tdir_deg": 60.79241748953661,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 185,
        "pred_step_length": 14.844458172981588,
        "gt_step_length": 0.00583025983196693,
        "tmag_ratio": 2546.105765576759,
        "tdir_deg": 64.14082767233522,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 186,
        "pred_step_length": 12.28561832970088,
        "gt_step_length": 0.004721720966512095,
        "tmag_ratio": 2601.936543229531,
        "tdir_deg": 58.88321391365677,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 187,
        "pred_step_length": 8.10637409223273,
        "gt_step_length": 0.0037271912575566335,
        "tmag_ratio": 2174.9283930084334,
        "tdir_deg": 65.92023087873898,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 180,
        "pred_step_length": 4.335193868350683,
        "gt_step_length": 0.004148355919362418,
        "tmag_ratio": 1045.0390353721098,
        "tdir_deg": 59.858593242706064,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 49,
        "pred_step_length": 2.951101099342888,
        "gt_step_length": 0.01618478905811875,
        "tmag_ratio": 182.3379401946874,
        "tdir_deg": 125.06535736197851,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 181,
        "pred_step_length": 2.554272902999844,
        "gt_step_length": 0.005073569632379076,
        "tmag_ratio": 503.4468999299228,
        "tdir_deg": 69.64240392154545,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 189,
        "pred_step_length": 2.053286728654286,
        "gt_step_length": 0.007176443841928079,
        "tmag_ratio": 286.1147908185448,
        "tdir_deg": 61.825567940593885,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 188,
        "pred_step_length": 1.8256197365137257,
        "gt_step_length": 0.0050802043290295975,
        "tmag_ratio": 359.3595096326468,
        "tdir_deg": 58.64145273404894,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 369,
        "pred_step_length": 1.77833578739167,
        "gt_step_length": 0.22464183412227567,
        "tmag_ratio": 7.916316185451447,
        "tdir_deg": 142.62229310642826,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 89,
        "pred_step_length": 1.6517074996750476,
        "gt_step_length": 0.013597616926037701,
        "tmag_ratio": 121.47036562798283,
        "tdir_deg": 57.44179060200817,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 190,
        "pred_step_length": 1.09906055756718,
        "gt_step_length": 0.007603654693443208,
        "tmag_ratio": 144.54372296981384,
        "tdir_deg": 61.61140758560795,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 7,
        "pred_step_length": 1.0465236797589264,
        "gt_step_length": 0.4384642701865327,
        "tmag_ratio": 2.3867935221123293,
        "tdir_deg": 57.86867788139028,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 191,
        "pred_step_length": 0.8957179701317283,
        "gt_step_length": 0.004375041031155719,
        "tmag_ratio": 204.7336159256805,
        "tdir_deg": 77.29789189046997,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 91,
        "pred_step_length": 0.7935694837729785,
        "gt_step_length": 0.0073252830827057625,
        "tmag_ratio": 108.3329442443684,
        "tdir_deg": 64.2324149063695,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 47,
        "pred_step_length": 0.785776153892735,
        "gt_step_length": 0.030595177618136248,
        "tmag_ratio": 25.68300676989506,
        "tdir_deg": 121.8692792822527,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 193,
        "pred_step_length": 0.6868441126144604,
        "gt_step_length": 0.008218915837366605,
        "tmag_ratio": 83.56870008228844,
        "tdir_deg": 56.09615588480501,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 4,
        "pred_step_length": 0.6335212546653889,
        "gt_step_length": 0.45345601774439104,
        "tmag_ratio": 1.3970952636524474,
        "tdir_deg": 118.04446161426446,
        "anti_parallel_flag": true
      }
    ],
    "top20_tmag_ratio_edges": [
      {
        "edge_index": 183,
        "tmag_ratio": 6509.131202696071,
        "gt_step_length": 0.0036358983250567713,
        "tdir_deg": 50.00814201276764,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 184,
        "tmag_ratio": 4737.666461946314,
        "gt_step_length": 0.0046069030977747396,
        "tdir_deg": 67.3072632435378,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 186,
        "tmag_ratio": 2601.936543229531,
        "gt_step_length": 0.004721720966512095,
        "tdir_deg": 58.88321391365677,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 185,
        "tmag_ratio": 2546.105765576759,
        "gt_step_length": 0.00583025983196693,
        "tdir_deg": 64.14082767233522,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 182,
        "tmag_ratio": 2281.5869049940725,
        "gt_step_length": 0.007141982570100127,
        "tdir_deg": 60.79241748953661,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 187,
        "tmag_ratio": 2174.9283930084334,
        "gt_step_length": 0.0037271912575566335,
        "tdir_deg": 65.92023087873898,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 180,
        "tmag_ratio": 1045.0390353721098,
        "gt_step_length": 0.004148355919362418,
        "tdir_deg": 59.858593242706064,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 181,
        "tmag_ratio": 503.4468999299228,
        "gt_step_length": 0.005073569632379076,
        "tdir_deg": 69.64240392154545,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 188,
        "tmag_ratio": 359.3595096326468,
        "gt_step_length": 0.0050802043290295975,
        "tdir_deg": 58.64145273404894,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 194,
        "tmag_ratio": 333.89706591198967,
        "gt_step_length": 0.001625072697240159,
        "tdir_deg": 76.20388982189283,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 189,
        "tmag_ratio": 286.1147908185448,
        "gt_step_length": 0.007176443841928079,
        "tdir_deg": 61.825567940593885,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 192,
        "tmag_ratio": 209.47843660100034,
        "gt_step_length": 0.0029632241606054772,
        "tdir_deg": 58.310512776494576,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 191,
        "tmag_ratio": 204.7336159256805,
        "gt_step_length": 0.004375041031155719,
        "tdir_deg": 77.29789189046997,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 49,
        "tmag_ratio": 182.3379401946874,
        "gt_step_length": 0.01618478905811875,
        "tdir_deg": 125.06535736197851,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 190,
        "tmag_ratio": 144.54372296981384,
        "gt_step_length": 0.007603654693443208,
        "tdir_deg": 61.61140758560795,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 197,
        "tmag_ratio": 140.69867889105095,
        "gt_step_length": 0.002123774290125119,
        "tdir_deg": 25.834479439556155,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 346,
        "tmag_ratio": 129.49053511446957,
        "gt_step_length": 0.0010631146775815922,
        "tdir_deg": 122.83654358767454,
        "anti_parallel_flag": true
      },
      {
        "edge_index": 89,
        "tmag_ratio": 121.47036562798283,
        "gt_step_length": 0.013597616926037701,
        "tdir_deg": 57.44179060200817,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 91,
        "tmag_ratio": 108.3329442443684,
        "gt_step_length": 0.0073252830827057625,
        "tdir_deg": 64.2324149063695,
        "anti_parallel_flag": false
      },
      {
        "edge_index": 289,
        "tmag_ratio": 106.58790085306187,
        "gt_step_length": 0.00042475181630888194,
        "tdir_deg": 84.10500877117067,
        "anti_parallel_flag": false
      }
    ],
    "top20_path_contribution_fraction": 0.7275107981090755
  },
  "struct1b_over_scale_small_motion_coupled": true,
  "path_error_outlier_dominated": false,
  "global_scale_bias_suspected": true
}

## 5. STRUCT1B train-side 与 heldout gap
{
  "train_best": {
    "w_path": 10.0,
    "w_tmag": 10.0,
    "optimizer_step_count": 80,
    "tmag_median_ratio": 1.3685223460197449,
    "tmag_p95_ratio": 4.788041591644287,
    "path_ratio": 1.212698577783018,
    "score": 4.9909298994968525,
    "valid": true
  },
  "heldout": {
    "tmag_median_ratio": 11.866910847001154,
    "tmag_p95_ratio": 94.01508919525963,
    "path_ratio": 6.39102545028478,
    "signed_tdir_mean_deg": 79.71209808275077
  },
  "scale_generalization_failed": true,
  "likely_reason": "motion_distribution_shift"
}

## 6. confidence / entropy 是否可靠
{
  "confidence_available": true,
  "confidence_correlates_with_tdir": false,
  "confidence_correlates_with_tmag": true,
  "high_confidence_reliable": false,
  "confidence_not_reliable": true
}

## 7. 数据可观测性对 tdir/path_ratio 的影响
final_classification=DATA2_TRAIN_EVAL_MOTION_DISTRIBUTION_SHIFT

## 8. 是否继续 STRUCT 路线
{
  "keep_s5e15_as_best_candidate": true,
  "continue_struct_line": false,
  "do_struct1c": false,
  "do_gen1_external_dataset": true,
  "do_final8_freeze": false,
  "main_next_step": "GEN1_external_dataset_feasibility"
}

## 9. 是否需要外部数据集泛化
如果 low-motion / split shift 被确认是主因，则更值得做 GEN1 external dataset feasibility，而不是继续在 adjacent tdir 上细调主干。

## 10. caveats
- 本审计不训练新模型。
- 本审计不刷新 official S5 locked metrics。
- confidence 相关结论主要来自已存在的 STRUCT1A 审计产物；STRUCT1B 当前提交产物中不包含 raw provenance。
