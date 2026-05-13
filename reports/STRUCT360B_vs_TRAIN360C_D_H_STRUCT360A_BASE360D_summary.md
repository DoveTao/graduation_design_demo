# STRUCT360B vs TRAIN360C / D / H / STRUCT360A / BASE360D

| model | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| STRUCT360B | val | 104.85806009386242 | 0.6679146387120929 | 1.0119845231298872 | 0.5718190267578538 | 1.0 |
| STRUCT360B | test | 45.924701790734524 | 0.2042670881074674 | 0.8509755191001558 | 0.656656775908536 | 1.0 |
| TRAIN360C | test | 54.956992341554056 | 0.21572500987751878 | 0.7692854374461574 | 0.5888660831060318 | 1.0 |
| TRAIN360D | test | 44.986174454415895 | 0.19695772421967603 | 0.7572524310356576 | 0.5787942300950458 | 1.0 |
| TRAIN360H | test | 53.73375854133177 | 0.20209403397866457 | 0.8503917514492008 | 0.6493755813576354 | 1.0 |
| STRUCT360A | test | 45.806573800840134 | 0.21197155274595023 | 0.7683920813313971 | 0.5926014164348259 | 1.0 |
| T57b | reference | 111.96493221327962 | 0.6746724890829694 | 0.1751560082454769 | 0.14878731297064046 | 1.0 |
| BASE360D | test component | 128.40257804384538 | 0.8148952983010668 | 0.039909732236488166 | 0.04354171873324947 | 1.0 |

- classification: `balanced_success`
- compared to TRAIN360D: `partial`
- compared to TRAIN360H: `better`
- compared to STRUCT360A: `better`
- val/test discrepancy: `58.933358303127896`
- coarse test metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.345781279033832, 'rot_median_deg': 0.2373882383108139, 'rot_p90_deg': 6.743474245071431, 'signed_tdir_mean_deg': 45.92485488999007, 'signed_tdir_median_deg': 23.83328596237879, 'signed_tdir_p90_deg': 139.31859595863028, 'unsigned_tdir_mean_deg': 26.702938320467897, 'unsigned_tdir_median_deg': 20.46589612705367, 'unsigned_tdir_p90_deg': 59.3389262142436, 'anti_parallel_rate': 0.2042670881074674, 'tmag_ratio_p10': 0.30519594075222767, 'tmag_median_ratio': 0.8509821717490575, 'tmag_ratio_p50': 0.8509821717490575, 'tmag_mean_ratio': 1.1327566582403266, 'tmag_ratio_p90': 2.4785317255858605, 'tmag_p90_ratio': 2.4785317255858605, 'log_tmag_mae': 0.6511783092916308, 'scale_collapse_rate': 0.001382852627419992, 'scale_explosion_rate': 0.0, 'path_ratio': 0.656661907762025, 'path_length_pred': 5353.987677514553, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- residual stats: `{'delta_rot_mean_deg': 0.04087159471422234, 'delta_tdir_norm_mean': 0.00016387407589857057, 'delta_log_tmag_abs_mean': 3.126561110484965e-05}`
- gate stats: `{'mean': 0.03950810081541468, 'median': 0.039508022367954254, 'max': 0.039510324597358704}`
