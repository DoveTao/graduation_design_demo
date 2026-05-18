# FINAL360M full-train thesis main

## Summary
- FINAL360M full retrain executed: `true`
- wall time budget hours: `8.0`
- wall time used: `19286.46 sec`
- full train manifest used: `true`
- train subset disabled: `true`
- train sample count: `12154`
- val sample count: `5342`
- test sample count: `5062`
- epochs requested: `20`
- epochs completed: `20`
- best epoch: `10`
- selected by full val: `true`
- test used for selection: `false`
- init checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`
- selected checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- FINAL360M test rot_mean: `2.322841551013553`
- FINAL360M test signed_tdir_mean: `56.07180781302633`
- FINAL360M test anti_parallel_rate: `0.20762544448834452`
- FINAL360M test tmag_median_ratio: `0.8838564229456283`
- FINAL360M test path_ratio: `0.6814703365128686`
- old FINAL360I test rot_mean: `2.330108616583517`
- old FINAL360I test signed_tdir_mean: `45.264702006380205`
- old FINAL360I test anti_parallel_rate: `0.20169893322797314`
- old FINAL360I test tmag_median_ratio: `0.8344251368086006`
- old FINAL360I test path_ratio: `0.6403519796204528`
- compared to old FINAL360I: `mixed`
- thesis main model recommendation: `thesis_main_model`
- needs trajectory reevaluation: `true`
- needs THESIS362 table update: `true`
- existing metrics modified: `false`
- checkpoints committed: `false`

## Baseline comparisons
- TRAIN360D: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.3460544469162485, 'rot_median_deg': 0.2373882383108139, 'rot_p90_deg': 6.743095779418965, 'signed_tdir_mean_deg': 44.986174454415895, 'signed_tdir_median_deg': 21.105894321746558, 'signed_tdir_p90_deg': 139.97897615069112, 'unsigned_tdir_mean_deg': 25.543904119901278, 'unsigned_tdir_median_deg': 18.518269029988943, 'unsigned_tdir_p90_deg': 57.49787609422275, 'anti_parallel_rate': 0.19695772421967603, 'tmag_ratio_p10': 0.2636780983810221, 'tmag_median_ratio': 0.7572524310356576, 'tmag_ratio_p50': 0.7572524310356576, 'tmag_mean_ratio': 0.994777083491742, 'tmag_ratio_p90': 2.2236141016906847, 'tmag_p90_ratio': 2.2236141016906847, 'log_tmag_mae': 0.683274985087713, 'scale_collapse_rate': 0.009877518767285659, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5787942300950458, 'path_length_pred': 4719.106040894985, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- TRAIN360H: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 3.1616807945222782, 'rot_median_deg': 1.4338136315345764, 'rot_p90_deg': 7.089140510559084, 'signed_tdir_mean_deg': 53.73375854133177, 'signed_tdir_median_deg': 32.94996166945526, 'signed_tdir_p90_deg': 138.49944561811802, 'unsigned_tdir_mean_deg': 34.57963931476379, 'unsigned_tdir_median_deg': 29.76936998575061, 'unsigned_tdir_p90_deg': 62.74972042000828, 'anti_parallel_rate': 0.20209403397866457, 'tmag_ratio_p10': 0.2765550670704756, 'tmag_median_ratio': 0.8503917514492008, 'tmag_ratio_p50': 0.8503917514492008, 'tmag_mean_ratio': 1.1252581257212217, 'tmag_ratio_p90': 2.5432877639863536, 'tmag_p90_ratio': 2.5432877639863536, 'log_tmag_mae': 0.6818262532323421, 'scale_collapse_rate': 0.014816278150928487, 'scale_explosion_rate': 0.0, 'path_ratio': 0.6493755813576354, 'path_length_pred': 5294.57978233695, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- BASE360D: `{'rot_mean_deg': 0.8561815635888619, 'rot_median_deg': 0.14195490039744968, 'signed_tdir_mean_deg': 128.40257804384538, 'signed_tdir_median_deg': 145.2855196105992, 'unsigned_tdir_mean_deg': 33.72977868753099, 'anti_parallel_rate': 0.8148952983010668, 'tmag_median_ratio': 0.039909732236488166, 'tmag_mean_ratio': 0.04229484518756699, 'tmag_ratio_p10': None, 'tmag_ratio_p90': None, 'log_tmag_mae': 3.1876028546854527, 'scale_collapse_rate': None, 'scale_explosion_rate': None, 'path_ratio': 0.04354171873324947, 'pair_component_path_ratio': 0.04354171873324947, 'coverage': 1.0}`

## Selection and lineage
- selected checkpoint was chosen by full validation score only: `true`
- mini-val monitor subset: `{'subset_count': 1024, 'subset_seed': 0}`
- candidate records: `[{'epoch': 10, 'mini_val_score': 165.74220376149628, 'checkpoint_path': '/home/dovetao/graduation_design_demo/checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/candidate_epoch_10.pt', 'full_val_score': 163.8322659875625, 'full_val_metrics': {'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.1201869844009549, 'rot_median_deg': 0.9230207800865173, 'rot_p90_deg': 2.399928808212282, 'signed_tdir_mean_deg': 103.7740035306785, 'signed_tdir_median_deg': 116.6162568468195, 'signed_tdir_p90_deg': 145.84198113798988, 'unsigned_tdir_mean_deg': 52.02262121719085, 'unsigned_tdir_median_deg': 54.21424231207092, 'unsigned_tdir_p90_deg': 76.19269938389407, 'anti_parallel_rate': 0.6843878697117185, 'tmag_ratio_p10': 0.26346820530023857, 'tmag_median_ratio': 1.0298021805337867, 'tmag_ratio_p50': 1.0298021805337867, 'tmag_mean_ratio': 1.4814743555145444, 'tmag_ratio_p90': 3.8296909240814427, 'tmag_p90_ratio': 3.8296909240814427, 'log_tmag_mae': 0.7541720929177724, 'scale_collapse_rate': 0.00037439161362785476, 'scale_explosion_rate': 0.0013103706476974915, 'path_ratio': 0.559709811105132, 'path_length_pred': 3400.6063099205494, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}}, {'epoch': 19, 'mini_val_score': 166.9833427635305, 'checkpoint_path': '/home/dovetao/graduation_design_demo/checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/candidate_epoch_19.pt', 'full_val_score': 165.3330202260504, 'full_val_metrics': {'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.501104569714621, 'rot_median_deg': 1.3082078099250793, 'rot_p90_deg': 3.1515074014663704, 'signed_tdir_mean_deg': 103.862875087544, 'signed_tdir_median_deg': 116.751327138042, 'signed_tdir_p90_deg': 147.07092956772988, 'unsigned_tdir_mean_deg': 51.504174397043734, 'unsigned_tdir_median_deg': 53.58242721820285, 'unsigned_tdir_p90_deg': 76.74669844280527, 'anti_parallel_rate': 0.6931860726319731, 'tmag_ratio_p10': 0.22555697588208382, 'tmag_median_ratio': 0.9756487402013543, 'tmag_ratio_p50': 0.9756487402013543, 'tmag_mean_ratio': 1.4295625717580602, 'tmag_ratio_p90': 3.5659040253145164, 'tmag_p90_ratio': 3.5659040253145164, 'log_tmag_mae': 0.7972723976789383, 'scale_collapse_rate': 0.006551853238487458, 'scale_explosion_rate': 0.0011231748408835642, 'path_ratio': 0.512653000879723, 'path_length_pred': 3114.7051472067833, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}}, {'epoch': 15, 'mini_val_score': 167.20651939373363, 'checkpoint_path': '/home/dovetao/graduation_design_demo/checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/candidate_epoch_15.pt', 'full_val_score': 165.77287184875385, 'full_val_metrics': {'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.3462482334279537, 'rot_median_deg': 1.0682618618011475, 'rot_p90_deg': 2.9610262870788584, 'signed_tdir_mean_deg': 103.8255441984078, 'signed_tdir_median_deg': 116.87989693513951, 'signed_tdir_p90_deg': 146.57467630561737, 'unsigned_tdir_mean_deg': 52.19779720675733, 'unsigned_tdir_median_deg': 53.2496581293581, 'unsigned_tdir_p90_deg': 78.97628229748038, 'anti_parallel_rate': 0.6929988768251591, 'tmag_ratio_p10': 0.23213823817340512, 'tmag_median_ratio': 0.9461677554531536, 'tmag_ratio_p50': 0.9461677554531536, 'tmag_mean_ratio': 1.3739274339041458, 'tmag_ratio_p90': 3.491083933470917, 'tmag_p90_ratio': 3.491083933470917, 'log_tmag_mae': 0.7721040017472607, 'scale_collapse_rate': 0.005803070011231749, 'scale_explosion_rate': 0.0013103706476974915, 'path_ratio': 0.5086388117863064, 'path_length_pred': 3090.3163005411625, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}}]`
- current artifact source manifest: `reports/FINAL360M_fulltrain_guarded_metric_source_manifest.json`

## Next task
- recommended next task: `rerun trajectory-level evaluation and update THESIS362 tables/figures if FINAL360M remains selected`
