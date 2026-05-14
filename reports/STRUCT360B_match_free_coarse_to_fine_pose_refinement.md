# STRUCT360B match-free coarse-to-fine pose refinement

## 1. Executive summary
- training executed true/false: `true`
- checkpoint saved true/false: `true`
- best checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/STRUCT360B_match_free_coarse_to_fine/best_val.pt`
- classification: `balanced_success`
- whether STRUCT360B improves over TRAIN360D/H/STRUCT360A: `vs TRAIN360D=partial, vs TRAIN360H=better, vs STRUCT360A=better`

## 2. LoFTR-inspired but match-free design
- borrowed coarse-to-fine hierarchy from LoFTR: `true`
- did not use explicit matching: `true`
- did not use confidence matrix / MNN / match list: `true`
- did not use RANSAC / PnP / BA: `true`
- fine stage is residual pose refinement, not correspondence refinement.

## 3. Architecture
- coarse stage: `TRAIN360D coarse interaction predicts R0 / tdir0 / log_tmag0`.
- fine pose-conditioned branch: `latent cross-image fine-token refinement conditioned on coarse pose embedding`.
- pose embedding: `R0(9) + tdir0(3) + log_tmag0(1) + coarse pair context stats`.
- residual pose heads: `predict ΔR / Δtdir / Δlog_tmag with conservative gate`.
- pose composition: `R_final = ΔR ∘ R0, tdir_final = normalize(tdir0 + 0.25 * Δtdir), log_tmag_final = log_tmag0 + 0.25 * Δlog_tmag`.
- auxiliary coarse loss: `enabled`.
- residual regularization: `enabled`.
- parameter count: `{'total': 13737769, 'trainable': 11568747}`

## 4. Initialization
- init checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`
- strict/non-strict load: `non-strict`
- missing keys: `['struct360b_pose_embed.net.0.weight', 'struct360b_pose_embed.net.0.bias', 'struct360b_pose_embed.net.1.weight', 'struct360b_pose_embed.net.1.bias', 'struct360b_pose_embed.net.3.weight', 'struct360b_pose_embed.net.3.bias', 'struct360b_fine_refiner.pos_enc.net.0.weight', 'struct360b_fine_refiner.pos_enc.net.0.bias', 'struct360b_fine_refiner.pos_enc.net.1.weight', 'struct360b_fine_refiner.pos_enc.net.1.bias', 'struct360b_fine_refiner.pos_enc.net.3.weight', 'struct360b_fine_refiner.pos_enc.net.3.bias', 'struct360b_fine_refiner.pose_to_film_a.weight', 'struct360b_fine_refiner.pose_to_film_a.bias', 'struct360b_fine_refiner.pose_to_film_b.weight', 'struct360b_fine_refiner.pose_to_film_b.bias', 'struct360b_fine_refiner.pose_to_token.weight', 'struct360b_fine_refiner.pose_to_token.bias', 'struct360b_fine_refiner.norm_a.weight', 'struct360b_fine_refiner.norm_a.bias', 'struct360b_fine_refiner.norm_b.weight', 'struct360b_fine_refiner.norm_b.bias', 'struct360b_fine_refiner.cross_attn_ab.in_proj_weight', 'struct360b_fine_refiner.cross_attn_ab.in_proj_bias', 'struct360b_fine_refiner.cross_attn_ab.out_proj.weight', 'struct360b_fine_refiner.cross_attn_ab.out_proj.bias', 'struct360b_fine_refiner.cross_attn_ba.in_proj_weight', 'struct360b_fine_refiner.cross_attn_ba.in_proj_bias', 'struct360b_fine_refiner.cross_attn_ba.out_proj.weight', 'struct360b_fine_refiner.cross_attn_ba.out_proj.bias', 'struct360b_fine_refiner.ffn_a.0.weight', 'struct360b_fine_refiner.ffn_a.0.bias', 'struct360b_fine_refiner.ffn_a.1.weight', 'struct360b_fine_refiner.ffn_a.1.bias', 'struct360b_fine_refiner.ffn_a.4.weight', 'struct360b_fine_refiner.ffn_a.4.bias', 'struct360b_fine_refiner.ffn_b.0.weight', 'struct360b_fine_refiner.ffn_b.0.bias', 'struct360b_fine_refiner.ffn_b.1.weight', 'struct360b_fine_refiner.ffn_b.1.bias', 'struct360b_fine_refiner.ffn_b.4.weight', 'struct360b_fine_refiner.ffn_b.4.bias', 'struct360b_fine_refiner.update_gate.weight', 'struct360b_fine_refiner.update_gate.bias', 'struct360b_residual_head.net.0.weight', 'struct360b_residual_head.net.0.bias', 'struct360b_residual_head.net.1.weight', 'struct360b_residual_head.net.1.bias', 'struct360b_residual_head.net.3.weight', 'struct360b_residual_head.net.3.bias', 'struct360b_residual_head.rot_head.weight', 'struct360b_residual_head.rot_head.bias', 'struct360b_residual_head.tdir_head.weight', 'struct360b_residual_head.tdir_head.bias', 'struct360b_residual_head.log_tmag_head.weight', 'struct360b_residual_head.log_tmag_head.bias', 'struct360b_residual_head.gate_head.weight', 'struct360b_residual_head.gate_head.bias']`
- unexpected keys: `[]`
- new fine branch params: `['struct360b_pose_embed.net.0.weight', 'struct360b_pose_embed.net.0.bias', 'struct360b_pose_embed.net.1.weight', 'struct360b_pose_embed.net.1.bias', 'struct360b_pose_embed.net.3.weight', 'struct360b_pose_embed.net.3.bias', 'struct360b_fine_refiner.pos_enc.net.0.weight', 'struct360b_fine_refiner.pos_enc.net.0.bias', 'struct360b_fine_refiner.pos_enc.net.1.weight', 'struct360b_fine_refiner.pos_enc.net.1.bias', 'struct360b_fine_refiner.pos_enc.net.3.weight', 'struct360b_fine_refiner.pos_enc.net.3.bias', 'struct360b_fine_refiner.pose_to_film_a.weight', 'struct360b_fine_refiner.pose_to_film_a.bias', 'struct360b_fine_refiner.pose_to_film_b.weight', 'struct360b_fine_refiner.pose_to_film_b.bias', 'struct360b_fine_refiner.pose_to_token.weight', 'struct360b_fine_refiner.pose_to_token.bias', 'struct360b_fine_refiner.norm_a.weight', 'struct360b_fine_refiner.norm_a.bias', 'struct360b_fine_refiner.norm_b.weight', 'struct360b_fine_refiner.norm_b.bias', 'struct360b_fine_refiner.cross_attn_ab.in_proj_weight', 'struct360b_fine_refiner.cross_attn_ab.in_proj_bias', 'struct360b_fine_refiner.cross_attn_ab.out_proj.weight', 'struct360b_fine_refiner.cross_attn_ab.out_proj.bias', 'struct360b_fine_refiner.cross_attn_ba.in_proj_weight', 'struct360b_fine_refiner.cross_attn_ba.in_proj_bias', 'struct360b_fine_refiner.cross_attn_ba.out_proj.weight', 'struct360b_fine_refiner.cross_attn_ba.out_proj.bias', 'struct360b_fine_refiner.ffn_a.0.weight', 'struct360b_fine_refiner.ffn_a.0.bias', 'struct360b_fine_refiner.ffn_a.1.weight', 'struct360b_fine_refiner.ffn_a.1.bias', 'struct360b_fine_refiner.ffn_a.4.weight', 'struct360b_fine_refiner.ffn_a.4.bias', 'struct360b_fine_refiner.ffn_b.0.weight', 'struct360b_fine_refiner.ffn_b.0.bias', 'struct360b_fine_refiner.ffn_b.1.weight', 'struct360b_fine_refiner.ffn_b.1.bias', 'struct360b_fine_refiner.ffn_b.4.weight', 'struct360b_fine_refiner.ffn_b.4.bias', 'struct360b_fine_refiner.update_gate.weight', 'struct360b_fine_refiner.update_gate.bias', 'struct360b_residual_head.net.0.weight', 'struct360b_residual_head.net.0.bias', 'struct360b_residual_head.net.1.weight', 'struct360b_residual_head.net.1.bias', 'struct360b_residual_head.net.3.weight', 'struct360b_residual_head.net.3.bias', 'struct360b_residual_head.rot_head.weight', 'struct360b_residual_head.rot_head.bias', 'struct360b_residual_head.tdir_head.weight', 'struct360b_residual_head.tdir_head.bias', 'struct360b_residual_head.log_tmag_head.weight', 'struct360b_residual_head.log_tmag_head.bias', 'struct360b_residual_head.gate_head.weight', 'struct360b_residual_head.gate_head.bias']`

## 5. Training setup
- freeze/unfreeze strategy: `2-epoch fine-only warmup, then partial unfreeze through epoch 5`
- epochs: `5`
- LR groups: `warmup coarse=0.0, warmup fine=0.0001, coarse=1e-05, fine=0.0001`
- loss weights: `final=1.0, coarse_aux=0.5, residual_reg=0.05`
- alpha/beta: `alpha=0.25, beta=0.25`
- observability/k-step/scale settings: `{'rot_weight': 1.0, 'tdir_weight': 1.0, 'tmag_weight': 0.5, 'scale_stability_weight': 0.1, 'tmag_loss_type': 'log_smooth_l1', 'coarse_aux_weight': 0.5, 'residual_reg_weight': 0.05, 'components': {'enable_observability': True, 'enable_k_step_balancing': True, 'enable_scale_stabilization': True}, 'k_step_balancing': {'weight_k1': 1.0, 'weight_k2': 1.05, 'weight_k3': 1.0, 'weight_k5': 0.9}, 'observability': {'tmag_epsilon': 1e-06, 'near_zero_tmag': 0.02, 'low_tmag': 0.05, 'medium_tmag': 0.15, 'high_tmag': 0.4, 'min_weight': 0.2, 'max_weight': 2.0, 'k1_factor': 1.05, 'k2_factor': 1.0, 'k3_factor': 0.95, 'k5_factor': 0.85}, 'scale_stability': {'collapse_ratio': 0.1, 'explosion_ratio': 10.0, 'mean_log_bias_weight': 1.0}}`
- val score: `{'signed_tdir_weight': 1.0, 'anti_parallel_weight': 60.0, 'tmag_ratio_weight': 20.0, 'path_ratio_weight': 10.0, 'rot_weight': 0.0}`
- train subset info: `{'subset_used': True, 'subset_count': 256, 'original_count': 12154, 'subset_seed': 3407}`
- runtime sec: `1968.94`
- system info: `{'device': 'cuda', 'cuda_available': True, 'torch_version': '2.9.1+cu128', 'gpu_name': 'NVIDIA GeForce RTX 3060 Laptop GPU'}`

## 6. Validation results
- coarse metrics: `{'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.3272813782821162, 'rot_median_deg': 1.1138176918029785, 'rot_p90_deg': 2.8179727554321308, 'signed_tdir_mean_deg': 104.85900952700996, 'signed_tdir_median_deg': 115.77619090382672, 'signed_tdir_p90_deg': 146.9746110487188, 'unsigned_tdir_mean_deg': 53.89997560618801, 'unsigned_tdir_median_deg': 52.60689770270267, 'unsigned_tdir_p90_deg': 81.21311571202196, 'anti_parallel_rate': 0.6679146387120929, 'tmag_ratio_p10': 0.2739630566120189, 'tmag_median_ratio': 1.0119925762813256, 'tmag_ratio_p50': 1.0119925762813256, 'tmag_mean_ratio': 1.5006580178110513, 'tmag_ratio_p90': 4.043361593203142, 'tmag_p90_ratio': 4.043361593203142, 'log_tmag_mae': 0.7426554340727693, 'scale_collapse_rate': 0.0, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5718235306972821, 'path_length_pred': 3474.2051471471786, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- final metrics: `{'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.3101146514525757, 'rot_median_deg': 1.0966436862945557, 'rot_p90_deg': 2.784472465515138, 'signed_tdir_mean_deg': 104.85806009386242, 'signed_tdir_median_deg': 115.77549118516109, 'signed_tdir_p90_deg': 146.9734111113021, 'unsigned_tdir_mean_deg': 53.90031262159957, 'unsigned_tdir_median_deg': 52.60732759612557, 'unsigned_tdir_p90_deg': 81.21254120916193, 'anti_parallel_rate': 0.6679146387120929, 'tmag_ratio_p10': 0.2739609081887106, 'tmag_median_ratio': 1.0119845231298872, 'tmag_ratio_p50': 1.0119845231298872, 'tmag_mean_ratio': 1.5006461519194054, 'tmag_ratio_p90': 4.043329909949669, 'tmag_p90_ratio': 4.043329909949669, 'log_tmag_mae': 0.7426552344074488, 'scale_collapse_rate': 0.0, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5718190267578538, 'path_length_pred': 3474.1777827441692, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- comparison to TRAIN360D val: `signed_tdir 100.96635328042524 -> 104.85806009386242, anti_parallel 0.6475102957693748 -> 0.6679146387120929, tmag_ratio 0.9443229312709943 -> 1.0119845231298872, path_ratio 0.5193150078709948 -> 0.5718190267578538`
- comparison to TRAIN360H val: `signed_tdir 104.39754351706829 -> 104.85806009386242, anti_parallel 0.6523773867465369 -> 0.6679146387120929, tmag_ratio 0.9151370322858172 -> 1.0119845231298872, path_ratio 0.5895168494209848 -> 0.5718190267578538`
- comparison to STRUCT360A val: `signed_tdir 103.0691856567912 -> 104.85806009386242, anti_parallel 0.615874204417821 -> 0.6679146387120929, tmag_ratio 1.0053238154085171 -> 1.0119845231298872, path_ratio 0.5238780909029992 -> 0.5718190267578538`

## 7. Test results
- coarse metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.345781279033832, 'rot_median_deg': 0.2373882383108139, 'rot_p90_deg': 6.743474245071431, 'signed_tdir_mean_deg': 45.92485488999007, 'signed_tdir_median_deg': 23.83328596237879, 'signed_tdir_p90_deg': 139.31859595863028, 'unsigned_tdir_mean_deg': 26.702938320467897, 'unsigned_tdir_median_deg': 20.46589612705367, 'unsigned_tdir_p90_deg': 59.3389262142436, 'anti_parallel_rate': 0.2042670881074674, 'tmag_ratio_p10': 0.30519594075222767, 'tmag_median_ratio': 0.8509821717490575, 'tmag_ratio_p50': 0.8509821717490575, 'tmag_mean_ratio': 1.1327566582403266, 'tmag_ratio_p90': 2.4785317255858605, 'tmag_p90_ratio': 2.4785317255858605, 'log_tmag_mae': 0.6511783092916308, 'scale_collapse_rate': 0.001382852627419992, 'scale_explosion_rate': 0.0, 'path_ratio': 0.656661907762025, 'path_length_pred': 5353.987677514553, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- final metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.33618272123001, 'rot_median_deg': 0.22028695046901703, 'rot_p90_deg': 6.725292921066288, 'signed_tdir_mean_deg': 45.924701790734524, 'signed_tdir_median_deg': 23.833112391604995, 'signed_tdir_p90_deg': 139.31838917253972, 'unsigned_tdir_mean_deg': 26.70284856430563, 'unsigned_tdir_median_deg': 20.46473381807475, 'unsigned_tdir_p90_deg': 59.33861771143946, 'anti_parallel_rate': 0.2042670881074674, 'tmag_ratio_p10': 0.305193587049523, 'tmag_median_ratio': 0.8509755191001558, 'tmag_ratio_p50': 0.8509755191001558, 'tmag_mean_ratio': 1.1327478057194367, 'tmag_ratio_p90': 2.478512479191511, 'tmag_p90_ratio': 2.478512479191511, 'log_tmag_mae': 0.6511796877899803, 'scale_collapse_rate': 0.001382852627419992, 'scale_explosion_rate': 0.0, 'path_ratio': 0.656656775908536, 'path_length_pred': 5353.9458357691765, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
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

## 8. Residual behavior analysis
- residual magnitude: `{'delta_rot_mean_deg': 0.04087159471422234, 'delta_tdir_norm_mean': 0.00016387407589857057, 'delta_log_tmag_abs_mean': 3.126561110484965e-05}`
- fine gate stats: `{'mean': 0.03950810081541468, 'median': 0.039508022367954254, 'max': 0.039510324597358704}`
- whether fine branch is too aggressive: `judge from residual magnitudes plus gate stats; current gate mean=0.03950810081541468`
- whether coarse branch preserved direction: `coarse signed_tdir=45.92485488999007, final signed_tdir=45.924701790734524`
- whether scale/path improved: `coarse tmag_ratio=0.8509821717490575, final tmag_ratio=0.8509755191001558, coarse path_ratio=0.656661907762025, final path_ratio=0.656656775908536`

## 9. Failure analysis
- if anti_parallel worsens: `reduce fine residual gate or add rotation-aware bias in STRUCT360C without converting affinity into explicit matches.`
- if scale worsens: `revisit beta / residual log-tmag scale and consider TRAIN360H-style scale weighting for a follow-up retrain.`
- if fine branch collapses: `inspect residual gate saturation and whether warmup froze coarse long enough.`
- if val/test discrepancy remains: `current discrepancy = 58.933358303127896; prefer STRUCT360C rotation-aware attention bias before any larger sweep.`

## 10. Next recommendation
- `proceed_to_TRAIN360I_final_retrain_with_best_hparams`

## 11. Compliance checklist
- `real_training_executed = true`
- `learned_weights_saved = true`
- `train360c_checkpoint_modified = false`
- `train360d_checkpoint_modified = false`
- `train360h_checkpoint_modified = false`
- `struct360a_checkpoint_modified = false`
- `explicit_matching_used = false`
- `confidence_matrix_used_for_matching = false`
- `mnn_match_selection_used = false`
- `match_list_output = false`
- `ransac_used = false`
- `pnp_used = false`
- `bundle_adjustment_used = false`
- `hkust_360dvo_teacher_used = false`
- `base360_outputs_used_as_training_input = false`
- `train_manifest_used = true`
- `val_manifest_used_for_selection_only = true`
- `test_manifest_used_for_final_eval_only = true`
- `direct_glob_data_360dvo_sequences = false`
- `random_pair_split_used = false`
- `s5_locked_metrics_modified = false`
- `large_checkpoints_committed_to_git = false`
