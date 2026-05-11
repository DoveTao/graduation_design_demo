# S5E4 temporal direction head report

## 执行摘要
S5E4 针对 S5E3 的 signed tdir 反向问题，引入 temporal sign disambiguation：使用 S5E2 signed direction prior 与 S5E3 magnitude calibration。`final_classification = S5E4_SIGNED_TDIR_IMPROVED`。

## S5E3 direction failure audit
- anti_parallel_rate = 0.8322295805739515
- severe_wrong_sign_rate = 0.7726269315673289
- direction_abs_good_but_signed_bad_rate = 0.6622516556291391

## 为什么 signed tdir 和 tdir_abs 不能混为一谈
S5E3 的 tdir_abs 改善但 signed tdir 恶化，说明轴线相近但时间方向符号可能反了；tdir_abs 不能替代 signed tdir。

## S5E4 temporal direction / sign head 设计
S5E4 使用 ordered features 和 S5E2 signed direction prior 做 sign disambiguation，magnitude 继续来自 S5E3 log-magnitude head。

## anti-parallel penalty 说明
训练记录中显式监控 `anti_parallel_rate`，并把 dot(t_pred,t_gt)<0 作为 sign error 风险；本轮未使用 scene01/seq03 GT 训练。

## training 结果
- classification = S5E4_TRAINING_SMOKE_ONLY
- anti_parallel_rate_train = 0.2927864214992928
- anti_parallel_rate_val = 0.620253164556962

## traceable dense export coverage
- coverage = 1.0
- all_edges_traceable = True

## component metrics，重点讨论 rot 和 tdir
- rot_mean/median/p90 = 0.9134395040767528 / 0.797895569319923 / 1.3218302317546538
- signed tdir mean/median/p90 = 51.47429479273258 / 42.61639148028056 / 98.61703755727075
- tdir_abs mean/median/p90 = 46.07968707914154 / 42.3027471139478 / 78.37357550448448
- anti_parallel_rate = 0.1368653421633554
- tmag median/mean/p90/p95 = 12.109093390318423 / 19.468279568761744 / 48.30194622818439 / 73.00488324816631
- path_ratio = 2.1345479454214416

## external evaluator none/se3/sim3
- none = {'ate': 28.742908168441225, 'drift': 0.20870319509634433, 'path_ratio': 2.1345479454110814, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}
- se3 = {'ate': 13.75936457268861, 'drift': 0.21604056005578093, 'path_ratio': 2.1345479454110814, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}
- sim3 = {'ate': 4.0211631491566155, 'drift': 0.1299505570591922, 'path_ratio': 2.1345479454110814, 'status': 'ok', 'num_matched_poses': 454, 'tracking_success_rate': 1.0}

## 与 S5E3 / S5E2 比较
{'rot_preserved': True, 'signed_tdir_improved': True, 'tdir_abs_improved_or_preserved': False, 'anti_parallel_rate_reduced': True, 'tmag_improved_or_preserved': True, 'path_ratio_improved': False, 'sim3_ate_improved': False, 'overall_geometry_improved': True}

## 与 ORB-SLAM3 比较
{'coverage_advantage': True, 'rot_close_to_orbslam3': True, 'tdir_gap_remaining': True, 'aligned_ate_gap_to_orbslam3': 13.450820161996127, 'summary': 'S5E4 improves temporal sign relative to S5E3 but remains far from ORB-SLAM3 trajectory accuracy.'}

## 是否更接近 ORB-SLAM3
S5E4 仍和 ORB-SLAM3 有明显 ATE/path_ratio 差距，不能替代 official S5。

## rot / tdir 是否仍是主要差距
rot 保持较好；signed tdir 是本轮重点改善项，但若 tmag/path_ratio 仍高，trajectory 仍会偏差。

## 下一步建议
下一步应训练真正的 temporal visual backbone；ORB-SLAM3 distillation 应另开 S5E5。

## caveats
- S5E4 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 是 external strong baseline。
