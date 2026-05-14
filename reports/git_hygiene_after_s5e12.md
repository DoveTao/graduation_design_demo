# Git hygiene after S5E12

## 1. 当前分支
- `experiment/s5e1-traceable-adjacent-dense`

## 2. remote 信息
- remote: `origin`
- URL: `git@github.com:DoveTao/graduation_design_demo.git`
- upstream: `origin/experiment/s5e1-traceable-adjacent-dense`

## 3. working tree 初始状态
- 初始状态包含大量 S5D2-S5E12 的 untracked configs / tools / reports / checkpoints / lightweight summary results。
- 另有一个已跟踪修改文件：`reports/external_algorithm_baseline_comparison.md`。
- 已先写入 `logs/git_hygiene_status_before_s5e12.log`。

## 4. 大文件审计结果
- `files_over_20mb`:
  - `./checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
  - `./logs/s5d12_dense_fill_code_search.log`
  - `./logs/s5d6_final_s5_pairwise_source_search.log`
- 这些文件未提交。
- `*.pt` / `*.npz` 等大权重与缓存继续留在本地，不推远端。
- 本次不需要 Git LFS。

## 5. .gitignore 是否修改
- 已修改。
- 原因：补充忽略可复现但不需要推远端的 raw trajectory / provenance / raw pairwise replay exports。
- 没有粗暴忽略整个 `external_baselines/results/`，只针对 raw `.txt` / `.jsonl` / `.npz` 等重产物做定向忽略。

## 6. staged / committed 文件分组
- Commit 0: `.gitignore` hygiene
- Commit 1: S5D2-S5E12 code / configs / tests / scripts
- Commit 2: S5D2-S5E12 reports / checkpoints / summary JSON / S5E12 feature gate artifacts
- Commit 3: S5D6-S5D13 remaining replay / traceability tools

## 7. commit hash 列表
- `b56c7e1` Repo: update gitignore for experimental artifacts
- `36cffac` S5D2-S5E12: add experimental code and tests
- `65979f0` S5D2-S5E12: add reports checkpoints and summaries
- `2c5e616` S5D6-S5D13: add remaining replay and traceability tools

## 8. remote push 状态
- 已 push 到 `origin/experiment/s5e1-traceable-adjacent-dense`。

## 9. tag 状态
- 已创建 annotated tag：`s5e12-feature-gate-passed`
- 已推送到远端。

## 10. validation 结果
- `verify_final_candidate`: PASS
- `project_health_check`: PASS
- `s6_eval_only`: PASS
- `unittest`: PASS
- 说明：`run_serial_validation_guard.sh` 外层 wrapper 仍有历史性的挂住现象，但子步骤结果已单独确认并写入 `logs/git_hygiene_validation_after_s5e12.log`。

## 11. 未提交但保留在本地的文件
- `./checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- `./logs/git_hygiene_status_before_s5e12.log`
- `./logs/git_hygiene_large_files_s5e12.log`
- `./logs/git_hygiene_validation_after_s5e12.log`
- `./logs/s5d12_dense_fill_code_search.log`
- `./logs/s5d6_final_s5_pairwise_source_search.log`
- `./external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.jsonl`
- `./external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.npz`
- `./external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.jsonl`
- `./external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.npz`
- `./external_baselines/results/s5_pairwise_replay_s5d10/componentwise_replay_declared/`
- `./external_baselines/results/s5_pairwise_replay_s5d10/componentwise_replay_inverse/`

## 12. 明确声明
- S5 locked metrics/policy were not changed.
- S5E12 is feature-gate only and does not replace official S5 locked result.
