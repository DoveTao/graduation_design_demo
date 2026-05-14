# Checkpoints Live Cleanup Summary

## 1. Cleanup size change

- before: `7.3G`
- after: `454M`
- freed: `~6.8G`

## 2. Actions

- moved from live checkpoints to archive: `76`
- deleted from live checkpoints after archive confirmation: `0`

## 3. Largest moved items

- `T51a4_from_t51a3b_upd500_scale_w001_1200` (668M)
- `T51a3b_from_t51a2_upd200_restore_t51a2cfg_1200` (623M)
- `T51a5_from_t51a3b_upd400_under_w002_target035_800` (623M)
- `T51a3_from_t51a2_upd200_lr5e6_savefix_1200` (577M)
- `T53a_from_t51a3b_upd400_chain_len_w001_target055_600` (577M)
- `T53a2_from_t51a3b_upd400_chain_len_w005_target065_lr5e6_400` (487M)
- `T53a3_from_t51a3b_upd400_chain_len_w010_target075_lr5e6_400` (487M)
- `T53a4_from_t51a3b_upd400_chain_len_w015_target080_lr5e6_400` (487M)
- `T53b2_from_t53b_best_vec_w003_400` (487M)
- `T53b_from_t53a4_vec_w001_400` (487M)
- `T55a_from_t53b_best_no_old_seqturn_400` (487M)
- `T57a_clean_no_dt_v1_400` (442M)
- `O39_c31_smallk_anchor3_tmag_nodetach_1200` (182M)
- `O49m3_o39_seqturn_pair_w007_acosclip_rerun3_260` (182M)

## 4. Kept whitelist

- `checkpoints/S1d5_clean_dt_anchor_policy.json`
- `checkpoints/S1d5_final_repro`
- `checkpoints/S1d5_final_figures`
- `checkpoints/S1d5_final_baseline_comparison.md`
- `checkpoints/S1d5_final_mainline_summary.md`
- `checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md`
- `checkpoints/S1d6_minimal_trainable_dt_anchor_head_proposal.md`
- `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`

## 5. Archive backup

- archive path: `/home/dovetao/graduation_design_demo/archive/s1d5_cleanup_20260505`
- archive total size: `8.3G`

## 6. S1d5 sanity check

- `python tools/eval_clean_policy.py --help`: PASS
- `bash -n scripts/eval_s1d5_clean_policy.sh`: PASS
- `bash scripts/eval_s1d5_clean_policy.sh dryrun`: PASS

## 7. Missing whitelist artifacts

- none

## 8. Current conclusion

- S1d5 remains current clean exported mainline
- F1d remains paused
- S1d6 proposal only, not run
