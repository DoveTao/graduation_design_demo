# FINAL360I final retrain and model selection

## 1. Executive summary
- final retrain executed: `true`
- selected checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- selected model: `FINAL360I_struct360b_final_selected`
- final classification: `final_balanced_model`
- current visible comparison set: `FINAL360I`, `STRUCT360B`, `BASE360D`, `T57b`

## 2. Main comparison recap

| model | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `T57b` | reference | 111.964932 | 0.674672 | 0.175156 | 0.148787 |
| `STRUCT360B` | test | 45.924702 | 0.204267 | 0.850976 | 0.656657 |
| `BASE360D` | test component | 128.402578 | 0.814895 | 0.039910 | 0.043542 |
| `FINAL360I` | test | 45.264702 | 0.201699 | 0.834425 | 0.640352 |

## 3. Selection summary
- validation-only selection: `true`
- selected seed/checkpoint: `seed0 -> /home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- fallback to locked `STRUCT360B` best was available but not needed
- architecture change: `false`
- hyperparameter sweep: `false`

## 4. Final interpretation
- `FINAL360I` remains the current pair-level thesis main model.
- Against `STRUCT360B`, it stays in the same quality band while preserving balanced direction and scale behavior.
- Against `T57b` and `BASE360D` component metrics, it is clearly stronger on pair-level translation behavior.
- Trajectory behavior is evaluated separately via `TRAIN360E`; this report remains pair-level only.

## 5. Caveats
- pair-level metrics only; see `TRAIN360E` for sequence-level ATE/path behavior
- `BASE360D` component metrics are trajectory-derived
- model selection used validation only, not test
- two non-promoted follow-ups are retained only as status summaries:
  - `SEQ360A`: `reports/SEQ360A_status_summary.md`
  - `STRUCT360C`: `reports/STRUCT360C_status_summary.md`

## 6. Compliance note
- no metric values were changed during MAINT15
- no checkpoints were modified during MAINT15
