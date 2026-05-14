# S5D7 GPU-safe validation and pairwise export hook

## Executive summary
- final classification: `S5D7_VALIDATION_RECOVERED_PAIRWISE_EXPORTED`

## S5D6 failure recap
- verify/project_health failed due to s6 eval-only CUDA OOM.

## CUDA OOM audit
- log: `logs/s5d6_s6_lockdown_eval_only.log`
- root cause: `CUDA_OOM`
- location: `tools/s6_final_clean_candidate_lockdown_audit.py`

## GPU-safe validation strategy
- Keep official evaluator default behavior unchanged.
- Use diagnostic-only export hook with per-pair streaming write and no_grad.

## Validation result
- verify_final_candidate: `PASS`
- project_health_check: `PASS`
- s6_eval_only: `PASS`
- unittest: `PASS` (count=108)

## Pairwise export hook design
- diagnostic only
- no prediction modification
- no official locked result replacement
- outputs jsonl/npz/metadata with frame convention fields

## Pairwise export result
- available: `True`
- num_pairs: `132`
- pair_selection: `selected_k1`
- coverage_vs_453: `0.291391`

## Caveats
- S5D7 is diagnostic only.
- S5D7 does not modify predictions.
- S5D7 does not replace official S5 locked result.
- S5 locked metrics/policy unchanged.
