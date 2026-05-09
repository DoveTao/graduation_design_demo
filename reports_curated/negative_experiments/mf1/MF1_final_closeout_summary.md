# MF1 Final Closeout Summary

## Scope

MF1 explored multi-frame / chain-level refinement after JRT1 failed to transfer pair-level gains to trajectory-level gains. MF1 is a new algorithm experiment, and it did not produce a final clean candidate. S5 remains final clean candidate.

## Motivation

JRT1 showed component gains but no trajectory proxy gains. MF1 therefore moved from pair-level residual correction to chain-level temporal refinement, with losses over pair components and local chain accumulation.

## MF1a Summary

classification = `MF1A-FRAMEWORK-SMOKE-PASS`

The chain dataset and train harness worked. GRU/JRT-style smoke variants improved ATE_proxy, drift_proxy, rotation, and tdir relative to the S5 chain reference, while path_ratio remained around 1.445.

| variant | ATE_proxy | drift_proxy | path_ratio | rot_mean | tdir_mean |
|---|---:|---:|---:|---:|---:|
| A_s5_chain_reference | 0.914107 | 1.755750 | 1.445230 | 20.614054 | 19.020077 |
| D_gru_chain_refiner | 0.281478 | 0.456276 | 1.445231 | 4.381053 | 12.343295 |

## MF1b Summary

classification = `NO_STABLE_MF1_CHAIN_GAIN`

MF1b used 3-fold train-CV on the train split only. No test GT was used, and no final test was run.

| variant | ATE_proxy | drift_proxy | path_ratio | rot_mean | tdir_mean | gate_status |
|---|---:|---:|---:|---:|---:|---|
| A_s5_chain_reference | 0.945790 | 1.699604 | 1.429945 | 20.803044 | 31.563480 | REFERENCE |
| B_pairwise_jrt_style_reference | 0.388014 | 0.614514 | 1.429945 | 2.158425 | 17.339511 | FAIL |
| C_temporal_conv_chain_refiner | 0.346048 | 0.552933 | 1.429945 | 1.841854 | 16.351796 | FAIL |
| D_gru_chain_refiner | 0.393461 | 0.635285 | 1.429945 | 2.803790 | 19.246766 | FAIL |
| E_gru_chain_refiner_pathratio_loss | 0.397351 | 0.625256 | 1.429945 | 1.909533 | 18.811731 | FAIL |
| F_gru_chain_refiner_tmag_head_diagnostic | 0.300380 | 0.452558 | 1.036448 | 2.414220 | 19.463260 | FAIL |

## Final Decision

No MF1 variant proceeds to final test. `selected_candidate_for_next_stage = null`. S5 remains final clean candidate.

## Final Classification

`NO_STABLE_MF1_CHAIN_GAIN`

## Thesis / Report Usage

MF1 is a negative but informative result. It shows that chain-level temporal refinement provides stronger trajectory-shape signal than JRT1. However, scale/path-ratio stability remains unresolved and train-CV stability is insufficient. This supports the final limitation that practical improvement likely requires a more principled multi-frame geometric architecture and scale-aware design, not just a lightweight chain refiner.

## Caveats

- train-CV only
- no final test
- proxy trajectory is not official final test
- small sequence protocol
- S5 unchanged
- no practical-ready claim
- F tmag-head is diagnostic only and did not pass stability gate
