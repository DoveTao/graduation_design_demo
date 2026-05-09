# MF1b Train-CV Chain Refiner

## Scope

MF1b is train-CV only. It runs no final test, selects no final candidate, and S5 remains final clean candidate.

## MF1a Recap

MF1a smoke passed. GRU and JRT-style variants improved proxy ATE/drift, while path_ratio stayed around 1.445 because scale behavior was unchanged.

## Dataset and Folds

- dataset source: `RflyPanoPanoramaPairsEvalFixedKList train split with S5 k=1 predictions`
- no test GT used: `True`

| fold | train_chains | val_chains |
| --- | ---: | ---: |
| 0 | 34 | 17 |
| 1 | 34 | 17 |
| 2 | 34 | 17 |

## Variants

| variant | role |
| --- | --- |
| A_s5_chain_reference | no refiner |
| B_pairwise_jrt_style_reference | independent pairwise residual across chain |
| C_temporal_conv_chain_refiner | temporal convolution over chain |
| D_gru_chain_refiner | GRU over chain |
| E_gru_chain_refiner_pathratio_loss | GRU with stronger path-ratio penalty |
| F_gru_chain_refiner_tmag_head_diagnostic | GRU with diagnostic tmag residual head |

## Mean CV Results

| variant | ATE_proxy | drift_proxy | path_ratio_proxy | rot_mean | tdir_mean | tdir_cos | gate_status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A_s5_chain_reference | 0.945790 | 1.699604 | 1.429945 | 20.803044 | 31.563480 | 0.787722 | REFERENCE |
| B_pairwise_jrt_style_reference | 0.388014 | 0.614514 | 1.429945 | 2.158425 | 17.339511 | 0.880433 | FAIL |
| C_temporal_conv_chain_refiner | 0.346048 | 0.552933 | 1.429945 | 1.841854 | 16.351796 | 0.886726 | FAIL |
| D_gru_chain_refiner | 0.393461 | 0.635285 | 1.429945 | 2.803790 | 19.246766 | 0.858823 | FAIL |
| E_gru_chain_refiner_pathratio_loss | 0.397351 | 0.625256 | 1.429945 | 1.909533 | 18.811731 | 0.858603 | FAIL |
| F_gru_chain_refiner_tmag_head_diagnostic | 0.300380 | 0.452558 | 1.036448 | 2.414220 | 19.463260 | 0.860672 | FAIL |

## Gate Decision

Final classification: `NO_STABLE_MF1_CHAIN_GAIN`
Selected candidate for next stage: `None`

## Interpretation

The report compares all variants against A_s5_chain_reference on train-CV folds. It must not be read as a final clean result. The key blocker is whether path_ratio enters the [0.90, 1.05] gate while ATE/drift and component metrics remain stable.

## Final Classification

`NO_STABLE_MF1_CHAIN_GAIN`

## Next Step

If gate pass, prepare a separate MF1c final-test plan. Otherwise close out or revise the model objective.

## Caveats

- train-CV only
- no final test
- no candidate replacement
- S5 unchanged
- no deployment-readiness claim
