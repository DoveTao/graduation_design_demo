# MF1a Chain Refiner Smoke Report

## Scope

MF1a is a multi-frame / chain-level refiner smoke experiment. It is not a final candidate, does not run final test, and does not replace S5.

## Motivation

JRT1 closed as `NO_STABLE_JRT1_TRAJECTORY_GAIN`, suggesting pair-level residual correction is not enough for trajectory accumulation. MF1a tests whether a lightweight chain-context refiner can provide a better smoke signal.

## Dataset

- source: `RflyPanoPanoramaPairsEvalFixedKList train split with S5 k=1 predictions`
- window_size: `8`
- train/val chains: `34` / `17`
- no test GT used for training: `True`

## Metrics

| variant | ATE_proxy | drift_proxy | path_ratio_proxy | rot_mean_deg | tdir_mean_deg | tdir_mean_cosine | tmag_mean_log_error | num_chains | num_pairs | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A_s5_chain_reference | 0.914107 | 1.755750 | 1.445230 | 20.614054 | 19.020077 | 0.942108 | 0.446958 | 17 | 136 | ok |
| B_pairwise_jrt_style_reference | 0.276064 | 0.442970 | 1.445230 | 4.529860 | 13.534210 | 0.968626 | 0.446958 | 17 | 136 | ok |
| C_temporal_conv_chain_refiner | 0.395852 | 0.698258 | 1.445230 | 5.470480 | 17.663731 | 0.949316 | 0.446958 | 17 | 136 | ok |
| D_gru_chain_refiner | 0.281478 | 0.456276 | 1.445231 | 4.381053 | 12.343295 | 0.970342 | 0.446958 | 17 | 136 | ok |

## Classification

`MF1A-FRAMEWORK-SMOKE-PASS`

## Caveats

- smoke only
- no final test
- no candidate replacement
- S5 locked metrics unchanged
- no practical-ready claim
