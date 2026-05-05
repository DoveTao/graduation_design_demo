# S1d6 minimal trainable dt anchor head proposal

## Goal

Internalize the S1d5 policy into a minimal trainable dt-anchor head while keeping the rest of the model frozen.

## Proposed setup

- initialize from the S1d5 effective factors
- freeze all backbone / encoder / coarse / fine / direct-head parameters
- train only dt-anchor parameters
- keep the checkpoint base as `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`

## Training scope

- do not train encoder/coarse/fine
- do not train the existing mag_head weights
- train only a tiny dt-anchor parameterization

## Objective

- primary objective should include train-chain path length consistency
- do not use pairwise-only tmag loss as the dominant term
- keep pairwise terms weak if included at all

## Success criterion

- must match or improve S1d5:
  - drift about `1.396`
  - ATE about `7.632`
  - path_ratio about `0.935`

## Risks

- training instability
- overfitting to limited train chains
- degrading ATE while chasing path_ratio
- losing the clean reproducibility that S1d5 currently provides

## Recommendation

Only run S1d6 if a model-internal policy is required. Otherwise S1d5 is already strong enough to serve as the current clean exported mainline.
