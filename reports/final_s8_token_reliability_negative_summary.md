# S8 Token Reliability Negative Summary

## Motivation

S3b already showed that the current fine-token representation does not provide a stable residual pose readout signal. S8 therefore narrowed the question: even if token features are not suitable for residual pose regression, can fine/spherical tokens still act as a reliability estimator for bad-pair or high-error detection?

## Method

S8 was executed in a staged, resume-friendly diagnostic mode rather than a full router pipeline, because runtime budget was exhausted before the original end-to-end plan could be completed. After the S8b reproduction contract gate passed, we ran a lightweight two-fold logistic probe using cached train rows only, with sample caps of `train_full=512`, `fold_train=256`, and `fold_val=256`. The probe compared four families: `regime_only`, `fine_token_only`, `spherical_token_only`, and `token_plus_regime`. Because the probe stage did not show a clear advantage for token-based features, we skipped the final route test and did not proceed to S8c.

## Main Result

The simple `regime_only` baseline outperformed token-only probing, and adding token summaries on top of regime features did not improve over the regime baseline.

- Best token-only family: `fine_token_only`
- `fine_token_only` vs `regime_only`: `delta_auc=-0.144171`, `delta_accuracy=-0.095895`, `delta_high_error_recall=-0.699057`
- `token_plus_regime` vs `regime_only`: `delta_auc=-0.025988`, `delta_accuracy=-0.009704`, `delta_high_error_recall=-0.100000`

## Interpretation

Under the current representation and probe setup, fine/spherical token features do not provide stable added reliability signal beyond `pred_tmag`, `dt`, and `k`. This reinforces the S4/S5 conclusion that the dominant structure of the current error distribution is still explained mainly by magnitude/regime features. It also reinforces the earlier S3b result: the present fine-token representation is not yet a strong source of either residual correction signal or added reliability signal.

## Decision

No S8c follow-up was justified. No final route test was run. S8 does not replace S5, and [S5_clean_tmag_calibration_policy.json](/home/dovetao/graduation_design_demo/checkpoints/S5_clean_tmag_calibration_policy.json) remains the final clean candidate.

## Thesis Wording

We further tested whether the spherical/fine token features could be used as a reliability estimator rather than a residual pose regressor. In a staged two-fold diagnostic, token-only and token-plus-regime probes failed to outperform a simple regime-only baseline using `pred_tmag`, `dt`, and `k`. This indicates that the current token representation does not provide stable additional reliability information beyond magnitude/regime features, and supports using S5 as the final clean candidate while leaving token reliability modeling as future work.

## Caveat

S8 was a staged diagnostic rather than a full final route test, and sample caps were used because of runtime budget constraints. The conclusion should therefore be read as a statement about the current token representation and the current lightweight probe setup, not as a universal impossibility result for token-based reliability modeling.
