# S5D5 Pairwise-to-Dense tdir Gap Audit

## Executive summary

Final classification: `S5D5_FINAL_S5_PAIRWISE_ARTIFACT_UNAVAILABLE`

## Question

historical pairwise/local-frame tdir around 20 deg vs dense TUM-derived tdir around 91 deg.

## Historical source review

| source | value summary | sequence | final S5 locked candidate | classification |
|---|---|---|---|---|
| S11_B | {'tdir': 38.42739820480347, 'tdir_abs': 38.42739820480347} | scene01_seq01 | False | OLD_TDIR_PAIRWISE_LOCAL_FRAME |
| S11_C | {'tdir': 37.726716288547806, 'tdir_abs': 19.357324309034116} | scene01_seq01 | False | OLD_TDIR_PAIRWISE_LOCAL_FRAME |
| S16 | {'current_model_mean_tdir_err': 31.54245074814257} | mixed/feasibility | False | OLD_TDIR_PAIRWISE_LOCAL_FRAME |

## Final S5 pairwise artifact availability

- available: `False`
- paths: `[]`

## Dense-derived bridging metrics

- dense_relative k1: `{'num_pairs': 453, 'num_valid_pairs': 453, 'tdir_mean_deg': 91.39241866186, 'tdir_median_deg': 95.67985920477633, 'tdir_p90_deg': 139.55126037861987, 'tmag_mean_ratio': 27.174141849809555}`
- dense_relative_tdir_abs k1: `{'num_pairs': 453, 'num_valid_pairs': 453, 'tdir_mean_deg': 57.17123032045375, 'tdir_median_deg': 60.61637277070892, 'tdir_p90_deg': 83.40551252427645, 'tmag_mean_ratio': 27.174141849809555}`
- world_delta k1: `{'num_pairs': 453, 'num_valid_pairs': 453, 'tdir_mean_deg': 85.63331890482517, 'tdir_median_deg': 85.32171669386678, 'tdir_p90_deg': 137.98472601483408, 'tmag_mean_ratio': 27.174141849809555}`
- world_delta_tdir_abs k1: `{'num_pairs': 453, 'num_valid_pairs': 453, 'tdir_mean_deg': 56.87800291046734, 'tdir_median_deg': 58.8645777631273, 'tdir_p90_deg': 84.39159802875092, 'tmag_mean_ratio': 27.174141849809555}`
- k_step: `{'k=1': {'num_pairs': 453, 'num_valid_pairs': 453, 'tdir_mean_deg': 91.39241866186, 'tdir_median_deg': 95.67985920477633, 'tdir_p90_deg': 139.55126037861987, 'tmag_mean_ratio': 27.174141849809555}, 'k=2': {'num_pairs': 452, 'num_valid_pairs': 452, 'tdir_mean_deg': 91.55914446916339, 'tdir_median_deg': 95.16242519374079, 'tdir_p90_deg': 141.19612184971794, 'tmag_mean_ratio': 25.261488640708137}, 'k=5': {'num_pairs': 449, 'num_valid_pairs': 449, 'tdir_mean_deg': 90.97930803125395, 'tdir_median_deg': 94.55024607998017, 'tdir_p90_deg': 141.03854705068608, 'tmag_mean_ratio': 23.085056095778462}, 'k=10': {'num_pairs': 444, 'num_valid_pairs': 444, 'tdir_mean_deg': 90.99395977826636, 'tdir_median_deg': 91.70821737702843, 'tdir_p90_deg': 142.03294291751206, 'tmag_mean_ratio': 17.495879602629508}, 'k=20': {'num_pairs': 434, 'num_valid_pairs': 434, 'tdir_mean_deg': 92.79939024525902, 'tdir_median_deg': 90.28257285315706, 'tdir_p90_deg': 127.29212497795513, 'tmag_mean_ratio': 12.36826794068604}}`
- thresholded: `{'median_gt_step_0p05': {'num_pairs': 453, 'num_valid_pairs': 453, 'tdir_mean_deg': 91.39241866186, 'tdir_median_deg': 95.67985920477633, 'tdir_p90_deg': 139.55126037861987, 'tmag_mean_ratio': 27.174141849809555}, 'median_gt_step_0p10': {'num_pairs': 453, 'num_valid_pairs': 452, 'tdir_mean_deg': 91.40147895585116, 'tdir_median_deg': 96.02873207056558, 'tdir_p90_deg': 139.56335447395884, 'tmag_mean_ratio': 26.49197705706998}, 'every_2nd_pair': {'num_pairs': 227, 'num_valid_pairs': 227, 'tdir_mean_deg': 92.04163658060574, 'tdir_median_deg': 96.41882139519305, 'tdir_p90_deg': 139.50288399726406, 'tmag_mean_ratio': 26.20117060611419}, 'every_5th_pair': {'num_pairs': 91, 'num_valid_pairs': 91, 'tdir_mean_deg': 91.31777300487447, 'tdir_median_deg': 98.1863899786247, 'tdir_p90_deg': 133.81719515148973, 'tmag_mean_ratio': 25.13940086174295}}`

## Pairwise-to-dense gap interpretation

Historical ~20deg tdir is real but comes from S11/S16 pairwise/local-frame heldout diagnostics (different candidate/sequence/protocol). Dense S5 export on scene01_seq03 remains ~91deg even with abs/sign-invariant, k-step, and thresholded variants; direct final-S5 pairwise-to-dense replay is blocked by missing explicit final pairwise vectors.

## Caveats

- S5D5 is diagnostic only.
- S5D5 does not modify predictions.
- S5D5 does not replace official S5 locked result.
- S5 locked metrics/policy were not changed.
- Historical tdir≈20 may belong to different candidate/sequence/protocol.
