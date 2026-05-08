# S18 Split Redesign And Representativeness Evaluation

## Executive summary

- final classification: `CURRENT-SPLIT-ACCEPTABLE-WITH-CAVEAT`
- S5 remains final clean candidate: `True`
- baseline gate pass: `True`
- classification rationale: `seq03 is not the most isolated sequence in the pairwise representativeness matrices ; S5 remains the best clean policy on all three sequences and the gain is not seq03-only ; only three sequences are available, so acceptance still needs an explicit representativeness caveat`

## Motivation from S17

- S17 concluded `CV-SPLIT-NOT-REPRESENTATIVE` while keeping dataset/supervision/convention audits clean.
- S18 therefore re-audits split representativeness without retraining, without changing S5, and without replacing locked historical metrics.

## Baseline gate

- contract path: `/home/dovetao/graduation_design_demo/checkpoints/S8b_reproduction_contract.json`
- current-architecture load_missing/unexpected accepted: `True` with observed `14 / 0`
- S5 locked metrics preserved: `True`
- final manifest overwritten: `False`

## Sequence-level distribution audit

### scene01/seq01

- pair count: `1507`
- k distribution: `{"1": 155, "2": 218, "3": 252, "5": 299, "10": 314, "20": 269}`
- dt p50/p90: `0.136140` / `1.456281`
- gt_tmag p50/p90: `0.136140` / `1.456281`
- pred_tmag p50/p90: `0.134422` / `0.354097`
- rot target p50/p90: `1.190254` / `10.101294`
- tdir dispersion p50/p90: `22.534107` / `136.573300`
- pair_tmag_sum / seq_path_length: `28.749863`
- high-risk bucket mass: `0.0372`
- high-pred bucket mass q90/q95: `0.0958` / `0.0167`
- image stats brightness/contrast/blur/texture: `0.6393` / `0.1787` / `0.0195` / `0.0055`

### scene01/seq02

- pair count: `1779`
- k distribution: `{"1": 138, "2": 211, "3": 253, "5": 317, "10": 454, "20": 406}`
- dt p50/p90: `0.093029` / `1.243331`
- gt_tmag p50/p90: `0.093029` / `1.243331`
- pred_tmag p50/p90: `0.135590` / `0.321807`
- rot target p50/p90: `2.003302` / `11.476674`
- tdir dispersion p50/p90: `19.042908` / `100.755593`
- pair_tmag_sum / seq_path_length: `28.609816`
- high-risk bucket mass: `0.0315`
- high-pred bucket mass q90/q95: `0.0292` / `0.0167`
- image stats brightness/contrast/blur/texture: `0.6086` / `0.1669` / `0.0194` / `0.0052`

### scene01/seq03

- pair count: `1695`
- k distribution: `{"1": 132, "2": 191, "3": 237, "5": 305, "10": 430, "20": 400}`
- dt p50/p90: `0.094673` / `1.336705`
- gt_tmag p50/p90: `0.094673` / `1.336705`
- pred_tmag p50/p90: `0.147000` / `0.402134`
- rot target p50/p90: `2.627635` / `11.436750`
- tdir dispersion p50/p90: `34.983948` / `95.885449`
- pair_tmag_sum / seq_path_length: `28.611747`
- high-risk bucket mass: `0.0342`
- high-pred bucket mass q90/q95: `0.0625` / `0.0542`
- image stats brightness/contrast/blur/texture: `0.6619` / `0.1261` / `0.0200` / `0.0051`

## Sequence distance matrix

### k_l1

| seq | scene01/seq01 | scene01/seq02 | scene01/seq03 |
| --- | --- | --- | --- |
| scene01/seq01 | 0.0000 | 0.1931 | 0.2056 |
| scene01/seq02 | 0.1931 | 0.0000 | 0.0197 |
| scene01/seq03 | 0.2056 | 0.0197 | 0.0000 |

### gt_tmag_l1

| seq | scene01/seq01 | scene01/seq02 | scene01/seq03 |
| --- | --- | --- | --- |
| scene01/seq01 | 0.0000 | 0.1122 | 0.1190 |
| scene01/seq02 | 0.1122 | 0.0000 | 0.0291 |
| scene01/seq03 | 0.1190 | 0.0291 | 0.0000 |

### pred_tmag_l1

| seq | scene01/seq01 | scene01/seq02 | scene01/seq03 |
| --- | --- | --- | --- |
| scene01/seq01 | 0.0000 | 0.4417 | 0.3583 |
| scene01/seq02 | 0.4417 | 0.0000 | 0.3083 |
| scene01/seq03 | 0.3583 | 0.3083 | 0.0000 |

### dt_k_l1

| seq | scene01/seq01 | scene01/seq02 | scene01/seq03 |
| --- | --- | --- | --- |
| scene01/seq01 | 0.0000 | 0.0732 | 0.0877 |
| scene01/seq02 | 0.0732 | 0.0000 | 0.0251 |
| scene01/seq03 | 0.0877 | 0.0251 | 0.0000 |

### combined_representativeness

| seq | scene01/seq01 | scene01/seq02 | scene01/seq03 |
| --- | --- | --- | --- |
| scene01/seq01 | 0.0000 | 0.2194 | 0.2019 |
| scene01/seq02 | 0.2194 | 0.0000 | 0.1102 |
| scene01/seq03 | 0.2019 | 0.1102 | 0.0000 |

- pooled train(seq01+seq02) vs seq03: `{"k_l1": 0.10107957779226072, "gt_tmag_l1": 0.06630219919314446, "pred_tmag_l1": 0.28333333333333344, "dt_k_l1": 0.051948285117697904}`

## Fixed-policy per-sequence evaluation

| seq | policy | drift | ATE | path_ratio | rot | tdir_abs |
| --- | --- | --- | --- | --- | --- | --- |
| scene01/seq01 | S1d5 | 1.118839 | 6.180546 | 1.006421 | nan | nan |
| scene01/seq01 | S2b | 1.118839 | 6.180546 | 1.006421 | nan | nan |
| scene01/seq01 | S5 | 1.120412 | 6.176209 | 0.988612 | nan | nan |
| scene01/seq02 | S1d5 | 1.608874 | 7.127564 | 1.005017 | nan | nan |
| scene01/seq02 | S2b | 1.608874 | 7.127564 | 1.005017 | nan | nan |
| scene01/seq02 | S5 | 1.606749 | 7.119733 | 1.000050 | nan | nan |
| scene01/seq03 | S1d5 | 1.327402 | 7.352371 | 0.934984 | nan | nan |
| scene01/seq03 | S2b | 1.327402 | 7.352371 | 0.934984 | nan | nan |
| scene01/seq03 | S5 | 1.327343 | 7.352288 | 0.932379 | nan | nan |

## S5/S2b/S1d5 stability across sequences

- scene01/seq01: S5-S2b delta drift=`0.001573`, ATE=`-0.004336`, path_ratio=`-0.017809`, ranking=`['S5', 'S1d5', 'S2b']`
- scene01/seq02: S5-S2b delta drift=`-0.002126`, ATE=`-0.007831`, path_ratio=`-0.004967`, ranking=`['S5', 'S1d5', 'S2b']`
- scene01/seq03: S5-S2b delta drift=`-0.000059`, ATE=`-0.000082`, path_ratio=`-0.002605`, ranking=`['S5', 'S1d5', 'S2b']`

## Alternative CV protocol proposal

### A. current protocol

- definition: `train/CV on seq01+seq02, final test on seq03`
- pros: `historically locked and directly comparable to S5`
- cons: `single-sequence final test; representativeness caveat from S17/S18`
- leakage risk: `low`
- thesis main result fit: `historical-baseline-only`
- future work fit: `limited`

### B. leave-one-sequence-out over seq01/seq02/seq03

- definition: `held out one full sequence at a time; evaluation diagnostic only`
- pros: `sequence-level robustness visibility; no pair-level leakage`
- cons: `still only three folds; cannot replace historical S5 lock`
- leakage risk: `low`
- thesis main result fit: `supplementary`
- future work fit: `strong`

### C. stratified pair-level CV by regime

- definition: `stratify by k / gt_tmag / pred_tmag regimes`
- pros: `better regime balancing in principle`
- cons: `sequence leakage risk; temporal correlation can invalidate optimistic CV`
- leakage risk: `high`
- thesis main result fit: `not recommended`
- future work fit: `research-only with explicit leakage controls`

### D. collect-new-seq protocol

- definition: `keep S5 as historical best clean candidate; expand data, then redesign train/CV/test`
- pros: `best route to representative split; supports stronger thesis claims`
- cons: `requires new data collection effort`
- leakage risk: `low`
- thesis main result fit: `recommended future-work direction`
- future work fit: `strongest`

## Impact on thesis claims

- `S5_clean_tmag_calibration_policy` remains the best clean candidate under the historical protocol.
- Current historical CV/test should be presented with an explicit representativeness caveat rather than as practical-ready evidence.
- Future improvements should prefer leave-one-sequence-out diagnostics and, ideally, additional sequences before promoting stronger claims.

## Final classification

- `CURRENT-SPLIT-ACCEPTABLE-WITH-CAVEAT`
