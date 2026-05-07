# S13 Practical Usability Gap Analysis

## Executive summary

- final classification: `PRACTICAL-GAP-QUANTIFIED`
- final clean candidate remains S5: `True`
- primary bottleneck: `R-TDIR-COUPLED-LIMITED`
- recommended next major direction: `Multi-frame / pose-graph optimization`

## Baseline gate

- passed: `True`
- load_missing / load_unexpected: `14 / 0`
- locked metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## S5 best-clean vs practical-ready distinction

- S5 is still the best clean deployable candidate under current project constraints.
- S13 asks whether this clean candidate is practical-ready under progressively stricter usability thresholds.

## Trajectory-level usability thresholds

- global threshold checks below are anchored to the locked S5 metrics.
- supplementary all-chain diagnostic from this script: ATE=`7.522779`, drift=`4.201069`, path_ratio=`4.944008`
- `lenient`: ATE<5.0, drift<1.0, path_ratio in [0.9,1.05], pass=`False`
  gaps: ate=`2.352288`, drift=`0.327343`, path_low=`-0.032379`, path_high=`-0.117621`
- `moderate`: ATE<3.0, drift<0.5, path_ratio in [0.95,1.05], pass=`False`
  gaps: ate=`4.352288`, drift=`0.827343`, path_low=`0.017621`, path_high=`-0.117621`
- `strict`: ATE<2.0, drift<0.3, path_ratio in [0.97,1.03], pass=`False`
  gaps: ate=`5.352288`, drift=`1.027343`, path_low=`0.037621`, path_high=`-0.097621`

## Per-scene / per-chain usability pass rate

- `lenient` chain pass rate: `0.000000`
- `moderate` chain pass rate: `0.000000`
- `strict` chain pass rate: `0.000000`
- worst chains by drift:
  - `scene01/seq03` chain `15`: drift=`6.104248`, ATE=`0.171752`, path_ratio=`5.537505`
  - `scene01/seq03` chain `16`: drift=`6.011870`, ATE=`0.213046`, path_ratio=`6.918752`
  - `scene01/seq03` chain `6`: drift=`5.050242`, ATE=`0.107488`, path_ratio=`5.992971`
  - `scene01/seq03` chain `17`: drift=`5.013847`, ATE=`0.134571`, path_ratio=`5.724691`
  - `scene01/seq03` chain `10`: drift=`4.852346`, ATE=`0.103920`, path_ratio=`5.822346`

## Rotation diagnostics

- mean / median / p90 / max rot error = `20.715353` / `20.433789` / `21.371863` / `21.717051` deg
- per-scene mean rot error: `{'scene01/seq03': 20.71535270992738}`
- worst dt×k rot regimes: `{'<0.1|k=1': 20.896876811480933, '[0.1,0.3)|k=1': 20.510704330713324, '[0.5,1.0)|k=1': 21.192149875009086, '[0.3,0.5)|k=1': 20.604867913851066}`

## Translation-direction diagnostics

- mean / median / p90 / max tdir error = `72.349483` / `102.517940` / `110.918120` / `141.610521` deg
- mean cosine similarity = `0.226300`
- per-scene mean tdir error: `{'scene01/seq03': 72.34948284989224}`
- worst dt×k tdir regimes: `{'<0.1|k=1': 60.26758872213477, '[0.1,0.3)|k=1': 90.69584271049251, '[0.5,1.0)|k=1': 26.488513252978503, '[0.3,0.5)|k=1': 72.81630591581832}`

## Translation-magnitude diagnostics

- mean / median / p90 / max log tmag error = `0.899289` / `0.793141` / `1.742507` / `1.985574`
- mean speed-normalized log error = `0.899289`
- per-scene mean log tmag error: `{'scene01/seq03': 0.8992888241641145}`
- worst dt×k tmag regimes: `{'<0.1|k=1': 1.337356399465439, '[0.1,0.3)|k=1': 0.325057406556135, '[0.5,1.0)|k=1': 1.6395977875548085, '[0.3,0.5)|k=1': 0.9272010075179}`

## Coupling / oracle diagnostics

- `oracle_R`: ATE=`10.590301`, drift=`1.516962`, path_ratio=`0.934986`, delta_ATE=`-3.238013`
- `oracle_tdir`: ATE=`7.658319`, drift=`1.375470`, path_ratio=`0.934985`, delta_ATE=`-0.306031`
- `oracle_R_tdir`: ATE=`0.911024`, drift=`0.304305`, path_ratio=`0.934986`, delta_ATE=`6.441264`
- `oracle_tmag`: ATE=`7.214477`, drift=`1.262237`, path_ratio=`0.898234`, delta_ATE=`0.137811`
- `oracle_all`: ATE=`0.000000`, drift=`0.000000`, path_ratio=`1.000000`, delta_ATE=`7.352288`
- `oracle_R`, `oracle_tdir`, and `oracle_R_tdir` are reused from the established S2c/S2c2 coupling diagnostics; `oracle_tmag` uses the stabilized oracle-only tmag diagnostic from the final results table.
- coupling interpretation: `Replacing R or tdir alone does not solve the practical gap, but replacing them together collapses ATE/drift dramatically; this points to coupled R-tdir error rather than isolated tmag scale as the primary bottleneck.`

## Regime-level practical failure analysis

- overall pair practical failure proxy rate = `1.000000`
- proxy definition: `rot_err>=10deg OR tdir_err>=20deg OR log_tmag_err>=0.30`
- pred_tmag bucket fail rate: `{'pred_q1': 1.0, 'pred_q0': 1.0, 'pred_q2': 1.0, 'pred_q3': 1.0}`
- gt_tmag bucket fail rate: `{'gt_q0': 1.0, 'gt_q1': 1.0, 'gt_q2': 1.0, 'gt_q3': 1.0}`
- dt bucket fail rate: `{'<0.1': 1.0, '[0.1,0.3)': 1.0, '[0.5,1.0)': 1.0, '[0.3,0.5)': 1.0}`
- k bucket fail rate: `{'k=1': 1.0}`
- dt×k bucket fail rate: `{'<0.1|k=1': 1.0, '[0.1,0.3)|k=1': 1.0, '[0.5,1.0)|k=1': 1.0, '[0.3,0.5)|k=1': 1.0}`
- high_pred failure contribution = `0.106061`
- high_risk failure contribution = `0.000000`

## Chain accumulation analysis

- corr(mean_pair_pos_err, drift) = `-0.840904`
- corr(mean_rot_error, drift) = `0.044300`
- corr(mean_tdir_error, drift) = `-0.076035`
- worst chain summary = `{'scene_seq': 'scene01/seq03', 'chain_id': 15, 'drift': 6.104248365570641, 'ate': 0.17175178656514017, 'path_ratio': 5.537505086781352, 'mean_rot_error': 20.41643398349136, 'mean_tdir_error': 100.85522578214515, 'mean_tmag_log_error': 1.7115440884781765, 'chain_sum_tmag_ratio': 5.537505277832099}`
- S10 closure note: smoother did not solve this because practical gap is not explained by chain-sum tmag alone.

## Post-S5 optimization closure

- S8 token reliability did not add value beyond regime features.
- S9 regime-only router had diagnostic signal but no stable clean gain.
- S10 chain-level smoother had no stable chain-smoother gain.
- S11 tmag consistency training had weak proxy signal but no stable clean gain.
- S12 regime-balanced sampling improved some high-risk proxy metrics but no clean-eligible path_ratio evidence.
- Taken together, post-S5 small fixes did not close the practical usability gap.

## Final bottleneck classification

- primary: `R-TDIR-COUPLED-LIMITED`
- secondary: `['CHAIN-ACCUMULATION-LIMITED', 'ROTATION-LIMITED', 'TDIR-LIMITED', 'TMAG-SCALE-LIMITED']`
- rationale: `oracle_R_tdir is dramatically stronger than oracle_R or oracle_tdir alone; a small number of bad chains dominate drift`

## Recommended next major direction

- `Multi-frame / pose-graph optimization`
- rationale: `The main gap is not a small pairwise tmag tweak. Oracle evidence and chain accumulation both point toward coupled pose errors that need local-window or trajectory-level joint optimization rather than another pairwise post-processing fix.`

## Final classification

- `PRACTICAL-GAP-QUANTIFIED`
