# Git Branch Remote Sync Audit

## Current branch
- original branch: `optimize/s19-geometry-aware-pretraining-feasibility`
- final current branch after audit actions: `optimize/s19-geometry-aware-pretraining-feasibility`

## Git status before
- `git status --short`: clean

## Local branches
- `main`
- `optimize/s10-chain-pathratio-smoother`
- `optimize/s11-tmag-scale-consistency-training`
- `optimize/s12-regime-balanced-sampling`
- `optimize/s14-local-window-pose-graph`
- `optimize/s15-trajectory-level-training-objective`
- `optimize/s16-stronger-visual-backbone-feasibility`
- `optimize/s17-pose-supervision-dataset-quality-audit`
- `optimize/s19-geometry-aware-pretraining-feasibility`
- `optimize/s2-fine-refinement-on-s1d5`
- `optimize/s3a0-coupled-pose-residual-head`

## Remote origin branches
- `origin/main`
- `origin/optimize/s10-chain-pathratio-smoother`
- `origin/optimize/s11-tmag-scale-consistency-training`
- `origin/optimize/s14-local-window-pose-graph`
- `origin/optimize/s2-fine-refinement-on-s1d5`
- `origin/optimize/s3a0-coupled-pose-residual-head`

## Local branches missing on origin
- `optimize/s12-regime-balanced-sampling`
- `optimize/s15-trajectory-level-training-objective`
- `optimize/s16-stronger-visual-backbone-feasibility`
- `optimize/s17-pose-supervision-dataset-quality-audit`
- `optimize/s19-geometry-aware-pretraining-feasibility`

## Branches with no upstream
- `optimize/s11-tmag-scale-consistency-training`
- `optimize/s12-regime-balanced-sampling`
- `optimize/s14-local-window-pose-graph`
- `optimize/s15-trajectory-level-training-objective`
- `optimize/s16-stronger-visual-backbone-feasibility`
- `optimize/s17-pose-supervision-dataset-quality-audit`
- `optimize/s19-geometry-aware-pretraining-feasibility`

## Ahead/behind audit
- `main`: upstream=`origin/main` ahead=`0` behind=`0`
- `optimize/s10-chain-pathratio-smoother`: upstream=`origin/optimize/s10-chain-pathratio-smoother` ahead=`0` behind=`0`
- `optimize/s11-tmag-scale-consistency-training`: no upstream
- `optimize/s12-regime-balanced-sampling`: no upstream
- `optimize/s14-local-window-pose-graph`: no upstream
- `optimize/s15-trajectory-level-training-objective`: no upstream
- `optimize/s16-stronger-visual-backbone-feasibility`: no upstream
- `optimize/s17-pose-supervision-dataset-quality-audit`: no upstream
- `optimize/s19-geometry-aware-pretraining-feasibility`: no upstream
- `optimize/s2-fine-refinement-on-s1d5`: upstream=`origin/optimize/s2-fine-refinement-on-s1d5` ahead=`0` behind=`0`
- `optimize/s3a0-coupled-pose-residual-head`: upstream=`origin/optimize/s3a0-coupled-pose-residual-head` ahead=`0` behind=`0`

## Branches ahead of upstream
- none

## Branches behind upstream
- none

## Branches diverged from upstream
- none

## Tag check
- local tag `final-s5-clean-candidate-post-s9`: exists
- remote tag `origin/final-s5-clean-candidate-post-s9`: exists

## Actions taken
- ran `git fetch origin --prune`
- attempted safe push for branch without upstream:
  - checked out `optimize/s11-tmag-scale-consistency-training`
  - ran `git push -u origin optimize/s11-tmag-scale-consistency-training`
  - result: failed before branch update because Git could not authenticate to `https://github.com/DoveTao/graduation_design_demo.git`
- error observed:
  - `fatal: could not read Username for 'https://github.com': 没有那个设备或地址`
- because the failure is authentication-related rather than branch-state-related, no further push attempts were executed for the remaining no-upstream branches.
- returned to original branch `optimize/s19-geometry-aware-pretraining-feasibility`

## Manual follow-up required
- configure working GitHub credentials for the HTTPS `origin` remote or switch `origin` to an authenticated SSH remote
- after authentication is fixed, the remaining safe push candidates are:
  - no-upstream branches:
    - `optimize/s11-tmag-scale-consistency-training`
    - `optimize/s12-regime-balanced-sampling`
    - `optimize/s14-local-window-pose-graph`
    - `optimize/s15-trajectory-level-training-objective`
    - `optimize/s16-stronger-visual-backbone-feasibility`
    - `optimize/s17-pose-supervision-dataset-quality-audit`
    - `optimize/s19-geometry-aware-pretraining-feasibility`
  - existing-upstream branches with ahead>0:
    - none
  - behind/diverged branches requiring manual review:
    - none

## Git status after
- `git status --short`: clean after sync actions before writing this audit report
