# MAINT8 Safe Reorganization Steps

## Step 1: Low Risk, Can Do Immediately

- affected files: new index documents only, such as `PROJECT_STRUCTURE.md`, `models/README.md`, `tools/README.md`, `datasets/README.md`, `reports/README.md`, plus optional `.gitignore` additions for clearly local-only artifacts.
- required import updates: none.
- risk: low. These changes document the current layout without touching runtime paths.
- validation command: `git diff --cached --name-only` and confirm that only docs or `.gitignore` changes are staged.
- rollback plan: revert the added docs or `.gitignore` hunk before merge.

## Step 2: Medium Risk, Requires Import Rewiring

- affected files: `model.py`, `interaction.py`, `erp_sampling.py`, `healpix_utils.py`, `transformer_encoder.py`, `pose_head.py`, `losses.py`, `config.py`, `train_mvp.py`, `dataset_pano_only.py`, and every tool/model that imports them.
- required import updates: move reusable root modules into `models/core/`, move `train_mvp.py` into `tools/legacy/`, move `dataset_pano_only.py` into `datasets/legacy/`, and only move `config.py` after every active import path is rewritten.
- risk: medium to high. The active FINAL360I / STRUCT360B / STRUCT360C path still depends on several root-level modules, so a partial migration would break training and evaluation entrypoints.
- validation command: run `python -m py_compile` or a targeted import smoke test over the active tools, then run at least `tools/train360b_forward_sanity.py` and a no-train config load on the mainline tools.
- rollback plan: keep the migration in a dedicated branch, move modules back to their original locations, and restore the previous imports in one revert commit.

## Step 3: High Risk, Do Not Do Now

- affected files: anything under `checkpoints/`, historical files under `reports/`, historical links inside reports, and anything under `external_baselines/results/`.
- required import updates: potentially many path and documentation rewrites, including scripts that assume fixed result locations.
- risk: high. These paths are part experiment archive, part reproducibility record, and part local-only storage. Moving them now would create link rot and complicate recovery.
- validation command: only after a separate archive migration plan exists; not part of MAINT8.
- rollback plan: avoid doing this step until the repo has stable index files and a migration manifest.

## Recommended Order

1. Finish MAINT8 audit reports and root index documentation.
2. Optionally add family-level README files and `.gitignore` clarifications.
3. Plan a dedicated import migration for root-level runtime modules.
4. Treat checkpoint, raw-data, and historical-results movement as a separate archive project rather than as code cleanup.
