# MAINT8 Recommended Project Structure

## Goal

This is a future-state structure proposal only. No files are moved by MAINT8. The purpose is to separate active code, archival code, reports, and local-only artifacts so later maintenance work can happen with lower risk.

## Proposed Layout

```text
project/
  configs/
    final/
    train360/
    struct360/
    seq360/
    baseline/
    legacy/
  datasets/
    dset2c_manifest_dataset.py
    dset2c_sequence_clip_dataset.py
    legacy/
  models/
    core/
    train360/
    struct360/
    seq360/
    legacy/
  tools/
    train/
    eval/
    baseline/
    maintenance/
    legacy/
  reports/
    final360/
    train360/
    struct360/
    seq360/
    base360/
    maint/
    archive/
  external_baselines/
    results/
  checkpoints/        local-only
  data/               local-only
  reports_curated/    reference archive
```

## Mapping Guidance

- `configs/final/`: FINAL360I and other final-selection configs.
- `configs/train360/`: TRAIN360C/D/H and related training configs.
- `configs/struct360/`: STRUCT360A/B/C family configs.
- `configs/seq360/`: SEQ360A and any future sequence-level configs.
- `configs/baseline/`: BASE360 or external baseline wrappers.
- `configs/legacy/`: old one-off configs after imports are fully updated.

- `models/core/`: migration target for root-level reusable modules such as `model.py`, `interaction.py`, `erp_sampling.py`, `healpix_utils.py`, `transformer_encoder.py`, `pose_head.py`, and `losses.py`.
- `models/struct360/`: STRUCT360A/B/C family models.
- `models/train360/`: pair-level mainline or training-specific wrappers if they split out from the root stack later.
- `models/seq360/`: sequence-level heads and refiners.

- `tools/train/`: train and fine-tune entrypoints.
- `tools/eval/`: pure evaluation, export, and report-generation entrypoints.
- `tools/baseline/`: BASE360 and external-system wrappers.
- `tools/maintenance/`: inventory, audit, migration, and cleanup helpers.
- `tools/legacy/`: older scripts that are kept for reproducibility but not part of the active thesis path.

- `reports/` should be organized by experiment family rather than chronology.
- `external_baselines/results/` should stay in place as a mixed results area until links, scripts, and downstream documentation are audited.
- `checkpoints/` and `data/` remain local-only storage zones and should not be restructured during MAINT8.

## Why This Structure

- It preserves the actual current mainline while making future migration boundaries explicit.
- It reduces the risk of accidentally treating runtime-critical root files as dead legacy code.
- It creates a place for archival tools without deleting them.
- It acknowledges that `reports_curated/` already acts as a soft archive and should stay separate from live report generation.

## Explicit Non-Goals For MAINT8

- no checkpoint moves
- no raw data moves
- no `external_baselines/results/` moves
- no current tool/config/model renames
- no historical link rewriting
