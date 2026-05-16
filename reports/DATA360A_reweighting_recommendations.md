# DATA360A reweighting recommendations

## Conservative weighting
- Lightly down-weight very_small gt_tmag pairs in the tdir loss.
- Keep full supervision on medium/large gt_tmag pairs.
- Add a small anti-parallel penalty before changing the translation head.

## Aggressive weighting
- Strongly down-weight very_small gt_tmag plus extreme_rotation pairs for tdir supervision.
- Up-weight medium/large gt_tmag pairs with non-extreme rotation for tdir loss.
- Consider explicit observability weighting or a lightweight tdir confidence branch.

## Diagnostic filtering
- Do not filter by default; use filtering only if a tiny subset is clearly geometry-unobservable or corrupted.
- If filtering is used, report the exact scene/sequence subset and proportion removed.

## Recommended next experiment
- `proceed_to_FINAL360J_loss_reweight`

## Rationale
- worst tmag bucket: `small`
- worst rotation bucket: `large_rotation`
- anti-parallel tmag concentration: `very_large`
- anti-parallel rotation concentration: `small_rotation`
- SEQ360B scale/path only: `True`
