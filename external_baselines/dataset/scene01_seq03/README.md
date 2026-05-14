# External Baseline Dataset Export: scene01/seq03

This export is intended for protocol-compatible external SLAM/VO baseline evaluation.
It does not change the final S5 candidate, the locked metrics, the historical split, or the eval convention.

## Contents
- `image_list.txt`: source panorama image paths for each timestamp.
- `timestamps.txt`: ordered timestamps for the historical final test sequence.
- `camera.yaml`: equirectangular camera metadata and caveat that pinhole intrinsics are unavailable.
- `groundtruth_tum.txt`: ground-truth trajectory in TUM pose format.
- `metadata.json`: evaluation metadata and protocol caveats.

## Notes
- Images are not copied to avoid heavy repository artifacts.
- Numeric comparison is valid only after an external baseline is run on this exact sequence under this protocol.
- Published results from different datasets must not be compared directly against S5.
