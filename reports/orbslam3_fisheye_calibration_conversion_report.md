# ORB1b Calibration Conversion Feasibility Audit

## Executive summary

- Experiment: `ORB1b_calibration_conversion_feasibility_audit`
- Final classification: `ORB1B_READY_FITTED_KB8`
- Camera index: `0`
- Ready for ORB-SLAM3 YAML: `True`

This audit does not run ORB-SLAM3 and does not produce baseline metrics.

## Source calibration model

- cam_infos path: `/home/dovetao/graduation_design_demo/data/PanoramaView/scene01/seq03/cam_infos.txt`
- model: `dataset_polynomial_fisheye`
- image size: `[640, 640]`
- distortion center: `[320.7997, 321.6138]`
- mapping coeffs: `[167.6389, -0.002, 0.0, -0.0]`
- stretch matrix: `[1.0, 0.0, 0.0, 1.0]`

The dataset provides a polynomial fisheye projection model. This is not the same thing as native ORB-SLAM3 KannalaBrandt8 calibration.

## Why direct parameter copying is invalid

`mapping_coeffs` describe the dataset polynomial projection. They must not be copied directly into ORB-SLAM3 `k1/k2/k3/k4`; doing so would fabricate a calibration model.

## Conversion / fitting method

- target model: `ORB_SLAM3_KannalaBrandt8`
- method: `fitted_approximation`
- conversion classification: `FITTED_KB8_APPROXIMATION_SUPPORTED`
- thresholds: `{'mean_max': 1.0, 'p90_max': 2.0, 'max_max': 5.0}`

The fitted path samples the polynomial model over the image domain, derives angle/radius pairs under the documented audit assumption, and fits the KB8 radial polynomial by least squares.

## Reprojection error table

| metric | px |
| --- | ---: |
| mean | 0.072139 |
| median | 0.069148 |
| p90 | 0.124491 |
| max | 0.936757 |

## YAML generation status

- path: `/home/dovetao/graduation_design_demo/external_baselines/config/orbslam3_fisheye_cam0_scene01_seq03.yaml`
- written: `True`
- status: `official_candidate`

## Final classification

`ORB1B_READY_FITTED_KB8`

## Next-stage recommendation

If the classification is `ORB1B_READY_FITTED_KB8`, cam0 may proceed to ORB1c as a fitted KB8 compatibility candidate. The YAML must remain labeled as converted/fitted calibration. If unsupported, use a derived undistorted pinhole route instead of claiming a direct raw fisheye baseline.

## Notes

- Source calibration is dataset polynomial fisheye, not native ORB-SLAM3 KannalaBrandt8.
- Never copy mapping_coeffs directly into k1/k2/k3/k4.
- Direct KB8 conversion is not claimed; this audit uses sampled polynomial projection fitting.
- The generated YAML is only an ORB-SLAM3 compatibility candidate, not factory KannalaBrandt8 calibration.
