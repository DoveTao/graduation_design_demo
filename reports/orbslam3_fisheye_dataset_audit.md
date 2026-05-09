# ORB1a Raw Fisheye Dataset Audit

## Executive summary

- Experiment: `ORB1a_raw_fisheye_dataset_audit`
- Final classification: `RAW_FISHEYE_READY`
- Candidate ORB-SLAM3 monocular fisheye input: `cam0`
- Ready for next stage: `True`

This audit does not run ORB-SLAM3 and does not produce baseline metrics.

## Dataset root discovered / provided

- Raw root: `/home/dovetao/graduation_design_demo/data/FisheyeView`
- Scene: `scene01`
- Sequence: `seq03`

## scene01/seq03 availability

- scene/seq exists: `True`

## Camera stream inventory

- cam0: 454 images, path `/home/dovetao/graduation_design_demo/data/FisheyeView/scene01/seq03`, pattern `flat_filename_prefix`
- cam1: 454 images, path `/home/dovetao/graduation_design_demo/data/FisheyeView/scene01/seq03`, pattern `flat_filename_prefix`
- cam2: 454 images, path `/home/dovetao/graduation_design_demo/data/FisheyeView/scene01/seq03`, pattern `flat_filename_prefix`
- cam3: 454 images, path `/home/dovetao/graduation_design_demo/data/FisheyeView/scene01/seq03`, pattern `flat_filename_prefix`

## cam_infos.txt parse result

- cam_infos.txt exists: `True`
- parse ok: `True`
- cameras in calibration: `4`
- cam_infos source: `/home/dovetao/graduation_design_demo/data/PanoramaView/scene01/seq03/cam_infos.txt`
- parse status: `ok`

## Image size validation

- cam0: observed=[640, 640], calibration=[640, 640], match=True
- cam1: observed=[640, 640], calibration=[640, 640], match=True
- cam2: observed=[640, 640], calibration=[640, 640], match=True
- cam3: observed=[640, 640], calibration=[640, 640], match=True

## Timestamp / GT alignment result

- raw timestamps parseable: `True`
- raw timestamps: `454`
- exported timestamps: `454`
- alignment status: `aligned_exact`
- groundtruth poses: `454`

## Panorama-side labels / cam info

- panorama scene/seq path: `/home/dovetao/graduation_design_demo/data/PanoramaView/scene01/seq03`
- panorama images: `454`
- labels: `454`
- labels align with exported timestamps: `True`
- panorama images align with exported timestamps: `True`

## Exported baseline dataset

- groundtruth_tum.txt: `True`
- timestamps.txt: `True`
- image_list.txt: `True`
- metadata.json: `True`
- camera.yaml: `True`

## ORB-SLAM3 next-stage feasibility

- candidate camera: `cam0`
- ready for next stage: `True`

- cam0 is a plausible monocular fisheye input for the next stage.

## Notes

- Using provided raw root: /home/dovetao/graduation_design_demo/data/FisheyeView
- Using panorama-side cam_infos.txt: /home/dovetao/graduation_design_demo/data/PanoramaView/scene01/seq03/cam_infos.txt

## Final classification

`RAW_FISHEYE_READY`
