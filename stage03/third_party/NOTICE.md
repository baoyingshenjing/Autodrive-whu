# Third-party source and checkpoint provenance

`mmdetection3d-1.4.0/` contains the teacher-supplied MMDetection3D 1.4.0 source distribution, excluding local build and cache files. The upstream Apache 2.0 license is included in that directory.

Upstream: https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0

`../checkpoints/pointpillars_nuscenes_official.pth` is the supplied OpenMMLab PointPillars checkpoint. Its SHA256 is recorded in `../checkpoints/official_weight.sha256`. The original model identifier is `hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth`.

`../outputs/run1_compact/epoch_1.pth` and `../outputs/run5_compact/epoch_5.pth` are the actual course fine-tuning outputs derived from that checkpoint. This repository does not include raw nuScenes sensor data or original dataset metadata; obtain those through the course or the dataset provider.
