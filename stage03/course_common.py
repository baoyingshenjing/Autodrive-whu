"""Shared, deliberately small helpers for the Windows PointPillars course.

The course never uses nuScenes sweeps.  Keep that rule here instead of
relying on students to remember to edit an upstream MMDetection3D config.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mmengine.config import Config


COURSE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = COURSE_ROOT.parent
DEFAULT_DATA_ROOT = COURSE_ROOT / 'data' / 'nuscenes'
CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
# The 8-GB classroom route intentionally uses a compact 50 m x 50 m BEV.
# It is a teaching configuration, not a claim of the 100 m official benchmark.
POINT_CLOUD_RANGE = [-25, -25, -5, 25, 25, 3]
VOXEL_SIZE = [0.25, 0.25, 8]
BEV_SHAPE = [200, 200]


def path_from_arg(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default.resolve()


def data_root_from_arg(value: str | None) -> Path:
    env_value = os.getenv('NUSCENES_ROOT')
    return path_from_arg(value or env_value, DEFAULT_DATA_ROOT)


def course_info_paths() -> tuple[Path, Path, Path]:
    info_dir = COURSE_ROOT / 'data_infos'
    return info_dir / 'train.pkl', info_dir / 'val.pkl', info_dir / 'manifest.json'


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=json_default), encoding='utf-8')


def load_manifest() -> dict[str, Any]:
    _, _, manifest = course_info_paths()
    if not manifest.is_file():
        raise FileNotFoundError(
            f'索引清单不存在：{manifest}\n请先运行：python preprocess.py')
    return json.loads(manifest.read_text(encoding='utf-8'))


def assert_manifest_is_single_frame(manifest: dict[str, Any]) -> None:
    if manifest.get('input_mode') != 'single_keyframe_xyz_intensity':
        raise RuntimeError('清单不是单关键帧输入，拒绝继续。请重新运行 preprocess.py。')
    train_tokens = set(manifest.get('train_sample_tokens', []))
    val_tokens = set(manifest.get('val_sample_tokens', []))
    overlap = train_tokens & val_tokens
    if overlap:
        raise RuntimeError(f'训练/验证样本重叠 {len(overlap)} 个，拒绝继续。')


def single_frame_pipeline(train: bool) -> list[dict[str, Any]]:
    """Return a pipeline that contains no LoadPointsFromMultiSweeps stage."""
    common = [
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5,
             use_dim=4, backend_args=None),
    ]
    if train:
        common += [
            dict(type='LoadAnnotations3D', with_bbox_3d=True,
                 with_label_3d=True),
            dict(type='GlobalRotScaleTrans', rot_range=[-0.3925, 0.3925],
                 scale_ratio_range=[0.95, 1.05], translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(type='PointsRangeFilter', point_cloud_range=POINT_CLOUD_RANGE),
            dict(type='ObjectRangeFilter', point_cloud_range=POINT_CLOUD_RANGE),
            dict(type='ObjectNameFilter', classes=CLASSES),
            dict(type='PointShuffle'),
            dict(type='Pack3DDetInputs',
                 keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
        ]
    else:
        common += [
            dict(type='GlobalRotScaleTrans', rot_range=[0, 0],
                 scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type='PointsRangeFilter', point_cloud_range=POINT_CLOUD_RANGE),
            dict(type='Pack3DDetInputs', keys=['points']),
        ]
    return common


def _dataset_cfg(ann_file: Path, data_root: Path, pipeline: list[dict[str, Any]],
                 test_mode: bool) -> dict[str, Any]:
    # ann_file is absolute: os.path.join(data_root, ann_file) preserves it on Windows.
    return dict(
        type='NuScenesDataset', data_root=str(data_root), ann_file=str(ann_file),
        data_prefix=dict(pts='samples/LIDAR_TOP'),
        pipeline=pipeline, metainfo=dict(classes=CLASSES, version='v1.0-mini'),
        modality=dict(use_lidar=True, use_camera=False), box_type_3d='LiDAR',
        test_mode=test_mode, backend_args=None)


def build_course_config(*, data_root: Path, work_dir: Path, max_epochs: int,
                        load_from: str | None = None, resume: bool = False,
                        amp: bool = False) -> Config:
    """Adapt the downloaded official config to an 8-GB, one-keyframe lesson."""
    template = COURSE_ROOT / 'checkpoints' / 'official_pointpillars_config.py'
    train_info, val_info, _ = course_info_paths()
    if not template.is_file():
        raise FileNotFoundError('缺少官方配置；请先运行 python download_weights.py')
    if not train_info.is_file() or not val_info.is_file():
        raise FileNotFoundError('缺少课程索引；请先运行 python preprocess.py')
    cfg = Config.fromfile(template)
    cfg.work_dir = str(work_dir)
    cfg.load_from = load_from
    cfg.resume = resume
    # RTX 3070 Laptop (8 GB) route: compact BEV + capped occupied pillars.
    cfg.model.data_preprocessor.voxel_layer.max_voxels = (6000, 8000)
    cfg.model.data_preprocessor.voxel_layer.point_cloud_range = POINT_CLOUD_RANGE
    cfg.model.data_preprocessor.voxel_layer.voxel_size = VOXEL_SIZE
    cfg.model.pts_voxel_encoder.point_cloud_range = POINT_CLOUD_RANGE
    cfg.model.pts_voxel_encoder.voxel_size = VOXEL_SIZE
    cfg.model.pts_middle_encoder.output_shape = BEV_SHAPE
    for anchor_range in cfg.model.pts_bbox_head.anchor_generator.ranges:
        anchor_range[0] = anchor_range[1] = -24.6
        anchor_range[3] = anchor_range[4] = 24.6
    cfg.train_dataloader = dict(
        batch_size=1, num_workers=0, persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=_dataset_cfg(train_info, data_root, single_frame_pipeline(True), False))
    eval_dataset = _dataset_cfg(val_info, data_root, single_frame_pipeline(False), True)
    cfg.val_dataloader = dict(
        batch_size=1, num_workers=0, persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=False), dataset=eval_dataset)
    cfg.test_dataloader = dict(
        batch_size=1, num_workers=0, persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=_dataset_cfg(val_info, data_root, single_frame_pipeline(False), True))
    evaluator = dict(type='NuScenesMetric', data_root=str(data_root),
                     ann_file=str(val_info), metric='bbox',
                     modality=dict(use_lidar=True, use_camera=False),
                     backend_args=None)
    cfg.val_evaluator = evaluator.copy()
    cfg.test_evaluator = evaluator.copy()
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                         val_interval=max_epochs)
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=1,
                                        max_keep_ckpts=-1, save_last=True)
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=10)
    cfg.optim_wrapper = dict(type='OptimWrapper',
                             optimizer=dict(type='AdamW', lr=0.0001,
                                            weight_decay=0.01),
                             clip_grad=dict(max_norm=35, norm_type=2))
    cfg.param_scheduler = []
    cfg.randomness = dict(seed=20261012, deterministic=False)
    cfg.env_cfg.mp_cfg = dict(mp_start_method='spawn', opencv_num_threads=0)
    cfg.visualizer = dict(type='Det3DLocalVisualizer', vis_backends=[])
    if amp:
        # MMDet3D 1.4's naiveSyncBN asserts float32.  On this single-GPU
        # course route ordinary BatchNorm is equivalent for synchronization
        # purposes and allows autocast to execute.
        cfg.model.pts_voxel_encoder.norm_cfg.type = 'BN1d'
        cfg.model.pts_backbone.norm_cfg.type = 'BN2d'
        cfg.model.pts_neck.norm_cfg.type = 'BN2d'
        cfg.optim_wrapper = dict(type='AmpOptimWrapper',
                                 optimizer=dict(type='AdamW', lr=0.0001,
                                                weight_decay=0.01),
                                 clip_grad=dict(max_norm=35, norm_type=2))
    return cfg


def write_resolved_config(cfg: Config, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / 'resolved_config.py'
    cfg.dump(str(path))
    return path


def patch_mmengine_windows_collect_env() -> None:
    """Prevent an informational compiler probe from aborting Chinese Windows.

    MMEngine 0.10.7 decodes the compiler banner with the active GBK code page.
    Some installed compiler banners are UTF-8.  This affects logging only, not
    CUDA execution; preserve normal reporting whenever decoding succeeds.
    """
    from mmengine.runner import runner as runner_module
    original = runner_module.collect_env

    def safe_collect_env():
        try:
            return original()
        except UnicodeDecodeError as exc:
            return {'MMEngine environment probe':
                    f'skipped due to Windows compiler encoding: {exc}'}
    runner_module.collect_env = safe_collect_env
