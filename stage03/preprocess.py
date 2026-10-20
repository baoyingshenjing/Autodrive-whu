"""Create verifiable, single-keyframe nuScenes mini train/validation indexes."""
from __future__ import annotations

import argparse
import copy
import pickle
from collections import Counter
from pathlib import Path

from nuscenes.nuscenes import NuScenes

from course_common import (CLASSES, course_info_paths, data_root_from_arg,
                           save_json)


def read_info(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'找不到已有索引：{path}')
    with path.open('rb') as f:
        content = pickle.load(f)
    if not isinstance(content, dict) or 'data_list' not in content:
        raise RuntimeError(f'索引格式错误：{path}')
    return content


def make_single_frame(source: dict, data_root: Path) -> tuple[dict, list[str]]:
    result = dict(metainfo=copy.deepcopy(source.get('metainfo', {})), data_list=[])
    sweeps_seen = 0
    for item in source['data_list']:
        item = copy.deepcopy(item)
        sweeps_seen += len(item.pop('lidar_sweeps', []))
        item.pop('sweeps', None)
        points = item.get('lidar_points', {})
        lidar_name = Path(points.get('lidar_path', '')).name
        if not lidar_name:
            raise RuntimeError('索引中有样本没有 lidar_points.lidar_path')
        lidar_file = data_root / 'samples' / 'LIDAR_TOP' / lidar_name
        if not lidar_file.is_file():
            raise FileNotFoundError(f'关键帧点云缺失：{lidar_file}')
        points['lidar_path'] = lidar_name
        points['num_pts_feats'] = 5
        item['lidar_points'] = points
        result['data_list'].append(item)
    result['metainfo']['classes'] = CLASSES
    result['metainfo']['course_input_mode'] = 'single_keyframe_xyz_intensity'
    return result, [str(sweeps_seen)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=None,
                        help='nuScenes root; defaults to the supplied course data')
    args = parser.parse_args()
    data_root = data_root_from_arg(args.data_root)
    version_dir = data_root / 'v1.0-mini'
    if not version_dir.is_dir():
        raise FileNotFoundError(f'不是可用的 v1.0-mini 数据根目录：{data_root}')
    train_src = read_info(data_root / 'nuscenes_infos_train.pkl')
    val_src = read_info(data_root / 'nuscenes_infos_val.pkl')
    train, train_sweeps = make_single_frame(train_src, data_root)
    val, val_sweeps = make_single_frame(val_src, data_root)
    train_tokens = [x['token'] for x in train['data_list']]
    val_tokens = [x['token'] for x in val['data_list']]
    overlap = set(train_tokens) & set(val_tokens)
    if overlap:
        raise RuntimeError(f'训练/验证样本重叠：{len(overlap)}')
    nusc = NuScenes(version='v1.0-mini', dataroot=str(data_root), verbose=False)
    samples = {x['token']: x for x in nusc.sample}
    scenes = {x['token']: x['name'] for x in nusc.scene}
    def scene_names(tokens):
        return sorted({scenes[samples[token]['scene_token']] for token in tokens})
    out_train, out_val, manifest_path = course_info_paths()
    out_train.parent.mkdir(parents=True, exist_ok=True)
    with out_train.open('wb') as f:
        pickle.dump(train, f, protocol=4)
    with out_val.open('wb') as f:
        pickle.dump(val, f, protocol=4)
    raw_sample_count = len(list((data_root / 'samples' / 'LIDAR_TOP').glob('*.bin')))
    raw_sweep_count = len(list((data_root / 'sweeps' / 'LIDAR_TOP').glob('*.bin')))
    manifest = dict(
        dataset_version='v1.0-mini', data_root=str(data_root),
        input_mode='single_keyframe_xyz_intensity',
        input_dimensions=['x', 'y', 'z', 'intensity'],
        raw_keyframe_lidar_files=raw_sample_count,
        raw_sweep_lidar_files=raw_sweep_count,
        train_samples=len(train_tokens), val_samples=len(val_tokens),
        overlap_samples=len(overlap),
        train_scenes=scene_names(train_tokens), val_scenes=scene_names(val_tokens),
        train_sample_tokens=train_tokens, val_sample_tokens=val_tokens,
        source_info_files=[str(data_root / 'nuscenes_infos_train.pkl'),
                           str(data_root / 'nuscenes_infos_val.pkl')],
        removed_lidar_sweep_entries=dict(train=int(train_sweeps[0]),
                                         val=int(val_sweeps[0])),
        class_count=Counter(instance['bbox_label_3d']
                            for item in train['data_list']
                            for instance in item.get('instances', [])
                            if instance['bbox_label_3d'] >= 0
                            and instance.get('num_lidar_pts', 0) > 0),
        class_count_definition='All training annotations in detection classes with num_lidar_pts > 0; before range filtering')
    save_json(manifest_path, manifest)
    print('PREPROCESS_OK')
    print(f'dataset=v1.0-mini keyframes={raw_sample_count} raw_sweeps={raw_sweep_count}')
    print(f'train={len(train_tokens)} val={len(val_tokens)} overlap={len(overlap)}')
    print('input_mode=single_keyframe_xyz_intensity; LoadPointsFromMultiSweeps is absent')
    print(f'wrote={out_train}\n      {out_val}\n      {manifest_path}')


if __name__ == '__main__':
    main()
