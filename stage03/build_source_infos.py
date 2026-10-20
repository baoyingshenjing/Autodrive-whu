"""Build MMDet3D v1.4 LiDAR-only infos from genuine nuScenes mini metadata.

Uses the same center, length/width/height, yaw, velocity and category conventions
as the bundled official nuscenes_converter.py and update_infos_to_v2.py.
No camera images or historical sweeps are inputs.
"""
import argparse
import ast
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parent
CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']


def matrix(record):
    result = np.eye(4)
    result[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    result[:3, 3] = record['translation']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(ROOT / 'data' / 'nuscenes'))
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    mapping_file = ROOT / 'third_party' / 'mmdetection3d-1.4.0' / 'mmdet3d' / 'datasets' / 'convert_utils.py'
    tree = ast.parse(mapping_file.read_text(encoding='utf-8'))
    mapping = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'NuScenesNameMapping' for t in node.targets))
    nusc = NuScenes(version='v1.0-mini', dataroot=str(data_root), verbose=False)
    splits = {'train': set(mini_train), 'val': set(mini_val)}
    rows = {key: [] for key in splits}
    counts = {key: Counter() for key in splits}
    max_center_error = 0.0
    point_counts = []
    for sample in sorted(nusc.sample, key=lambda x: (x['timestamp'], x['token'])):
        scene = nusc.get('scene', sample['scene_token'])['name']
        split = next(key for key, names in splits.items() if scene in names)
        token = sample['data']['LIDAR_TOP']
        sd = nusc.get('sample_data', token)
        calibrated = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ego = nusc.get('ego_pose', sd['ego_pose_token'])
        l2e, e2g = matrix(calibrated), matrix(ego)
        filename, boxes, _ = nusc.get_sample_data(token)
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)
        assert np.isfinite(points).all() and len(points) > 0
        point_counts.append(len(points))
        instances = []
        for box in boxes:
            annotation = nusc.get('sample_annotation', box.token)
            name = mapping.get(box.name, box.name)
            label = CLASSES.index(name) if name in CLASSES else -1
            global_velocity = np.r_[nusc.box_velocity(box.token)[:2], 0.0]
            velocity = l2e[:3, :3].T @ e2g[:3, :3].T @ global_velocity
            global_center = (e2g @ l2e @ np.r_[box.center, 1.0])[:3]
            error = float(np.max(np.abs(global_center - annotation['translation'])))
            max_center_error = max(max_center_error, error)
            assert error < 1e-6
            bbox = np.r_[box.center, box.wlh[[1, 0, 2]], box.orientation.yaw_pitch_roll[0]]
            assert np.isfinite(bbox).all() and np.all(bbox[3:6] > 0)
            instances.append(dict(bbox_3d=bbox.tolist(), bbox_label=label,
                                  bbox_label_3d=label, velocity=velocity[:2].tolist(),
                                  num_lidar_pts=annotation['num_lidar_pts'],
                                  num_radar_pts=annotation['num_radar_pts'],
                                  bbox_3d_isvalid=(annotation['num_lidar_pts'] + annotation['num_radar_pts']) > 0))
            if label >= 0 and annotation['num_lidar_pts'] > 0:
                counts[split][name] += 1
        rows[split].append(dict(sample_idx=len(rows[split]), token=sample['token'],
                                timestamp=sample['timestamp'] / 1e6,
                                ego2global=e2g.tolist(), lidar_sweeps=[],
                                lidar_points=dict(num_pts_feats=5, lidar_path=Path(filename).name,
                                                  lidar2ego=l2e.tolist()), instances=instances))
    assert (len(rows['train']), len(rows['val'])) == (323, 81)
    train_tokens = {x['token'] for x in rows['train']}
    val_tokens = {x['token'] for x in rows['val']}
    assert not train_tokens & val_tokens and len(train_tokens | val_tokens) == 404
    metainfo = dict(categories={name: i for i, name in enumerate(CLASSES)},
                    classes=CLASSES, dataset='nuscenes', version='v1.0-mini', info_version='1.1')
    for split, records in rows.items():
        with (data_root / f'nuscenes_infos_{split}.pkl').open('wb') as stream:
            pickle.dump(dict(metainfo=metainfo, data_list=records), stream, protocol=4)
    audit = dict(train_samples=323, val_samples=81, overlap_samples=0,
                 train_scenes=sorted(splits['train']), val_scenes=sorted(splits['val']),
                 max_global_center_roundtrip_error_m=max_center_error,
                 point_count_min=min(point_counts), point_count_max=max(point_counts),
                 classes=CLASSES, lidar_visible_annotation_counts=counts,
                 source='nuScenes SDK boxes in LiDAR coordinates; MMDet3D v1.4 official schema',
                 box_dimensions='center x/y/z, length/width/height, yaw; separate vx/vy',
                 missing_velocity_policy='retain SDK NaN; NuScenesDataset replaces missing velocities with zero')
    (data_root / 'source_infos_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SOURCE_INFOS_OK train=323 val=81 overlap=0', flush=True)
    print('max_global_center_roundtrip_error_m=', max_center_error)


if __name__ == '__main__':
    main()
