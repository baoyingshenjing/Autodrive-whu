"""Small dependency-free (apart from matplotlib) LiDAR BEV prediction renderer."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from course_common import CLASSES, POINT_CLOUD_RANGE, save_json


def _as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def _box_lines(box):
    x, y, _, dx, dy, _, yaw = box[:7]
    corners = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2],
                        [-dx / 2, -dy / 2], [-dx / 2, dy / 2],
                        [dx / 2, dy / 2]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
    return corners @ rotation.T + np.array([x, y])


def render_predictions(prediction_file: Path, output_dir: Path, count: int,
                       score_threshold: float, data_root: Path) -> list[dict]:
    with prediction_file.open('rb') as f:
        samples = pickle.load(f)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for index, sample in enumerate(samples):
        if len(rendered) >= count:
            break
        if isinstance(sample, dict):
            metainfo = sample
            pred = sample.get('pred_instances_3d')
        else:
            metainfo = getattr(sample, 'metainfo', {})
            pred = getattr(sample, 'pred_instances_3d', None)
        if pred is None:
            continue
        if isinstance(pred, dict):
            boxes = _as_numpy(pred['bboxes_3d'].tensor)
            scores = _as_numpy(pred['scores_3d'])
            labels = _as_numpy(pred['labels_3d']).astype(int)
        else:
            boxes = _as_numpy(pred.bboxes_3d.tensor)
            scores = _as_numpy(pred.scores_3d)
            labels = _as_numpy(pred.labels_3d).astype(int)
        keep = scores >= score_threshold
        lidar_path = metainfo.get('lidar_path', '')
        point_file = Path(lidar_path)
        if not point_file.is_file():
            point_file = data_root / 'samples' / 'LIDAR_TOP' / point_file.name
        if not point_file.is_file():
            continue
        points = np.fromfile(point_file, dtype=np.float32).reshape(-1, 5)
        in_range = ((points[:, 0] >= POINT_CLOUD_RANGE[0]) &
                    (points[:, 0] <= POINT_CLOUD_RANGE[3]) &
                    (points[:, 1] >= POINT_CLOUD_RANGE[1]) &
                    (points[:, 1] <= POINT_CLOUD_RANGE[4]))
        points = points[in_range]
        if len(points) > 50000:
            points = points[::max(1, len(points) // 50000)]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.scatter(points[:, 0], points[:, 1], s=0.15, c='0.35', alpha=0.45)
        for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
            line = _box_lines(box)
            color = plt.cm.tab10(int(label) % 10)
            ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.0)
            ax.text(box[0], box[1], f'{CLASSES[int(label)]}:{score:.2f}',
                    fontsize=5, color='black')
        ax.set(xlim=POINT_CLOUD_RANGE[0::3], ylim=POINT_CLOUD_RANGE[1::3],
               xlabel='x / m', ylabel='y / m', title='LiDAR BEV: predicted boxes')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.2)
        for text_artist in fig.findobj(match=matplotlib.text.Text):
            text_artist.set_color('black')
        output = output_dir / f'bev_{len(rendered) + 1:02d}.png'
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
        rendered.append(dict(file=str(output), source_lidar=str(point_file),
                             detections=int(keep.sum()), sample_index=index,
                             score_threshold=score_threshold))
    save_json(output_dir / 'visualization_summary.json',
              dict(rendered=rendered, score_threshold=score_threshold))
    return rendered
