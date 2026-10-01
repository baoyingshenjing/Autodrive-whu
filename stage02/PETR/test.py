"""第三步：推理、计算官方 mAP / NDS，并生成真实框与预测框对比图。

运行：python test.py
指定模型：python test.py --checkpoint outputs/run1/epoch_1.pth
仅重画已有结果：python test.py --vis-only --score-thr 0.1
mini 没有独立带标注测试集，本脚本评估官方 mini_val。
"""
import argparse
import copy
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'PETR-official'


def visualize_results(output_dir, start, count, score_thr, top_k):
    """直接读取官方格式预测，绘制六路相机对比图和俯视图，不重新推理。

预测框在全局坐标系中，需要先变换到自车坐标，再变换到相机坐标。
分数阈值和 top_k 只影响展示，不改变评估指标。
"""
    import matplotlib
    matplotlib.use('Agg')  # 保存图片，不依赖图形窗口或额外的 GUI 环境。
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from PIL import Image
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility
    from pyquaternion import Quaternion

    result_file = output_dir / 'eval' / 'pts_bbox' / 'results_nusc.json'
    if not result_file.is_file():
        raise FileNotFoundError('缺少预测结果，请先运行 python test.py：' + str(result_file))
    predictions = json.loads(result_file.read_text(encoding='utf-8'))['results']
    nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT / 'v1.0-mini'), verbose=False)
    tokens = sorted(predictions, key=lambda token: nusc.get('sample', token)['timestamp'])
    if start >= len(tokens):
        raise ValueError('--vis-start 超出范围，有效范围为 0 到 %d' % (len(tokens) - 1))
    # 参数写进目录名，改变阈值或展示帧数时保留各次对比结果。
    vis_dir = output_dir / 'visualizations' / (
        'start%d_count%d_thr%g_top%d' % (start, count, score_thr, top_k))
    vis_dir.mkdir(parents=True, exist_ok=True)
    channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    green, red = '#24c76a', '#f44336'
    records = []

    def to_local(box, pose, sensor=None):
        # 平移后再旋转：全局 → 自车 → 相机。复制框，避免原始结果被修改。
        box = copy.deepcopy(box)
        box.translate(-np.array(pose['translation']))
        box.rotate(Quaternion(pose['rotation']).inverse)
        if sensor is not None:
            box.translate(-np.array(sensor['translation']))
            box.rotate(Quaternion(sensor['rotation']).inverse)
        return box

    for index in range(start, min(start + count, len(tokens))):
        token = tokens[index]
        sample = nusc.get('sample', token)
        scene = nusc.get('scene', sample['scene_token'])['name']
        selected = sorted(
            (p for p in predictions[token] if p['detection_score'] >= score_thr),
            key=lambda p: p['detection_score'], reverse=True)[:top_k]
        pred_boxes = [Box(p['translation'], p['size'], Quaternion(p['rotation']),
                          name=p['detection_name'], score=p['detection_score']) for p in selected]
        # 仅显示模型支持的 10 类 GT；不绘制数据集里的其他类别。
        gt_global = [nusc.get_box(t) for t in sample['anns']]
        gt_global = [b for b in gt_global if category_to_detection_name(b.name) is not None]

        # 上两排是真实框，下两排是预测框；每个区域均按相同六相机顺序排列。
        fig, axes = plt.subplots(4, 3, figsize=(18, 14))
        camera_counts = {}
        for camera_index, channel in enumerate(channels):
            sd = nusc.get('sample_data', sample['data'][channel])
            image_path, gt_camera, intrinsic = nusc.get_sample_data(sd['token'])
            pose = nusc.get('ego_pose', sd['ego_pose_token'])
            sensor = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            with Image.open(image_path) as source:
                pixels = np.array(source.convert('RGB'))
            size = (pixels.shape[1], pixels.shape[0])
            gt_camera = [b for b in gt_camera if category_to_detection_name(b.name) is not None]
            pred_camera = [to_local(b, pose, sensor) for b in pred_boxes]
            camera_counts[channel] = {}
            for offset, name, boxes, color in (
                    (0, 'GT', gt_camera, green), (6, 'Prediction', pred_camera, red)):
                ax = axes.flat[camera_index + offset]
                ax.imshow(pixels)
                visible = 0
                for box in boxes:
                    # devkit ANY：至少一个角点在图内，且所有角点均在相机前方。
                    # 画面外或跨越相机近裁剪面的框不画；俯视图仍保留它们。
                    if box_in_image(box, intrinsic, size, vis_level=BoxVisibility.ANY):
                        box.render(ax, view=intrinsic, normalize=True,
                                   colors=(color, color, color), linewidth=1.3)
                        visible += 1
                camera_counts[channel][name] = visible
                ax.set(xlim=(0, size[0]), ylim=(size[1], 0))
                ax.set_title('%s | %s | %d boxes' % (name, channel, visible),
                             color=color, fontsize=10)
                ax.axis('off')
        subtitle = 'score >= %g | top %d per frame | %d selected predictions' % (
            score_thr, top_k, len(pred_boxes))
        if not pred_boxes:
            subtitle += '\nNo predictions above threshold'
        fig.suptitle('PETR | Ground truth (green) / Predictions (red)\n'
                     '%s | frame %d | %s\n%s' % (scene, index, token, subtitle), fontsize=13)
        fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.91, hspace=.23, wspace=.025)
        filename = 'frame_%03d_%s' % (index, scene)
        fig.savefig(str(vis_dir / (filename + '_cameras.png')), dpi=120)
        plt.close(fig)

        # 俯视图以该帧前相机时间戳的自车位姿为原点，x 向前、y 向左，单位米。
        # 无须读取激光雷达文件。真实框和预测框用同一变换，便于观察定位误差。
        front = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        ego_pose = nusc.get('ego_pose', front['ego_pose_token'])
        fig, ax = plt.subplots(figsize=(8, 8))
        for boxes, color in ((gt_global, green), (pred_boxes, red)):
            for box in boxes:
                to_local(box, ego_pose).render(ax, colors=(color, color, color), linewidth=1.2)
        ax.scatter([0], [0], color='#1769aa', marker='o', s=45, zorder=5)
        ax.set(xlim=(-55, 55), ylim=(-55, 55), xlabel='Ego x / forward (m)',
               ylabel='Ego y / left (m)', title='BEV | %s | frame %d\n%s' % (scene, index, subtitle))
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
        ax.legend(handles=[Line2D([0], [0], color=green, label='Ground truth (10 classes)'),
                           Line2D([0], [0], color=red, label='Prediction'),
                           Line2D([0], [0], marker='o', color='#1769aa', linestyle='', label='Ego origin')],
                  loc='upper right')
        fig.tight_layout()
        fig.savefig(str(vis_dir / (filename + '_bev.png')), dpi=150)
        plt.close(fig)
        records.append(dict(index=index, sample_token=token, scene=scene,
                            selected_predictions=len(pred_boxes), camera_boxes=camera_counts))

    # 保存实际展示的帧、分数阈值、框数，便于课堂复现和解释“为什么没有框”。
    info = dict(score_threshold=score_thr, top_k=top_k, frames=records,
                note='Display filters only; metrics unchanged. GT: 10 detection classes, '
                     'without evaluation range/point-count filtering. Camera visibility: devkit ANY. '
                     'BEV uses front-camera ego pose, view range +/-55 m.')
    (vis_dir / 'visualization_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('\n可视化完成：%d 帧，每帧六相机对比图 + 俯视图。' % len(records))
    print('图片目录：' + str(vis_dir))


def main():
    # Windows 重定向到日志时统一用 UTF-8，避免中文说明乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure') and not stream.isatty():
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='权重路径；默认使用最近一次成功训练的权重')
    parser.add_argument('--output-dir', help='结果目录；默认放到该权重所在目录的 test 子目录')
    parser.add_argument('--vis-count', type=int, default=3, help='可视化帧数，默认 3；设为 0 可关闭')
    parser.add_argument('--vis-start', type=int, default=0, help='按时间排序后的起始帧序号，从 0 开始')
    parser.add_argument('--score-thr', type=float, default=0.1, help='展示预测框的最低分数，默认 0.1')
    parser.add_argument('--vis-top-k', type=int, default=20, help='每帧最多展示多少个高分预测框，默认 20')
    parser.add_argument('--vis-only', action='store_true', help='只读取已有预测画图，跳过推理与评估')
    args = parser.parse_args()
    if args.vis_count < 0 or args.vis_start < 0 or args.vis_top_k < 1 or not 0 <= args.score_thr <= 1:
        parser.error('要求 vis-count >= 0、vis-start >= 0、vis-top-k >= 1、0 <= score-thr <= 1')
    if args.vis_only and args.vis_count == 0:
        parser.error('--vis-only 需要 --vis-count 大于 0')
    # train.py 成功后更新此记录；学生拿到干净代码后需要先训练。
    latest = ROOT / 'outputs' / 'latest_checkpoint.txt'
    if args.checkpoint:
        checkpoint = ROOT / args.checkpoint
    elif latest.is_file():
        checkpoint = ROOT / latest.read_text(encoding='utf-8').strip()
    else:
        parser.error('请先运行 python train.py，或用 --checkpoint 指定权重')
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error('找不到权重：' + str(checkpoint))
    if not (ROOT / 'v1.0-mini' / 'nuscenes_mini_infos_val.pkl').is_file():
        parser.error('请先运行 python preprocess.py')
    output_dir = (ROOT / args.output_dir).resolve() if args.output_dir else checkpoint.parent / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.vis_only:
        visualize_results(output_dir, args.vis_start, args.vis_count, args.score_thr, args.vis_top_k)
        return

    os.chdir(str(REPO))
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    sys.path[:0] = [str(REPO), str(REPO / 'mmdetection3d')]
    # predictions.pkl 保存原始预测；eval/pts_bbox 保存官方格式预测及指标。
    # 同一个 output-dir 再次测试会更新其预测和指标文件。
    sys.argv = [str(REPO / 'tools' / 'test.py'),
                'projects/configs/teaching/petr_mini_r50.py', str(checkpoint),
                '--eval', 'bbox', '--out', str(output_dir / 'predictions.pkl'),
                '--eval-options', 'jsonfile_prefix=' + (output_dir / 'eval').as_posix()]
    runpy.run_path(str(REPO / 'tools' / 'test.py'), run_name='__main__')
    print('\n测试完成，结果目录：' + str(output_dir))
    if args.vis_count:
        visualize_results(output_dir, args.vis_start, args.vis_count, args.score_thr, args.vis_top_k)


if __name__ == '__main__':
    main()
