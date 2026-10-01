"""第一步：把 nuScenes-mini 的图片、标注和相机参数整理成训练/验证索引。

运行：python preprocess.py
原始数据保持不变，生成 v1.0-mini/nuscenes_mini_infos_{train,val}.pkl。
PETR 使用六个相机；当前教学数据没有激光雷达文件，因此不读取点云和 sweeps。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    # Windows 重定向到日志时统一用 UTF-8，避免中文说明乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure') and not stream.isatty():
            stream.reconfigure(encoding='utf-8')
    argparse.ArgumentParser(description=__doc__).parse_args()
    # 路径以脚本所在目录为准，从其他工作目录启动也能找到数据。
    data_root = ROOT / 'v1.0-mini'
    metadata = data_root / 'v1.0-mini'
    if not (metadata / 'sample_annotation.json').is_file():
        raise FileNotFoundError('缺少标注文件：' + str(metadata / 'sample_annotation.json'))
    sys.path.insert(0, str(ROOT / 'PETR-official' / 'mmdetection3d'))
    import mmcv
    from nuscenes.utils.splits import mini_train, mini_val
    from tools.data_converter.nuscenes_converter import create_nuscenes_infos

    # 转换器会关联多个 JSON 表，读取相机内外参，转换三维框坐标。
    # 标注不能仅靠 sample_annotation.json 一张表完成匹配。
    # 按官方场景划分：8 个训练场景、2 个验证场景，避免相邻帧混入两组。
    create_nuscenes_infos(str(data_root), 'nuscenes_mini',
                          version='v1.0-mini', max_sweeps=0, camera_only=True)
    train = mmcv.load(str(data_root / 'nuscenes_mini_infos_train.pkl'))['infos']
    val = mmcv.load(str(data_root / 'nuscenes_mini_infos_val.pkl'))['infos']
    scenes = {s['token']: s['name'] for s in json.loads(
        (metadata / 'scene.json').read_text(encoding='utf-8'))}
    samples = {s['token']: s for s in json.loads(
        (metadata / 'sample.json').read_text(encoding='utf-8'))}

    # 检查完整 mini 的样本数、训练验证隔离，以及每帧六路图片是否存在。
    assert (len(train), len(val)) == (323, 81), 'mini 数据不完整，应为 323/81 帧'
    assert not ({s['token'] for s in train} & {s['token'] for s in val}), '训练验证有重复'
    for records, split in ((train, mini_train), (val, mini_val)):
        for record in records:
            scene_name = scenes[samples[record['token']]['scene_token']]
            assert scene_name in split, '样本所属场景不符合官方划分'
            assert len(record['cams']) == 6 and not record['sweeps']
            for camera in record['cams'].values():
                assert Path(camera['data_path']).is_file(), camera['data_path']
    print('\n预处理完成：训练 323 帧，验证 81 帧，每帧 6 张图片。')
    print('下一步：python train.py')


if __name__ == '__main__':
    main()
