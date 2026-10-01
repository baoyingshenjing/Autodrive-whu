"""第二步：用单张 GPU 训练 PETR，默认训练 1 轮。

运行：python train.py
增加轮数：python train.py --epochs 5
每次默认创建独立结果目录，完成后 test.py 自动找到新权重。
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'PETR-official'


def main():
    # Windows 重定向到日志时统一用 UTF-8，避免中文说明乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure') and not stream.isatty():
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1, help='总训练轮数，默认 1')
    parser.add_argument('--work-dir', help='结果目录，相对路径以项目根目录为准')
    parser.add_argument('--resume-from', help='继续训练的 epoch_N.pth 路径')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs 必须大于 0')
    if not (ROOT / 'v1.0-mini' / 'nuscenes_mini_infos_train.pkl').is_file():
        parser.error('请先运行 python preprocess.py')

    # 默认使用时间命名，避免覆盖之前的教学结果。
    work_dir = ROOT / (args.work_dir or ('outputs/train_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    work_dir = work_dir.resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        parser.error('结果目录已有文件，请换一个 --work-dir，以免覆盖旧结果')
    resume = (ROOT / args.resume_from).resolve() if args.resume_from else None
    if resume and not resume.is_file():
        parser.error('找不到恢复权重：' + str(resume))
    if resume:
        import torch
        completed = torch.load(str(resume), map_location='cpu')['meta']['epoch']
        if args.epochs <= completed:
            parser.error('--epochs 是目标总轮数，必须大于已完成的 %s 轮' % completed)

    # 官方代码从仓库目录加载插件和骨干权重；这里统一处理路径。
    os.chdir(str(REPO))
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    sys.path[:0] = [str(REPO), str(REPO / 'mmdetection3d')]
    # 使用现有轻量配置：六路 192×512 图片、batch=1、Windows workers=0。
    # train.py 只训练，评估由 test.py 单独执行。
    sys.argv = [str(REPO / 'tools' / 'train.py'),
                'projects/configs/teaching/petr_mini_r50.py',
                '--gpus', '1', '--seed', '42', '--no-validate',
                '--work-dir', str(work_dir)]
    if resume:
        sys.argv += ['--resume-from', str(resume)]
    # 框架要求这两个轮数一致，统一由 --epochs 设置。
    sys.argv += ['--cfg-options', 'runner.max_epochs=%d' % args.epochs,
                 'total_epochs=%d' % args.epochs]
    runpy.run_path(str(REPO / 'tools' / 'train.py'), run_name='__main__')

    checkpoint = work_dir / ('epoch_%d.pth' % args.epochs)
    if not checkpoint.is_file():
        raise RuntimeError('训练没有生成预期权重：' + str(checkpoint))
    # 仅在训练成功后更新记录；失败时仍可测试上一次成功训练的模型。
    (ROOT / 'outputs').mkdir(exist_ok=True)
    try:
        checkpoint_record = checkpoint.relative_to(ROOT).as_posix()
    except ValueError:
        checkpoint_record = str(checkpoint)
    (ROOT / 'outputs' / 'latest_checkpoint.txt').write_text(checkpoint_record, encoding='utf-8')
    print('\n训练完成，权重：' + str(checkpoint))
    print('下一步：python test.py')


if __name__ == '__main__':
    main()
