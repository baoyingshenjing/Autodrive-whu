"""Evaluate a checkpoint only on the course validation split and save BEV images."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules

from course_common import (assert_manifest_is_single_frame, build_course_config,
                           data_root_from_arg, load_manifest, save_json,
                           patch_mmengine_windows_collect_env,
                           write_resolved_config)
from course_visualization import render_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--vis-count', type=int, default=3)
    parser.add_argument('--score-thr', type=float, default=0.30)
    parser.add_argument('--vis-only', action='store_true')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    prediction_file = output_dir / 'predictions.pkl'
    root = data_root_from_arg(args.data_root)
    manifest = load_manifest()
    assert_manifest_is_single_frame(manifest)
    if args.vis_only:
        if not prediction_file.is_file():
            raise FileNotFoundError(f'缺少 {prediction_file}；不能只重绘。')
        images = render_predictions(prediction_file, output_dir / 'visualizations',
                                    args.vis_count, args.score_thr, root)
        print(f'VISUALIZATION_OK count={len(images)}')
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f'检查点不存在：{checkpoint}')
    register_all_modules(init_default_scope=True)
    patch_mmengine_windows_collect_env()
    cfg = build_course_config(data_root=root, work_dir=output_dir,
                              max_epochs=1, load_from=str(checkpoint), resume=False)
    cfg.test_evaluator = [
        cfg.test_evaluator,
        dict(type='DumpResults', out_file_path=str(prediction_file),
             collect_device='cpu')]
    write_resolved_config(cfg, output_dir)
    save_json(output_dir / 'test_meta.json', dict(
        command=' '.join(sys.argv), checkpoint=str(checkpoint),
        data_root=str(root), input_mode=manifest['input_mode'],
        train_samples=manifest['train_samples'], val_samples=manifest['val_samples'],
        overlap_samples=manifest['overlap_samples'], status='STARTED'))
    print('TEST_START validation_split_only')
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    serializable = {str(k): float(v) if hasattr(v, '__float__') else str(v)
                    for k, v in metrics.items()}
    save_json(output_dir / 'metrics_summary.json', serializable)
    images = render_predictions(prediction_file, output_dir / 'visualizations',
                                args.vis_count, args.score_thr, root)
    meta = json.loads((output_dir / 'test_meta.json').read_text(encoding='utf-8'))
    meta.update(status='SUCCESS', metrics_file=str(output_dir / 'metrics_summary.json'),
                prediction_file=str(prediction_file), visualization_count=len(images))
    save_json(output_dir / 'test_meta.json', meta)
    print('TEST_OK')
    print(f'metrics={output_dir / "metrics_summary.json"}')
    print(f'predictions={prediction_file}')
    print(f'visualizations={len(images)}')


if __name__ == '__main__':
    main()
