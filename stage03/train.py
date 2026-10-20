"""Run auditable head-only PointPillars fine-tuning on the course train split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules

import course_modules  # noqa: F401  registers HeadOnlyGradientAuditHook
from course_common import (COURSE_ROOT, assert_manifest_is_single_frame,
                           build_course_config, data_root_from_arg, load_manifest,
                           patch_mmengine_windows_collect_env, save_json,
                           write_resolved_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, required=True,
                        help='cumulative epoch count, not extra epochs')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--checkpoint', default=None,
                        help='official checkpoint for a new run')
    parser.add_argument('--resume-from', default=None,
                        help='epoch_N.pth to resume; --epochs remains cumulative')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--amp', action='store_true',
                        help='enable AMP only after the FP32 route was tested')
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('--epochs 必须至少为 1')
    if args.checkpoint and args.resume_from:
        raise ValueError('--checkpoint 与 --resume-from 不能同时使用')
    root = data_root_from_arg(args.data_root)
    manifest = load_manifest()
    assert_manifest_is_single_frame(manifest)
    work_dir = Path(args.work_dir).resolve()
    official = COURSE_ROOT / 'checkpoints' / 'pointpillars_nuscenes_official.pth'
    if args.resume_from:
        load_from = str(Path(args.resume_from).resolve())
        resume = True
    else:
        load_from = str(Path(args.checkpoint).resolve()) if args.checkpoint else str(official)
        resume = False
    if not Path(load_from).is_file():
        raise FileNotFoundError(f'检查点不存在：{load_from}')
    register_all_modules(init_default_scope=True)
    patch_mmengine_windows_collect_env()
    cfg = build_course_config(data_root=root, work_dir=work_dir,
                              max_epochs=args.epochs, load_from=load_from,
                              resume=resume, amp=args.amp)
    cfg.custom_hooks = [dict(type='HeadOnlyGradientAuditHook',
                             audit_file=str(work_dir / 'training_audit.json'))]
    write_resolved_config(cfg, work_dir)
    save_json(work_dir / 'run_meta.json', dict(
        command=' '.join(sys.argv), data_root=str(root), input_mode=manifest['input_mode'],
        train_samples=manifest['train_samples'], val_samples=manifest['val_samples'],
        overlap_samples=manifest['overlap_samples'], max_epochs=args.epochs,
        amp=args.amp, max_voxels_train=6000, max_voxels_test=8000,
        bev_range_meters='x,y in [-25,25]',
        mode='head_only', load_from=load_from, resume=resume,
        status='STARTED'))
    save_json(work_dir / 'data_manifest.json', manifest)
    print('TRAIN_START')
    print(f'work_dir={work_dir}')
    print(f'epochs_cumulative={args.epochs} mode=head_only amp={args.amp}')
    print('input=single_keyframe_xyz_intensity train/val_overlap=0')
    runner = Runner.from_cfg(cfg)
    runner.train()
    audit_path = work_dir / 'training_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('validation_status') != 'PARAMETERS_UPDATED':
        raise RuntimeError(f'训练审计未通过：{audit_path}')
    meta = json.loads((work_dir / 'run_meta.json').read_text(encoding='utf-8'))
    meta['status'] = 'SUCCESS_PARAMETERS_UPDATED'
    meta['audit_file'] = str(audit_path)
    save_json(work_dir / 'run_meta.json', meta)
    print('TRAIN_OK PARAMETERS_UPDATED')
    print(f'audit={audit_path}')


if __name__ == '__main__':
    main()
