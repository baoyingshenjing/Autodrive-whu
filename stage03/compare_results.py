"""Make a small, honest before/after report for the same mini-val split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from course_common import COURSE_ROOT, save_json


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def metric(metrics: dict, suffix: str) -> float:
    choices = [(key, value) for key, value in metrics.items() if key.endswith(suffix)]
    if len(choices) != 1:
        raise RuntimeError(f'未能唯一找到 {suffix}: {choices}')
    return float(choices[0][1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='outputs/baseline_compact_bev')
    parser.add_argument('--finetuned', default='outputs/finetuned_compact')
    parser.add_argument('--run1', default='outputs/run1_compact')
    parser.add_argument('--run5', default='outputs/run5_compact')
    parser.add_argument('--output', default='outputs/comparison.json')
    args = parser.parse_args()
    base_dir, fine_dir = Path(args.baseline), Path(args.finetuned)
    base, fine = (read_json(base_dir / 'metrics_summary.json'),
                  read_json(fine_dir / 'metrics_summary.json'))
    report = dict(
        configuration='single keyframe; x/y range [-25,25] m; batch=1; AMP; head-only',
        split='nuScenes v1.0-mini: train=323 keyframes, validation=81 keyframes, overlap=0',
        baseline=dict(NDS=metric(base, '/NDS'), mAP=metric(base, '/mAP'),
                      metrics_file=str(base_dir / 'metrics_summary.json')),
        finetuned_5_epoch=dict(NDS=metric(fine, '/NDS'), mAP=metric(fine, '/mAP'),
                               metrics_file=str(fine_dir / 'metrics_summary.json')),
        run1_audit=read_json(Path(args.run1) / 'training_audit.json'),
        run5_audit=read_json(Path(args.run5) / 'training_audit.json'),
        caveat=('This is a tiny mini split and a teaching-range model, not a '
                'full nuScenes benchmark. The validation set was never used '
                'for gradient updates, but it is used for model selection/reporting.'))
    report['delta'] = dict(
        NDS=round(report['finetuned_5_epoch']['NDS'] - report['baseline']['NDS'], 4),
        mAP=round(report['finetuned_5_epoch']['mAP'] - report['baseline']['mAP'], 4))
    output = Path(args.output)
    save_json(output, report)
    print('COMPARISON_OK')
    print(f"NDS: {report['baseline']['NDS']:.4f} -> {report['finetuned_5_epoch']['NDS']:.4f} "
          f"(delta {report['delta']['NDS']:+.4f})")
    print(f"mAP: {report['baseline']['mAP']:.4f} -> {report['finetuned_5_epoch']['mAP']:.4f} "
          f"(delta {report['delta']['mAP']:+.4f})")
    print(f'output={output.resolve()}')


if __name__ == '__main__':
    main()
