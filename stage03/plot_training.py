"""Plot the audited training-loss points after a successful training run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    points = json.loads((work / 'loss_history.json').read_text(encoding='utf-8'))['points']
    if not points:
        raise RuntimeError('没有记录到 loss，不能绘图。')
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot([x['iteration'] for x in points], [x['loss'] for x in points], linewidth=0.8)
    ax.set(xlabel='iteration', ylabel='training loss', title='PointPillars head-only fine-tuning')
    ax.grid(alpha=0.25)
    output = work / 'loss.png'
    fig.savefig(output, bbox_inches='tight')
    print(f'PLOT_OK {output}')


if __name__ == '__main__':
    main()
