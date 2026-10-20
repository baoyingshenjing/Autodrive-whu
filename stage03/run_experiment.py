"""Execute every required course step, keeping timestamped console evidence."""
import json
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-from', default='environment',
                        choices=['environment', 'cuda_ops', 'preprocess', 'baseline', 'run1', 'plot1',
                                 'run5', 'plot5', 'finetuned', 'comparison'])
    args = parser.parse_args()
    environment = os.environ.copy()
    environment.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                       NUSCENES_ROOT=str(ROOT / 'data' / 'nuscenes'),
                       TEMP=str(ROOT / 'tmp'), TMP=str(ROOT / 'tmp'))
    records = []
    steps = [
        ('environment', ['check_environment.py']),
        ('cuda_ops', ['check_cuda_ops.py']),
        ('preprocess', ['preprocess.py']),
        ('baseline', ['test.py', '--checkpoint', 'checkpoints/pointpillars_nuscenes_official.pth',
                      '--output-dir', 'outputs/baseline_compact_bev', '--vis-count', '3', '--score-thr', '0.10']),
        ('run1', ['train.py', '--epochs', '1', '--work-dir', 'outputs/run1_compact', '--amp']),
        ('plot1', ['plot_training.py', '--work-dir', 'outputs/run1_compact']),
        ('run5', ['train.py', '--resume-from', 'outputs/run1_compact/epoch_1.pth', '--epochs', '5',
                  '--work-dir', 'outputs/run5_compact', '--amp']),
        ('plot5', ['plot_training.py', '--work-dir', 'outputs/run5_compact']),
        ('finetuned', ['test.py', '--checkpoint', 'outputs/run5_compact/epoch_5.pth',
                       '--output-dir', 'outputs/finetuned_compact', '--vis-count', '3', '--score-thr', '0.10']),
        ('comparison', ['compare_results.py', '--output', 'outputs/comparison.json']),
    ]
    setup = ROOT / 'outputs' / 'setup'
    setup.mkdir(parents=True, exist_ok=True)
    start_index = next(i for i, step in enumerate(steps) if step[0] == args.start_from)
    if start_index and (setup / 'execution.json').is_file():
        old_records = json.loads((setup / 'execution.json').read_text(encoding='utf-8'))
        prior = {name for name, _ in steps[:start_index]}
        records = [item for item in old_records if item['step'] in prior]
    steps = steps[start_index:]
    for name, arguments in steps:
        checkpoint_to_create = {'run1': ROOT / 'outputs/run1_compact/epoch_1.pth',
                                'run5': ROOT / 'outputs/run5_compact/epoch_5.pth'}.get(name)
        if checkpoint_to_create and checkpoint_to_create.exists():
            raise RuntimeError(f'Existing completed checkpoint: {checkpoint_to_create}; use --start-from to skip completed training.')
        print('STEP_START', name, datetime.now().isoformat(), flush=True)
        started = time.perf_counter()
        with (setup / (name + '_console.log')).open('w', encoding='utf-8') as log:
            completed = subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT,
                                       env=environment, stdout=log, stderr=subprocess.STDOUT)
        records.append(dict(step=name, seconds=round(time.perf_counter()-started, 3),
                            exit_code=completed.returncode, command=[sys.executable, *arguments]))
        (setup / 'execution.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        print('STEP_END', name, records[-1]['seconds'], completed.returncode, flush=True)
        if completed.returncode:
            print((setup / (name + '_console.log')).read_text(encoding='utf-8')[-6000:], flush=True)
            sys.exit(completed.returncode)
    from course_visualization import render_predictions
    fine = ROOT / 'outputs' / 'finetuned_compact'
    shutil.copytree(fine / 'visualizations', fine / 'visualizations_thr010', dirs_exist_ok=True)
    render_predictions(fine / 'predictions.pkl', fine / 'visualizations_thr030', 3, .30,
                       ROOT / 'data' / 'nuscenes')
    print('EXPERIMENT_OK', flush=True)


if __name__ == '__main__':
    main()
