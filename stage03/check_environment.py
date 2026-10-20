"""Report the exact software, GPU and supplied-data state used by the course."""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import mmcv
import mmdet
import mmdet3d
import mmengine
import numpy as np
import torch

from course_common import COURSE_ROOT, data_root_from_arg, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=None)
    args = parser.parse_args()
    root = data_root_from_arg(args.data_root)
    gpu_query = 'unavailable'
    try:
        gpu_query = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'], text=True, encoding='utf-8').strip()
    except Exception as exc:  # nvidia-smi may not be on PATH in a student PC.
        gpu_query = f'unavailable: {exc}'
    report = dict(
        python=sys.version, platform=platform.platform(),
        torch=torch.__version__, torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_query=gpu_query, numpy=np.__version__, mmcv=mmcv.__version__,
        mmengine=mmengine.__version__, mmdet=mmdet.__version__,
        mmdet3d=mmdet3d.__version__, data_root=str(root),
        v1_mini_exists=(root / 'v1.0-mini').is_dir(),
        keyframe_bins=len(list((root / 'samples' / 'LIDAR_TOP').glob('*.bin'))),
        sweep_bins=len(list((root / 'sweeps' / 'LIDAR_TOP').glob('*.bin'))))
    output = COURSE_ROOT / 'outputs' / 'environment.json'
    save_json(output, report)
    print('ENVIRONMENT_OK' if report['cuda_available'] and report['v1_mini_exists']
          else 'ENVIRONMENT_CHECK_FAILED')
    for key, value in report.items():
        print(f'{key}={value}')
    print(f'wrote={output}')


if __name__ == '__main__':
    main()
