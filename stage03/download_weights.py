"""Copy the already supplied official PointPillars checkpoint; never download data."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from course_common import COURSE_ROOT

CHECKPOINT = 'hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth'
CONFIG = 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py'
EXPECTED_SHA256 = 'f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', default=None,
                        help='directory that contains the supplied official .pth and .py')
    args = parser.parse_args()
    default = COURSE_ROOT.parent / '.pointpillars' / 'course' / 'checkpoints'
    source = Path(args.source_dir).resolve() if args.source_dir else default
    destination = COURSE_ROOT / 'checkpoints'
    source_ckpt, source_cfg = source / CHECKPOINT, source / CONFIG
    if not source_ckpt.is_file() or not source_cfg.is_file():
        raise FileNotFoundError(
            '找不到已提供的官方权重/配置。不会自动下载。\n'
            f'请把它们放到 {source}，或传入 --source-dir。')
    actual = sha256(source_ckpt)
    if actual.lower() != EXPECTED_SHA256:
        raise RuntimeError(f'权重 SHA256 不匹配：{actual}')
    destination.mkdir(exist_ok=True)
    shutil.copy2(source_ckpt, destination / 'pointpillars_nuscenes_official.pth')
    shutil.copy2(source_cfg, destination / 'official_pointpillars_config.py')
    (destination / 'official_weight.sha256').write_text(actual + '\n', encoding='ascii')
    print('WEIGHTS_OK')
    print(f'checkpoint={destination / "pointpillars_nuscenes_official.pth"}')
    print(f'sha256={actual}')
    print('source=local supplied file; no nuScenes data was downloaded')


if __name__ == '__main__':
    main()
