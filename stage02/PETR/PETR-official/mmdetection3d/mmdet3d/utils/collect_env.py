# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash
from distutils.errors import DistutilsPlatformError
import sys
import platform
import torch
import mmcv

import mmdet
import mmdet3d
import mmseg
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE


def collect_env():
    """Collect the information of the running environments."""
    try:
        env_info = collect_base_env()
    except DistutilsPlatformError:
        if sys.platform != 'win32':
            raise
        # Precompiled MMCV operators do not require a local MSVC compiler.
        # The optional compiler inventory must not block GPU execution.
        env_info = {
            'sys.platform': sys.platform, 'Python': sys.version,
            'Platform': platform.platform(), 'PyTorch': torch.__version__,
            'CUDA available': torch.cuda.is_available(),
            'CUDA runtime': torch.version.cuda, 'MMCV': mmcv.__version__,
            'MSVC': 'not installed; using precompiled operators',
        }
        if torch.cuda.is_available():
            env_info['GPU 0'] = torch.cuda.get_device_name(0)
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMSegmentation'] = mmseg.__version__
    env_info['MMDetection3D'] = mmdet3d.__version__ + '+' + get_git_hash()[:7]
    env_info['spconv2.0'] = IS_SPCONV2_AVAILABLE
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
