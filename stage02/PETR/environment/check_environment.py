"""可选检查：打印版本，并真实执行 DCNv2 的 GPU 前向/反向计算。

激活环境后运行：python environment/check_environment.py
正常教学只需要根目录的三个入口，无需每次运行此检查。
"""
import json
import platform
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'PETR-official'), str(ROOT/'PETR-official'/'mmdetection3d')]
import torch
import torchvision
import mmcv
import mmdet
import mmdet3d
import mmseg
from mmcv.ops import ModulatedDeformConv2dPack
import projects.mmdet3d_plugin


def main():
    assert torch.cuda.is_available(), 'CUDA unavailable'
    conv = ModulatedDeformConv2dPack(3, 8, 3, padding=1).cuda()
    x = torch.randn(1, 3, 16, 16, device='cuda', requires_grad=True)
    conv(x).square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(x.grad).all()
    report = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  torch=torch.__version__, torchvision=torchvision.__version__,
                  cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  gpu_total_mib=torch.cuda.get_device_properties(0).total_memory//1024**2,
                  mmcv=mmcv.__version__, mmdet=mmdet.__version__,
                  mmdet3d=mmdet3d.__version__, mmseg=mmseg.__version__,
                  dcn_cuda_forward_backward='passed', petr_plugin_import='passed')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
