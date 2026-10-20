"""Actually execute the CUDA voxelization and rotated NMS used by PointPillars."""
import json
from pathlib import Path
import torch
from mmcv.ops import Voxelization, nms_rotated

assert torch.cuda.is_available()
points = torch.tensor([[0., 0., 0., .5], [.1, .1, .1, .4],
                       [2., 2., 0., .2]], device='cuda')
voxelizer = Voxelization(voxel_size=[.25, .25, 8],
                        point_cloud_range=[-25, -25, -5, 25, 25, 3],
                        max_num_points=64, max_voxels=(6000, 8000))
voxels, coords, numbers = voxelizer(points)
assert voxels.is_cuda and int(numbers.sum()) == 3
boxes = torch.tensor([[0., 0., 2., 2., 0.], [0., 0., 2., 2., 0.]], device='cuda')
scores = torch.tensor([.9, .8], device='cuda')
_, keep = nms_rotated(boxes, scores, .1)
assert keep.numel() == 1
torch.cuda.synchronize()
result = dict(status='CUDA_OPS_OK', voxel_count=len(coords), nms_keep=int(keep.numel()),
              torch=torch.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(0))
output = Path(__file__).resolve().parent / 'outputs' / 'setup' / 'cuda_ops.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
