"""Record actual intermediate shapes for one validation frame, without training."""
import json
import pickle
from pathlib import Path
import torch
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.utils import register_all_modules

ROOT = Path(__file__).resolve().parent


def shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [shape(x) for x in value]
    return str(type(value))


def main():
    register_all_modules(init_default_scope=True)
    model = init_model(ROOT / 'outputs/run5_compact/resolved_config.py',
                       str(ROOT / 'outputs/run5_compact/epoch_5.pth'), device='cuda:0')
    with (ROOT / 'data_infos/val.pkl').open('rb') as stream:
        info = pickle.load(stream)['data_list'][0]
    point_file = ROOT / 'data/nuscenes/samples/LIDAR_TOP' / info['lidar_points']['lidar_path']
    records = {}
    handles = []
    def hook(name):
        def record(module, inputs, output):
            records[name] = dict(input_shapes=shape(inputs), output_shapes=shape(output))
        return record
    for name in ['pts_voxel_encoder', 'pts_voxel_encoder.vfe_layers.0',
                 'pts_voxel_encoder.vfe_layers.1', 'pts_middle_encoder',
                 'pts_backbone', 'pts_neck', 'pts_bbox_head.conv_cls',
                 'pts_bbox_head.conv_reg', 'pts_bbox_head.conv_dir_cls']:
        handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
    prediction, data = inference_detector(model, str(point_file))
    for handle in handles:
        handle.remove()
    head_parameters = [name for name, _ in model.named_parameters() if name.startswith('pts_bbox_head.')]
    assert len(head_parameters) == 6
    result = dict(status='MODEL_SHAPE_TRACE_OK', sample_token=info['token'],
                  sample_idx=info['sample_idx'], source_lidar=str(point_file),
                  point_input_shape=shape(data['inputs']['points']), modules=records,
                  head_parameter_names=head_parameters,
                  anchor_count_per_location=int(model.pts_bbox_head.num_anchors),
                  final_prediction_count=int(len(prediction.pred_instances_3d.scores_3d)),
                  mode='eval; one validation frame; no gradient or optimizer update')
    (ROOT / 'outputs/setup/model_shape_trace.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
