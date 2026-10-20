"""Validate actual outputs and make a compact, machine-readable report input."""
import hashlib
import json
import math
import pickle
from pathlib import Path
from collections import Counter
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from course_common import CLASSES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifest = read(ROOT / 'data_infos' / 'manifest.json')
    assert manifest['train_samples'] == 323 and manifest['val_samples'] == 81
    assert manifest['overlap_samples'] == 0
    comparison = read(OUT / 'comparison.json')
    points = []
    train = {}
    for name, expected in [('run1_compact', 1), ('run5_compact', 5)]:
        work = OUT / name
        checkpoint = torch.load(work / f'epoch_{expected}.pth', map_location='cpu', weights_only=False)
        assert checkpoint['meta']['epoch'] == expected
        finite = all(bool(torch.isfinite(value).all()) for value in checkpoint['state_dict'].values()
                     if torch.is_floating_point(value))
        assert finite
        audit = read(work / 'training_audit.json')
        assert audit['validation_status'] == 'PARAMETERS_UPDATED'
        assert audit['changed_parameter_tensors'] > 0 and audit['gradient_parameter_tensors'] > 0
        history = read(work / 'loss_history.json')['points']
        assert history and all(math.isfinite(p['loss']) for p in history)
        assert len(history) == (323 if expected == 1 else 4*323)
        points.extend(history)
        train[name] = dict(audit=audit, checkpoint_epoch=expected, all_model_weights_finite=finite,
                           recorded_steps=len(history), first_iteration=history[0]['iteration'],
                           last_iteration=history[-1]['iteration'])
    assert len(points) == 1615
    epoch_means = []
    for index in range(5):
        values = [p['loss'] for p in points[index*323:(index+1)*323]]
        epoch_means.append(float(np.mean(values)))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
    ax.plot([p['iteration']+1 for p in points], [p['loss'] for p in points], color='#688bb6', linewidth=.5, alpha=.6, label='Batch loss')
    ax.plot([323*(i+.5) for i in range(5)], epoch_means, '-o', color='#bf5d33', label='Epoch mean')
    ax.axvline(323, color='#374151', linestyle='--', linewidth=.8, label='Resume after epoch 1')
    ax.set(xlabel='Cumulative training iteration', ylabel='Training loss', title='PointPillars: 1 epoch + resumed epochs 2-5')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'loss_all_5epochs.png')
    plt.close(fig)
    predictions = {}
    with (ROOT / 'data_infos' / 'val.pkl').open('rb') as stream:
        val_infos = pickle.load(stream)['data_list']
    val_by_index = {int(info['sample_idx']): info for info in val_infos}
    for name in ['baseline_compact_bev', 'finetuned_compact']:
        with (OUT / name / 'predictions.pkl').open('rb') as stream:
            samples = pickle.load(stream)
        assert len(samples) == 81
        tokens = []
        for sample in samples:
            meta = sample if isinstance(sample, dict) else sample.metainfo
            info = val_by_index[int(meta['sample_idx'])]
            assert Path(meta['lidar_path']).name == Path(info['lidar_points']['lidar_path']).name
            tokens.append(info['token'])
        assert len(set(tokens)) == 81 and set(tokens) == set(manifest['val_sample_tokens'])
        class_counts = Counter()
        scores_all = []
        for sample in samples:
            pred = sample['pred_instances_3d'] if isinstance(sample, dict) else sample.pred_instances_3d
            labels = pred['labels_3d'] if isinstance(pred, dict) else pred.labels_3d
            scores = pred['scores_3d'] if isinstance(pred, dict) else pred.scores_3d
            boxes = pred['bboxes_3d'] if isinstance(pred, dict) else pred.bboxes_3d
            assert bool(torch.isfinite(scores).all()) and bool(torch.isfinite(boxes.tensor).all())
            assert all(0 <= int(label) < len(CLASSES) for label in labels)
            scores_all.extend(scores.tolist())
            class_counts.update(CLASSES[int(label)] for label in labels)
        predictions[name] = dict(sample_count=len(samples), prediction_count=len(scores_all),
                                 class_counts=dict(class_counts), tokens=tokens,
                                 token_verification='sample_idx mapped to val.pkl, with LiDAR filename equality checked',
                                 score_min=min(scores_all) if scores_all else None,
                                 score_max=max(scores_all) if scores_all else None)
    assert predictions['baseline_compact_bev']['tokens'] == predictions['finetuned_compact']['tokens']
    checkpoint_path = OUT / 'run5_compact' / 'epoch_5.pth'
    result = dict(train=train, epoch_mean_loss=epoch_means, comparison=comparison,
                  predictions=predictions, environment=read(OUT / 'environment.json'),
                  data_audit=read(ROOT / 'data' / 'nuscenes' / 'source_infos_audit.json'),
                  execution=read(OUT / 'setup' / 'execution.json'),
                  final_checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  thresholds={str(thr): read(OUT / 'finetuned_compact' / f'visualizations_thr{suffix}' / 'visualization_summary.json')
                              for thr, suffix in [(0.1, '010'), (0.3, '030')]})
    (OUT / 'final_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FINAL_VALIDATION_OK epochs=5 train_steps=1615 validation_frames=81')


if __name__ == '__main__':
    main()
