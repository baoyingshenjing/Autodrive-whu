"""Repaint text in existing BEV figures; never execute model inference."""
import json
from pathlib import Path
from course_visualization import render_predictions

ROOT = Path(__file__).resolve().parent


def main():
    audit = []
    jobs = [('baseline_compact_bev','visualizations',.10),
            ('finetuned_compact','visualizations',.10),
            ('finetuned_compact','visualizations_thr010',.10),
            ('finetuned_compact','visualizations_thr030',.30)]
    for run, folder, threshold in jobs:
        work = ROOT/'outputs'/run
        directory = work/folder
        old = json.loads((directory/'visualization_summary.json').read_text(encoding='utf-8'))
        new = render_predictions(work/'predictions.pkl',directory,3,threshold,ROOT/'data/nuscenes')
        assert [row['detections'] for row in new] == [row['detections'] for row in old['rendered']]
        assert [row['source_lidar'] for row in new] == [row['source_lidar'] for row in old['rendered']]
        audit.append(dict(run=run,folder=folder,threshold=threshold,
                          detections=[row['detections'] for row in new],text_color='black'))
    (ROOT/'outputs/setup/report_figure_format.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('FIGURE_FORMAT_OK cached predictions only; counts and frames unchanged')


if __name__ == '__main__':
    main()
