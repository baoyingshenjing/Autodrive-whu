"""Evaluate a trained YOLO model and report detection metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import configure_runtime, report_path, resolve_dataset, workflow_path


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def collect_metrics(results: Any) -> dict:
    metrics = {}
    if getattr(results, "results_dict", None):
        metrics.update(jsonable(results.results_dict))
    box = getattr(results, "box", None)
    for name in ("map", "map50", "map75", "mp", "mr"):
        if box is not None and hasattr(box, name):
            metrics[name] = jsonable(getattr(box, name))
    return metrics


def run(args: argparse.Namespace) -> dict:
    configure_runtime()
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    dataset_root, data_path = resolve_dataset(args.dataset_root)
    project = workflow_path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    model_path = workflow_path(args.model)
    os.chdir(dataset_root)
    model = YOLO(str(model_path))
    output = {"model": report_path(model_path), "data": str(data_path), "splits": args.splits, "results": {}}
    for split in args.splits:
        dataset = check_det_dataset(str(data_path), autodownload=False, split=split)
        source = dataset[split]
        metrics = model.val(
            data=str(data_path), split=split, imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, conf=args.conf, iou=args.iou,
            plots=args.plots, project=str(project), name=f"{args.name}_val_{split}", exist_ok=True,
        )
        val_dir = getattr(metrics, "save_dir", "")
        result = {"data": str(data_path), "source": jsonable(source), "metrics": collect_metrics(metrics), "val_dir": report_path(Path(val_dir)) if val_dir else ""}
        output["results"][split] = result
    summary_path = project / f"{args.name}_summary.json"
    output["summary"] = report_path(summary_path)
    summary_path.write_text(json.dumps(jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(jsonable(output), ensure_ascii=False, indent=2))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="nuScenes-trained YOLO .pt checkpoint")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset directory containing data.yaml")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--name", default="nuscenes")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
