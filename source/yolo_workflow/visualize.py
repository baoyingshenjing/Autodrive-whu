"""Visualize model predictions for a named YOLO dataset split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import configure_runtime, report_path, resolve_dataset, workflow_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def resolve_source(data_path: Path, split: str, max_images: int) -> tuple[Any, int | None]:
    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(str(data_path), autodownload=False, split=split)
    source = dataset[split]
    if not max_images:
        return source, None
    source_path = Path(source) if isinstance(source, str) else None
    if source_path and source_path.is_file() and source_path.suffix.lower() == ".txt":
        items = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value:
                continue
            image_path = Path(value)
            if not image_path.is_absolute():
                # Manifest entries are relative to the manifest's directory,
                # not to the process working directory used by predict().
                image_path = (source_path.parent / image_path).resolve()
            items.append(str(image_path))
    elif source_path and source_path.is_dir():
        items = [str(path) for path in sorted(source_path.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES]
    else:
        raise ValueError(f"Cannot limit images for source: {source}")
    selected = items[:max_images]
    return selected, len(selected)


def run(args: argparse.Namespace) -> dict:
    configure_runtime()
    from ultralytics import YOLO

    model_path = workflow_path(args.model)
    dataset_root, data_path = resolve_dataset(args.dataset_root)
    project = workflow_path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    os.chdir(dataset_root)
    source, selected_count = resolve_source(data_path, args.split, args.max_images)
    model = YOLO(str(model_path))
    results = model.predict(
        source=source, imgsz=args.imgsz, device=args.device, workers=args.workers,
        conf=args.conf, iou=args.iou, classes=args.classes, save=True,
        save_txt=args.save_txt, save_conf=args.save_txt, show_labels=args.show_labels,
        show_conf=args.show_conf, line_width=args.line_width, project=str(project),
        name=args.name, exist_ok=True, verbose=args.verbose,
    )
    output_dir = project / args.name
    summary = {
        "model": report_path(model_path), "data": str(data_path), "split": args.split,
        "source": jsonable(source), "selected_images": selected_count if selected_count is not None else len(results),
        "prediction_results": len(results), "output_dir": report_path(output_dir), "save_txt": args.save_txt,
    }
    summary_path = project / f"{args.name}_summary.json"
    summary["summary"] = report_path(summary_path)
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="YOLO .pt checkpoint")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset directory containing data.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-images", type=int, default=0, help="First N images; 0 means all")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--classes", type=int, nargs="+", help="Optional class IDs")
    parser.add_argument("--project", type=Path, default=Path("runs/visualization"))
    parser.add_argument("--name", default="test")
    parser.add_argument("--save-txt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-conf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
