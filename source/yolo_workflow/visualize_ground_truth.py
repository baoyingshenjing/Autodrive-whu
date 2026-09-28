"""Visualize YOLO ground-truth labels for a named dataset split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import cv2

from common import configure_runtime, report_path, resolve_dataset, workflow_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def collect_images(data_path: Path, split: str, max_images: int) -> tuple[list[Path], dict]:
    """Resolve a data.yaml split and return its image paths in stable order."""
    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(str(data_path), autodownload=False, split=split)
    source = dataset[split]
    sources = source if isinstance(source, (list, tuple)) else [source]
    images: list[Path] = []

    for item in sources:
        source_path = Path(item)
        if source_path.is_file() and source_path.suffix.lower() == ".txt":
            for line in source_path.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if not value:
                    continue
                image_path = Path(value)
                if not image_path.is_absolute():
                    image_path = (source_path.parent / image_path).resolve()
                images.append(image_path)
        elif source_path.is_dir():
            images.extend(
                path for path in source_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(source_path)
        else:
            raise ValueError(f"Unsupported image source: {source_path}")

    images = sorted(dict.fromkeys(path.resolve() for path in images))
    if max_images:
        images = images[:max_images]
    return images, dataset


def label_path_for(image_path: Path, dataset_root: Path) -> Path:
    """Map images/<split>/<name> to labels/<split>/<name>.txt."""
    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Image is outside dataset root: {image_path}") from exc
    if not relative.parts or relative.parts[0] != "images":
        raise ValueError(f"Expected an images/... path, got: {relative}")
    return dataset_root / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")


def class_name(names: Sequence[str] | dict, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return a stable, high-contrast BGR color for a class ID."""
    palette = (
        (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
        (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
        (52, 147, 26), (187, 212, 0),
    )
    return palette[class_id % len(palette)]


def draw_ground_truth(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    names: Sequence[str] | dict,
    classes: set[int] | None,
    show_labels: bool,
    line_width: int,
) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    box_count = 0

    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        class_id = int(fields[0])
        if classes is not None and class_id not in classes:
            continue
        x_center, y_center, box_width, box_height = map(float, fields[1:])
        x1 = max(0, min(width - 1, round((x_center - box_width / 2) * width)))
        y1 = max(0, min(height - 1, round((y_center - box_height / 2) * height)))
        x2 = max(0, min(width - 1, round((x_center + box_width / 2) * width)))
        y2 = max(0, min(height - 1, round((y_center + box_height / 2) * height)))
        color = class_color(class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
        if show_labels:
            text = class_name(names, class_id)
            font_scale = max(0.45, line_width * 0.25)
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, line_width - 1)
            )
            text_y = max(text_height + baseline, y1)
            cv2.rectangle(
                image,
                (x1, text_y - text_height - baseline),
                (min(width - 1, x1 + text_width), text_y),
                color,
                -1,
            )
            cv2.putText(
                image, text, (x1, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), max(1, line_width - 1), cv2.LINE_AA,
            )
        box_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Cannot write image: {output_path}")
    return box_count


def run(args: argparse.Namespace) -> dict:
    configure_runtime()
    dataset_root, data_path = resolve_dataset(args.dataset_root)
    os.chdir(dataset_root)
    images, dataset = collect_images(data_path, args.split, args.max_images)
    output_dir = workflow_path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_classes = set(args.classes) if args.classes is not None else None

    total_boxes = 0
    for image_path in images:
        total_boxes += draw_ground_truth(
            image_path=image_path,
            label_path=label_path_for(image_path, dataset_root),
            output_path=output_dir / image_path.name,
            names=dataset["names"],
            classes=selected_classes,
            show_labels=args.show_labels,
            line_width=args.line_width,
        )

    summary = {
        "mode": "ground_truth",
        "data": str(data_path),
        "split": args.split,
        "selected_images": len(images),
        "ground_truth_boxes": total_boxes,
        "output_dir": report_path(output_dir),
    }
    summary_path = workflow_path(args.project) / f"{args.name}_summary.json"
    summary["summary"] = report_path(summary_path)
    summary_path.write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Dataset directory containing data.yaml",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-images", type=int, default=0, help="First N images; 0 means all")
    parser.add_argument("--classes", type=int, nargs="+", help="Optional class IDs")
    parser.add_argument("--project", type=Path, default=Path("runs/visualization"))
    parser.add_argument("--name", default="ground_truth")
    parser.add_argument(
        "--show-labels", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--line-width", type=int, default=2)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
