"""Train an Ultralytics YOLO detector on the converted nuScenes dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import WORKFLOW_ROOT, configure_runtime, resolve_dataset, workflow_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("weights/yolo11n.pt"), help="Path inside yolo_workflow, or absolute path")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset directory containing data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0", help="GPU id, cpu, or auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/checkpoints"))
    parser.add_argument("--name", default="nuscenes_yolo11n")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    configure_runtime()

    from ultralytics import YOLO

    dataset_root, data_path = resolve_dataset(args.dataset_root)
    model_path = workflow_path(args.model)
    project_path = workflow_path(args.project)
    os.chdir(dataset_root)
    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
