"""Shared path and runtime helpers for the YOLO workflow scripts."""

from __future__ import annotations

import os
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parent


def configure_runtime() -> None:
    """Keep tool caches inside yolo_workflow."""
    yolo_config = WORKFLOW_ROOT / ".ultralytics"
    matplotlib_config = WORKFLOW_ROOT / ".mpl-cache"
    yolo_config.mkdir(parents=True, exist_ok=True)
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))


def resolve_dataset(dataset_root: Path) -> tuple[Path, Path]:
    """Validate a portable YOLO dataset root and return root plus data.yaml."""
    root = dataset_root.expanduser().resolve()
    required = [root / "data.yaml", root / "images", root / "labels"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Invalid dataset root; missing: " + ", ".join(str(path) for path in missing)
        )
    return root, root / "data.yaml"


def workflow_path(path: Path) -> Path:
    """Resolve relative model/output paths against yolo_workflow."""
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (WORKFLOW_ROOT / expanded).resolve()


def report_path(path: Path) -> str:
    """Report internal artifacts relative to yolo_workflow."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKFLOW_ROOT).as_posix()
    except ValueError:
        return str(resolved)
