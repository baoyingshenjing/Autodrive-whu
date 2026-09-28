"""Create reproducible nuScenes devkit evidence for the internship report.

The supplied course package contains nuScenes metadata plus CAM_FRONT and
LIDAR_TOP archives. This script reads the metadata through the official
devkit, inventories every scene location and sensor channel, and creates
actual camera/LiDAR views for one synchronized Boston Seaport keyframe.

Radar projection is implemented in project_sensor_to_camera as well. It is
only rendered when a matching radar binary is present under --asset-root.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, view_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return rotation @ points + translation.reshape(3, 1)


def project_sensor_to_camera(nusc: NuScenes, source_sd: dict, camera_sd: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map source sensor points to the camera plane through the global frame."""
    source_cs = nusc.get("calibrated_sensor", source_sd["calibrated_sensor_token"])
    source_pose = nusc.get("ego_pose", source_sd["ego_pose_token"])
    camera_cs = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
    camera_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])
    points = apply_transform(points, Quaternion(source_cs["rotation"]).rotation_matrix, np.asarray(source_cs["translation"]))
    points = apply_transform(points, Quaternion(source_pose["rotation"]).rotation_matrix, np.asarray(source_pose["translation"]))
    points = Quaternion(camera_pose["rotation"]).rotation_matrix.T @ (points - np.asarray(camera_pose["translation"]).reshape(3, 1))
    points = Quaternion(camera_cs["rotation"]).rotation_matrix.T @ (points - np.asarray(camera_cs["translation"]).reshape(3, 1))
    depth = points[2].copy()
    pixels = np.asarray(camera_cs["camera_intrinsic"]) @ points
    pixels[:2] /= pixels[2:3]
    return pixels, depth


def render_lidar_topdown(points: np.ndarray, output_path: Path) -> None:
    mask = (points[0] > -25) & (points[0] < 55) & (points[1] > -35) & (points[1] < 35)
    fig, ax = plt.subplots(figsize=(7.3, 5.5), dpi=180)
    ax.scatter(points[0, mask], points[1, mask], s=0.45, c="#444444", alpha=0.55)
    ax.scatter([0], [0], s=25, c="#c0392b", marker="x", label="ego vehicle")
    ax.set_xlabel("forward x (m)")
    ax.set_ylabel("left y (m)")
    ax.set_title("LIDAR_TOP point cloud in the lidar coordinate frame")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_projection(image_path: Path, pixels: np.ndarray, depth: np.ndarray, output_path: Path, title: str) -> int:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    mask = (depth > 1.0) & (depth < 80.0) & (pixels[0] >= 0) & (pixels[0] < width) & (pixels[1] >= 0) & (pixels[1] < height)
    fig, ax = plt.subplots(figsize=(10, 5.7), dpi=180)
    ax.imshow(image)
    dots = ax.scatter(pixels[0, mask], pixels[1, mask], c=depth[mask], s=1.4, cmap="magma", alpha=0.75, linewidths=0)
    fig.colorbar(dots, ax=ax, fraction=0.026, pad=0.01, label="depth (m)")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return int(mask.sum())


def render_3d_boxes(nusc: NuScenes, camera_sd: dict, image_path: Path, output_path: Path) -> int:
    """Use devkit boxes in camera coordinates and project their 3D corners."""
    _, boxes, intrinsic = nusc.get_sample_data(camera_sd["token"], box_vis_level=BoxVisibility.ANY)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 5.7), dpi=180)
    ax.imshow(image)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    drawn = 0
    for box in boxes:
        corners = box.corners()
        if np.any(corners[2] <= 0):
            continue
        pixels = view_points(corners, np.asarray(intrinsic), normalize=True)[:2]
        if pixels[0].min() < 0 or pixels[0].max() >= width or pixels[1].min() < 0 or pixels[1].max() >= height:
            continue
        for start, end in edges:
            ax.plot(pixels[0, [start, end]], pixels[1, [start, end]], color="#d35400", linewidth=1.0)
        ax.text(pixels[0].mean(), pixels[1].min(), box.name.split(".")[-1], color="#7f1d1d", fontsize=6, ha="center")
        drawn += 1
    ax.set_title("CAM_FRONT with devkit-projected 3D annotation boxes")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return drawn


def asset_path(asset_root: Path, filename: str) -> Path:
    relative = Path(filename)
    # The official metadata uses samples/<channel>/..., while the supplied
    # course archives unpack directly to <channel>/....
    if relative.parts and relative.parts[0] == "samples":
        relative = Path(*relative.parts[1:])
    return asset_root / relative


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(args.dataroot), verbose=False)
    location_counts = Counter(nusc.get("log", scene["log_token"])["location"] for scene in nusc.scene)
    camera_files = sorted((args.asset_root / "CAM_FRONT").glob("*.jpg"))
    if not camera_files:
        raise FileNotFoundError("No extracted CAM_FRONT image found under asset root.")

    camera_sd = next(sd for sd in nusc.sample_data if Path(sd["filename"]).name == camera_files[0].name)
    sample = nusc.get("sample", camera_sd["sample_token"])
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    image_path = asset_path(args.asset_root, camera_sd["filename"])
    lidar_path = asset_path(args.asset_root, lidar_sd["filename"])
    if not lidar_path.exists():
        raise FileNotFoundError(f"Matching LIDAR_TOP file is missing: {lidar_path}")

    lidar = LidarPointCloud.from_file(str(lidar_path))
    lidar_points = lidar.points[:3]
    render_lidar_topdown(lidar_points, args.output_dir / "lidar_topdown.png")
    boxes_projected = render_3d_boxes(nusc, camera_sd, image_path, args.output_dir / "camera_3d_to_2d_boxes.png")
    pixels, depth = project_sensor_to_camera(nusc, lidar_sd, camera_sd, lidar_points)
    lidar_projected = render_projection(image_path, pixels, depth, args.output_dir / "camera_lidar_overlay.png", "CAM_FRONT with projected LIDAR_TOP points")

    sensor_records, radar_rendered = [], 0
    for channel, token in sorted(sample["data"].items()):
        sd = nusc.get("sample_data", token)
        sensor = nusc.get("sensor", nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])["sensor_token"])
        path = asset_path(args.asset_root, sd["filename"])
        record = {"channel": channel, "modality": sensor["modality"], "filename": sd["filename"], "available": path.exists()}
        if sensor["modality"] == "radar" and path.exists():
            radar = RadarPointCloud.from_file(str(path))
            radar_pixels, radar_depth = project_sensor_to_camera(nusc, sd, camera_sd, radar.points[:3])
            record["projected_points"] = render_projection(image_path, radar_pixels, radar_depth, args.output_dir / f"camera_{channel.lower()}_overlay.png", f"CAM_FRONT with projected {channel} points")
            radar_rendered += 1
        sensor_records.append(record)

    summary = {
        "selected_scene": scene["name"],
        "location": log["location"],
        "sample_token": sample["token"],
        "camera_file": camera_sd["filename"],
        "lidar_file": lidar_sd["filename"],
        "lidar_points_raw": int(lidar_points.shape[1]),
        "lidar_points_projected": lidar_projected,
        "annotation_boxes_projected": boxes_projected,
        "scene_counts_by_location": dict(sorted(location_counts.items())),
        "sensors": sensor_records,
        "radar_overlays_rendered": radar_rendered,
        "note": "The local course package has no extracted Radar files. Radar calls and projection are implemented but no Radar image is generated without raw input.",
    }
    (args.output_dir / "devkit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
