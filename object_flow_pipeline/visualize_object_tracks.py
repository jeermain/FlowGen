#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from objflow_schema import load_track_bundle, project_track_bundle, project_world_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a side-by-side visualization for all vs object-only tracks.")
    parser.add_argument("--track-npz", required=True, help="Input TAPIP3D result NPZ.")
    parser.add_argument("--object-npz", required=True, help="Filtered object track NPZ.")
    parser.add_argument("--output-video", required=True, help="Output comparison video path.")
    parser.add_argument("--max-all-points", type=int, default=512, help="Max points drawn on the all-track view.")
    parser.add_argument("--max-object-points", type=int, default=256, help="Max points drawn on the object-only view.")
    parser.add_argument("--fps", type=float, default=8.0, help="Visualization FPS.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for point subsampling.")
    return parser.parse_args()


def draw_points(frame: np.ndarray, pixels: np.ndarray, visible: np.ndarray, color: tuple[int, int, int], radius: int = 2) -> np.ndarray:
    canvas = frame.copy()
    rounded = np.rint(pixels).astype(np.int32)
    height, width = frame.shape[:2]
    for (x, y), is_visible in zip(rounded, visible):
        if not is_visible or x < 0 or y < 0 or x >= width or y >= height:
            continue
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    frame = frame.copy()
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    overlay = frame.copy()
    overlay[mask > 0] = (0, 180, 0)
    return cv2.addWeighted(frame, 0.75, overlay, 0.25, 0.0)


def choose_indices(count: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int32)
    return np.sort(rng.choice(count, size=limit, replace=False).astype(np.int32))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    track = load_track_bundle(args.track_npz)
    object_data = np.load(Path(args.object_npz).expanduser().resolve(), allow_pickle=False)

    required = ["coords_object", "visibs_object", "masks"]
    missing = [key for key in required if key not in object_data]
    if missing:
        raise ValueError(f"Object NPZ is missing keys: {missing}")

    coords_object = np.asarray(object_data["coords_object"]).astype(np.float32, copy=False)
    visibs_object = np.asarray(object_data["visibs_object"]).astype(bool, copy=False)
    masks = np.asarray(object_data["masks"]).astype(np.uint8, copy=False)

    all_pixels, all_positive_depth = project_track_bundle(track)
    all_visible = track.visibs & all_positive_depth

    object_pixels, object_positive_depth = project_world_points(coords_object, track.intrinsics, track.extrinsics)
    object_visible = visibs_object & object_positive_depth

    all_keep = choose_indices(track.num_tracks, args.max_all_points, rng)
    obj_keep = choose_indices(coords_object.shape[1], args.max_object_points, rng)

    rgb_video = np.clip(np.transpose(track.video, (0, 2, 3, 1)) * 255.0, 0, 255).astype(np.uint8)
    height, width = rgb_video.shape[1:3]

    output_path = Path(args.output_video).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for frame_idx, frame_rgb in enumerate(rgb_video):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            mask = masks[frame_idx] if frame_idx < len(masks) else np.zeros((height, width), dtype=np.uint8)

            left = overlay_mask(frame_bgr, mask)
            left = draw_points(left, all_pixels[frame_idx, all_keep], all_visible[frame_idx, all_keep], (255, 255, 0), radius=1)
            cv2.putText(left, f"all tracks ({len(all_keep)}/{track.num_tracks})", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            right = overlay_mask(frame_bgr, mask)
            right = draw_points(right, object_pixels[frame_idx, obj_keep], object_visible[frame_idx, obj_keep], (0, 255, 255), radius=2)
            cv2.putText(right, f"object tracks ({len(obj_keep)}/{coords_object.shape[1]})", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            combined = np.concatenate([left, right], axis=1)
            cv2.putText(combined, f"frame={frame_idx}", (width - 70, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(combined)
    finally:
        writer.release()

    print(f"Saved object flow visualization to: {output_path}")


if __name__ == "__main__":
    main()
