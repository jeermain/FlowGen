#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from objflow_schema import (
    align_masks_to_track,
    load_mask_bundle,
    load_track_bundle,
    project_first_frame_queries,
    project_track_bundle,
    sample_mask_at_pixels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter TAPIP3D tracks with Grounded-SAM2 masks.")
    parser.add_argument("--track-npz", required=True, help="Input TAPIP3D result NPZ.")
    parser.add_argument("--mask-npz", required=True, help="Input mask NPZ with masks=(T,H,W).")
    parser.add_argument("--output-npz", required=True, help="Output NPZ for object-only tracks.")
    parser.add_argument("--temporal-threshold", type=float, default=0.6, help="Minimum fraction of visible frames that must stay inside the mask.")
    parser.add_argument("--min-visible-frames", type=int, default=3, help="Minimum number of visible frames required for a track.")
    parser.add_argument("--min-inside-frames", type=int, default=2, help="Minimum number of frames inside the mask required for a track.")
    parser.add_argument("--mask-dilate", type=int, default=3, help="Optional dilation kernel size applied to masks before filtering. Use 0 to disable.")
    return parser.parse_args()


def dilate_masks(masks: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return masks
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return np.stack([cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) for mask in masks], axis=0)


def sample_single_mask(mask: np.ndarray, pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    rounded = np.rint(pixels).astype(np.int32)
    xs = rounded[:, 0]
    ys = rounded[:, 1]
    in_bounds = valid & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    sampled = np.zeros(pixels.shape[0], dtype=bool)
    sampled[in_bounds] = mask[ys[in_bounds], xs[in_bounds]] > 0
    return sampled


def main() -> None:
    args = parse_args()

    track = load_track_bundle(args.track_npz)
    masks = align_masks_to_track(track, load_mask_bundle(args.mask_npz))
    masks = dilate_masks(masks, args.mask_dilate)

    query_pixels, query_valid = project_first_frame_queries(track)
    first_inside = sample_single_mask(masks[0], query_pixels, query_valid) & track.visibs[0]

    projected_pixels, positive_depth = project_track_bundle(track)
    inside_mask, inside_image = sample_mask_at_pixels(masks, projected_pixels)

    visible_and_projectable = track.visibs & positive_depth & inside_image
    inside_mask_visible = inside_mask & visible_and_projectable

    visible_counts = visible_and_projectable.sum(axis=0)
    inside_counts = inside_mask_visible.sum(axis=0)
    temporal_ratio = np.divide(
        inside_counts,
        np.maximum(visible_counts, 1),
        out=np.zeros_like(inside_counts, dtype=np.float32),
        where=np.maximum(visible_counts, 1) > 0,
    ).astype(np.float32)

    keep = (
        first_inside
        & (visible_counts >= args.min_visible_frames)
        & (inside_counts >= args.min_inside_frames)
        & (temporal_ratio >= args.temporal_threshold)
    )

    output_path = Path(args.output_npz).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords_object = track.coords[:, keep]
    visibs_object = inside_mask_visible[:, keep]
    query_points_object = track.query_points[keep] if track.query_points is not None else np.zeros((coords_object.shape[1], 0), dtype=np.float32)
    projected_pixels_object = projected_pixels[:, keep]

    np.savez(
        output_path,
        schema_version=np.array("objflow_v1"),
        source_track_npz=np.array(str(track.path)),
        source_mask_npz=np.array(str(Path(args.mask_npz).expanduser().resolve())),
        video=track.video,
        depths=track.depths,
        intrinsics=track.intrinsics,
        extrinsics=track.extrinsics,
        masks=masks.astype(np.uint8),
        coords_object=coords_object.astype(np.float32),
        visibs_object=visibs_object.astype(bool),
        query_points_object=query_points_object.astype(np.float32),
        object_indices=np.flatnonzero(keep).astype(np.int32),
        frame0_membership=first_inside.astype(bool),
        temporal_ratio=temporal_ratio.astype(np.float32),
        visible_counts=visible_counts.astype(np.int32),
        inside_counts=inside_counts.astype(np.int32),
        projected_pixels_object=projected_pixels_object.astype(np.float32),
        inside_mask_object=inside_mask_visible[:, keep].astype(bool),
    )

    kept = int(keep.sum())
    total = int(track.num_tracks)
    print(f"Saved filtered object tracks to: {output_path}")
    print(f"Kept {kept}/{total} tracks ({(100.0 * kept / max(total, 1)):.2f}%).")


if __name__ == "__main__":
    main()
