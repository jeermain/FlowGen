from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackBundle:
    path: Path
    video: np.ndarray
    depths: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    coords: np.ndarray
    visibs: np.ndarray
    query_points: Optional[np.ndarray]

    @property
    def num_frames(self) -> int:
        return int(self.video.shape[0])

    @property
    def image_size(self) -> Tuple[int, int]:
        return int(self.video.shape[2]), int(self.video.shape[3])

    @property
    def num_tracks(self) -> int:
        return int(self.coords.shape[1])


@dataclass(frozen=True)
class MaskBundle:
    path: Path
    masks: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.masks.shape[0])

    @property
    def image_size(self) -> Tuple[int, int]:
        return int(self.masks.shape[1]), int(self.masks.shape[2])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    return np.load(path, allow_pickle=False)


def load_track_bundle(path: str | Path) -> TrackBundle:
    npz_path = Path(path).expanduser().resolve()
    data = _load_npz(npz_path)

    required = ["video", "depths", "intrinsics", "extrinsics", "coords", "visibs"]
    missing = [name for name in required if name not in data]
    _require(not missing, f"Track NPZ is missing keys: {missing}")

    video = np.asarray(data["video"])
    depths = np.asarray(data["depths"])
    intrinsics = np.asarray(data["intrinsics"])
    extrinsics = np.asarray(data["extrinsics"])
    coords = np.asarray(data["coords"])
    visibs = np.asarray(data["visibs"])
    query_points = np.asarray(data["query_points"]) if "query_points" in data else None

    _require(video.ndim == 4, f"Expected video shape (T, C, H, W), got {video.shape}")
    _require(video.shape[1] == 3, f"Expected 3 video channels, got {video.shape}")
    _require(depths.ndim == 3, f"Expected depths shape (T, H, W), got {depths.shape}")
    _require(intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3), f"Expected intrinsics shape (T, 3, 3), got {intrinsics.shape}")
    _require(extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4), f"Expected extrinsics shape (T, 4, 4), got {extrinsics.shape}")
    _require(coords.ndim == 3 and coords.shape[2] == 3, f"Expected coords shape (T, N, 3), got {coords.shape}")
    _require(visibs.ndim == 2, f"Expected visibs shape (T, N), got {visibs.shape}")

    num_frames = video.shape[0]
    _require(depths.shape[0] == num_frames, "Depth frame count does not match video frame count")
    _require(intrinsics.shape[0] == num_frames, "Intrinsics frame count does not match video frame count")
    _require(extrinsics.shape[0] == num_frames, "Extrinsics frame count does not match video frame count")
    _require(coords.shape[0] == num_frames, "Coords frame count does not match video frame count")
    _require(visibs.shape[0] == num_frames, "Visibs frame count does not match video frame count")
    _require(coords.shape[1] == visibs.shape[1], "Coords track count does not match visibs track count")

    if query_points is not None:
        _require(query_points.ndim == 2 and query_points.shape[0] == coords.shape[1], f"Expected query_points shape (N, D), got {query_points.shape}")

    return TrackBundle(
        path=npz_path,
        video=video.astype(np.float32, copy=False),
        depths=depths.astype(np.float32, copy=False),
        intrinsics=intrinsics.astype(np.float32, copy=False),
        extrinsics=extrinsics.astype(np.float32, copy=False),
        coords=coords.astype(np.float32, copy=False),
        visibs=visibs.astype(bool, copy=False),
        query_points=query_points.astype(np.float32, copy=False) if query_points is not None else None,
    )


def load_mask_bundle(path: str | Path) -> MaskBundle:
    npz_path = Path(path).expanduser().resolve()
    data = _load_npz(npz_path)
    _require("masks" in data, f"Mask NPZ does not contain 'masks': {npz_path}")

    masks = np.asarray(data["masks"])
    _require(masks.ndim == 3, f"Expected masks shape (T, H, W), got {masks.shape}")
    return MaskBundle(path=npz_path, masks=(masks > 0).astype(np.uint8, copy=False))


def align_masks_to_track(track: TrackBundle, mask_bundle: MaskBundle) -> np.ndarray:
    _require(mask_bundle.num_frames == track.num_frames, "Mask frame count does not match track frame count")

    target_h, target_w = track.image_size
    masks = mask_bundle.masks
    if mask_bundle.image_size == (target_h, target_w):
        return masks.astype(np.uint8, copy=False)

    resized = np.empty((track.num_frames, target_h, target_w), dtype=np.uint8)
    for frame_idx, mask in enumerate(masks):
        resized[frame_idx] = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized


def project_world_points(points_world: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _require(points_world.ndim == 3 and points_world.shape[2] == 3, f"Expected points shape (T, N, 3), got {points_world.shape}")
    _require(intrinsics.shape == (points_world.shape[0], 3, 3), "Intrinsics shape must match points frame count")
    _require(extrinsics.shape == (points_world.shape[0], 4, 4), "Extrinsics shape must match points frame count")

    ones = np.ones((*points_world.shape[:2], 1), dtype=points_world.dtype)
    points_h = np.concatenate([points_world, ones], axis=2)
    camera_points = np.einsum("tij,tnj->tni", extrinsics, points_h)[..., :3]
    depth = camera_points[..., 2]
    positive_depth = depth > 1e-6

    pixel_h = np.einsum("tij,tnj->tni", intrinsics, camera_points)
    denom = np.where(np.abs(pixel_h[..., 2]) > 1e-6, pixel_h[..., 2], 1.0)
    pixels = pixel_h[..., :2] / denom[..., None]
    return pixels.astype(np.float32, copy=False), positive_depth


def project_track_bundle(track: TrackBundle) -> Tuple[np.ndarray, np.ndarray]:
    return project_world_points(track.coords, track.intrinsics, track.extrinsics)


def project_first_frame_queries(track: TrackBundle) -> Tuple[np.ndarray, np.ndarray]:
    if track.query_points is not None and track.query_points.shape[1] >= 4:
        query_world = track.query_points[:, 1:4].astype(np.float32, copy=False)
    else:
        query_world = track.coords[0]

    pixels, positive = project_world_points(
        query_world[None, ...],
        track.intrinsics[:1],
        track.extrinsics[:1],
    )
    return pixels[0], positive[0]


def sample_mask_at_pixels(masks: np.ndarray, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _require(masks.ndim == 3, f"Expected masks shape (T, H, W), got {masks.shape}")
    _require(pixels.ndim == 3 and pixels.shape[2] == 2, f"Expected pixel shape (T, N, 2), got {pixels.shape}")
    _require(masks.shape[0] == pixels.shape[0], "Pixel frame count must match mask frame count")

    num_frames, height, width = masks.shape
    inside_mask = np.zeros((num_frames, pixels.shape[1]), dtype=bool)
    inside_image = np.zeros((num_frames, pixels.shape[1]), dtype=bool)

    rounded = np.rint(pixels).astype(np.int32)
    xs = rounded[..., 0]
    ys = rounded[..., 1]
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    inside_image[valid] = True

    for frame_idx in range(num_frames):
        frame_valid = valid[frame_idx]
        if not np.any(frame_valid):
            continue
        inside_mask[frame_idx, frame_valid] = masks[frame_idx, ys[frame_idx, frame_valid], xs[frame_idx, frame_valid]] > 0
    return inside_mask, inside_image
