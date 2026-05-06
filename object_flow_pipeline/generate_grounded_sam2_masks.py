#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

from objflow_schema import load_track_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-frame binary masks with Grounded-SAM2.")
    parser.add_argument("--track-npz", required=True, help="TAPIP3D track result NPZ.")
    parser.add_argument("--prompt", required=True, help="Text prompt for Grounding DINO, e.g. 'mug'.")
    parser.add_argument("--output-mask-npz", required=True, help="Output NPZ path for binary masks.")
    parser.add_argument("--output-video", help="Optional output MP4 path for mask visualization.")
    parser.add_argument("--grounded-sam-root", default="/mnt/nas/yangrun/Grounded-SAM-2", help="Path to Grounded-SAM-2 repository.")
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_large.pt", help="SAM2 checkpoint path, absolute or relative to --grounded-sam-root.")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 Hydra config name inside the Grounded-SAM-2 repo.")
    parser.add_argument("--grounding-dino-config", default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Grounding DINO config path, absolute or relative to --grounded-sam-root.")
    parser.add_argument("--grounding-dino-checkpoint", default="gdino_checkpoints/groundingdino_swint_ogc.pth", help="Grounding DINO checkpoint path, absolute or relative to --grounded-sam-root.")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="Grounding DINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.3, help="Grounding DINO text threshold.")
    parser.add_argument("--keep-all-detections", action="store_true", help="Use all detections instead of only the top-scoring one.")
    return parser.parse_args()


def ensure_trailing_period(prompt: str) -> str:
    prompt = prompt.strip()
    return prompt if prompt.endswith(".") else f"{prompt}."


def resolve_repo_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def prepare_grounded_sam_imports(grounded_sam_root: Path):
    import os
    import sys

    grounded_sam_root = grounded_sam_root.resolve()
    if not grounded_sam_root.is_dir():
        raise FileNotFoundError(f"Grounded-SAM2 root not found: {grounded_sam_root}")

    if str(grounded_sam_root) not in sys.path:
        sys.path.insert(0, str(grounded_sam_root))

    os.chdir(grounded_sam_root)
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import Model, load_model, predict

    return build_sam2_video_predictor, build_sam2, SAM2ImagePredictor, Model, load_model, predict


def save_track_frames(track_video: np.ndarray, frame_dir: Path) -> None:
    rgb_video = np.clip(np.transpose(track_video, (0, 2, 3, 1)) * 255.0, 0, 255).astype(np.uint8)
    for frame_idx, frame_rgb in enumerate(rgb_video):
        frame_path = frame_dir / f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def create_overlay_video(track_video: np.ndarray, masks: np.ndarray, output_path: Path) -> None:
    rgb_video = np.clip(np.transpose(track_video, (0, 2, 3, 1)) * 255.0, 0, 255).astype(np.uint8)
    height, width = rgb_video.shape[1:3]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for frame_idx, (frame_rgb, mask) in enumerate(zip(rgb_video, masks)):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            overlay = frame_bgr.copy()
            overlay[mask > 0] = (0, 255, 0)
            blended = cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0.0)
            cv2.putText(blended, f"frame={frame_idx}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(blended)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    prompt = ensure_trailing_period(args.prompt)
    track = load_track_bundle(args.track_npz)
    grounded_sam_root = Path(args.grounded_sam_root).expanduser()
    sam2_checkpoint = resolve_repo_path(args.sam2_checkpoint, grounded_sam_root)
    grounding_dino_config = resolve_repo_path(args.grounding_dino_config, grounded_sam_root)
    grounding_dino_checkpoint = resolve_repo_path(args.grounding_dino_checkpoint, grounded_sam_root)
    output_mask_npz = Path(args.output_mask_npz).expanduser().resolve()
    output_mask_npz.parent.mkdir(parents=True, exist_ok=True)

    if not sam2_checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint}")
    if not grounding_dino_config.is_file():
        raise FileNotFoundError(f"Grounding DINO config not found: {grounding_dino_config}")
    if not grounding_dino_checkpoint.is_file():
        raise FileNotFoundError(f"Grounding DINO checkpoint not found: {grounding_dino_checkpoint}")

    build_sam2_video_predictor, build_sam2, sam2_image_predictor_cls, grounding_model_cls, load_grounding_model, grounding_predict = prepare_grounded_sam_imports(grounded_sam_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    sam2_autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    video_predictor = build_sam2_video_predictor(args.sam2_config, str(sam2_checkpoint))
    sam2_image_model = build_sam2(args.sam2_config, str(sam2_checkpoint))
    sam2_image_model.to(device)
    image_predictor = sam2_image_predictor_cls(sam2_image_model)
    grounding_model = load_grounding_model(
        model_config_path=str(grounding_dino_config),
        model_checkpoint_path=str(grounding_dino_checkpoint),
        device=device,
    )

    with tempfile.TemporaryDirectory(prefix="objflow_gsam2_") as temp_dir:
        frame_dir = Path(temp_dir) / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        save_track_frames(track.video, frame_dir)

        inference_state = video_predictor.init_state(video_path=str(frame_dir))
        first_frame = np.clip(np.transpose(track.video[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)

        processed_image = grounding_model_cls.preprocess_image(
            image_bgr=cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        ).to(device)
        boxes, scores_tensor, labels = grounding_predict(
            model=grounding_model,
            image=processed_image,
            caption=prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=device,
        )

        if len(boxes) == 0:
            raise RuntimeError(f"No detections found for prompt: {prompt}")

        height, width = first_frame.shape[:2]
        all_boxes = box_convert(
            boxes=boxes * torch.tensor([width, height, width, height]),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ).cpu().numpy()

        if args.keep_all_detections:
            input_boxes = all_boxes
            scores = scores_tensor.cpu().numpy()
            labels = list(labels)
        else:
            max_idx = int(torch.argmax(scores_tensor).item())
            input_boxes = all_boxes[max_idx:max_idx + 1]
            scores = np.array([scores_tensor[max_idx].item()], dtype=np.float32)
            labels = [labels[max_idx]]

        with sam2_autocast:
            image_predictor.set_image(first_frame)
            masks, _, _ = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        elif masks.ndim == 2:
            masks = masks[None]

        with sam2_autocast:
            for object_id, box in enumerate(input_boxes, start=1):
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=object_id,
                    box=box,
                )

        video_segments = {}
        with sam2_autocast:
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        height, width = track.image_size
        binary_masks = np.zeros((track.num_frames, height, width), dtype=np.uint8)
        for frame_idx in range(track.num_frames):
            if frame_idx not in video_segments:
                continue
            frame_mask = np.zeros((height, width), dtype=bool)
            for mask in video_segments[frame_idx].values():
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                frame_mask |= mask
            binary_masks[frame_idx] = frame_mask.astype(np.uint8)

    np.savez(
        output_mask_npz,
        masks=binary_masks,
        prompt=np.array(prompt),
        scores=scores.astype(np.float32),
        labels=np.array(labels),
        boxes=input_boxes.astype(np.float32),
        source_track_npz=np.array(str(track.path)),
    )

    if args.output_video:
        create_overlay_video(track.video, binary_masks, Path(args.output_video).expanduser().resolve())

    print(f"Saved masks to: {output_mask_npz}")
    if args.output_video:
        print(f"Saved mask visualization to: {Path(args.output_video).expanduser().resolve()}")


if __name__ == "__main__":
    main()
