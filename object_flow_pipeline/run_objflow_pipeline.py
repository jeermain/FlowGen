#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TAPIP3D + Grounded-SAM2 + object-flow filtering across conda environments.")
    parser.add_argument("--output-dir", required=True, help="Directory for all intermediate and final outputs.")
    parser.add_argument("--prompt", required=True, help="Grounded-SAM2 text prompt, e.g. 'mug'.")
    parser.add_argument("--track-npz", help="Use an existing TAPIP3D result NPZ and skip stage 1.")
    parser.add_argument("--input-path", help="Input video/npz for TAPIP3D stage when --track-npz is not provided.")
    parser.add_argument("--tapip3d-root", default="/mnt/nas/yangrun/TAPIP3D", help="Path to TAPIP3D repository.")
    parser.add_argument("--grounded-sam-root", default="/mnt/nas/yangrun/Grounded-SAM-2", help="Path to Grounded-SAM-2 repository.")
    parser.add_argument("--tapip3d-env", default="tapip3d", help="Conda environment name for TAPIP3D.")
    parser.add_argument("--grounded-sam-env", default="grounded-sam2", help="Conda environment name for Grounded-SAM2.")
    parser.add_argument("--postprocess-env", default="objflow-post", help="Conda environment name for post-processing scripts.")
    parser.add_argument("--tapip3d-checkpoint", default="/mnt/nas/yangrun/TAPIP3D/checkpoints/tapip3d_final.pth", help="TAPIP3D checkpoint path.")
    parser.add_argument("--resolution-factor", type=int, default=2, help="TAPIP3D resolution factor.")
    parser.add_argument("--temporal-threshold", type=float, default=0.6, help="Object-track temporal consistency threshold.")
    parser.add_argument("--mask-dilate", type=int, default=3, help="Mask dilation kernel size used during filtering.")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip the final comparison video.")
    return parser.parse_args()


def require_conda() -> str:
    conda_path = shutil.which("conda")
    if not conda_path:
        raise RuntimeError("Could not find `conda` in PATH. Open a shell with conda available and rerun.")
    return conda_path


def run_in_env(conda_path: str, env_name: str, command: list[str], cwd: Path | None = None) -> None:
    full_cmd = [conda_path, "run", "--no-capture-output", "-n", env_name, *command]
    print("$", " ".join(full_cmd))
    subprocess.run(full_cmd, cwd=str(cwd) if cwd else None, check=True)


def find_latest_track_npz(output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("*.result.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No TAPIP3D result NPZ found under: {output_dir}")
    return candidates[0]


def main() -> None:
    args = parse_args()
    conda_path = require_conda()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    tapip3d_root = Path(args.tapip3d_root).expanduser().resolve()
    grounded_sam_root = Path(args.grounded_sam_root).expanduser().resolve()

    if args.track_npz:
        track_npz = Path(args.track_npz).expanduser().resolve()
    else:
        if not args.input_path:
            raise ValueError("Either --track-npz or --input-path must be provided.")
        tapip3d_output_dir = output_dir / "tapip3d_outputs"
        tapip3d_output_dir.mkdir(parents=True, exist_ok=True)
        run_in_env(
            conda_path,
            args.tapip3d_env,
            [
                "python",
                str(tapip3d_root / "inference.py"),
                "--input_path",
                str(Path(args.input_path).expanduser().resolve()),
                "--checkpoint",
                str(Path(args.tapip3d_checkpoint).expanduser().resolve()),
                "--resolution_factor",
                str(args.resolution_factor),
                "--output_dir",
                str(tapip3d_output_dir),
            ],
            cwd=tapip3d_root,
        )
        track_npz = find_latest_track_npz(tapip3d_output_dir)

    mask_npz = output_dir / "segmentation_masks.npz"
    mask_video = output_dir / "segmentation_overlay.mp4"
    run_in_env(
        conda_path,
        args.grounded_sam_env,
        [
            "python",
            str(scripts_dir / "generate_grounded_sam2_masks.py"),
            "--track-npz",
            str(track_npz),
            "--prompt",
            args.prompt,
            "--output-mask-npz",
            str(mask_npz),
            "--output-video",
            str(mask_video),
            "--grounded-sam-root",
            str(grounded_sam_root),
        ],
        cwd=scripts_dir,
    )

    object_npz = output_dir / "object_tracks.npz"
    run_in_env(
        conda_path,
        args.postprocess_env,
        [
            "python",
            str(scripts_dir / "filter_object_tracks.py"),
            "--track-npz",
            str(track_npz),
            "--mask-npz",
            str(mask_npz),
            "--output-npz",
            str(object_npz),
            "--temporal-threshold",
            str(args.temporal_threshold),
            "--mask-dilate",
            str(args.mask_dilate),
        ],
        cwd=scripts_dir,
    )

    if not args.skip_visualization:
        visualization_path = output_dir / "object_tracks_compare.mp4"
        run_in_env(
            conda_path,
            args.postprocess_env,
            [
                "python",
                str(scripts_dir / "visualize_object_tracks.py"),
                "--track-npz",
                str(track_npz),
                "--object-npz",
                str(object_npz),
                "--output-video",
                str(visualization_path),
            ],
            cwd=scripts_dir,
        )
        print(f"Saved comparison video to: {visualization_path}")

    print(f"Track NPZ: {track_npz}")
    print(f"Mask NPZ: {mask_npz}")
    print(f"Object NPZ: {object_npz}")


if __name__ == "__main__":
    main()
