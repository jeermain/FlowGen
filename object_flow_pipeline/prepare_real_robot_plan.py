#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_WORKSPACE_MIN = np.array([-1.0, -1.0, -0.1], dtype=np.float32)
DEFAULT_WORKSPACE_MAX = np.array([1.0, 1.0, 1.2], dtype=np.float32)


@dataclass(frozen=True)
class SafetyReport:
    ok: bool
    messages: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a dry-run real-robot execution plan from object_flow_pipeline object_tracks.npz."
    )
    parser.add_argument("--object-npz", required=True, help="Filtered object_tracks.npz from filter_object_tracks.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated pose trajectories and reports.")
    parser.add_argument(
        "--calibration",
        help=(
            "Optional transform file for visual-world to robot-base conversion. Supports .npy/.npz 4x4 matrices "
            "or JSON containing T_base_world/T_base_camera/transform/matrix."
        ),
    )
    parser.add_argument(
        "--grasp-pose",
        help=(
            "Optional 4x4 grasp transform file. Interpreted as T_obj_grasp by default. "
            "Supports .npy/.npz or JSON with T_obj_grasp/grasp_pose/transform/matrix."
        ),
    )
    parser.add_argument(
        "--grasp-pose-frame",
        choices=["object", "world", "base"],
        default="object",
        help="Frame of --grasp-pose. Use object for T_obj_grasp, world/base for absolute first-frame EE pose.",
    )
    parser.add_argument("--pregrasp-offset", type=float, default=0.08, help="Meters to retreat along EE approach axis.")
    parser.add_argument("--max-frame-translation", type=float, default=0.15, help="Max allowed EE translation jump per frame in meters.")
    parser.add_argument("--min-points", type=int, default=3, help="Minimum common visible object points for Kabsch.")
    parser.add_argument("--max-rms", type=float, default=0.06, help="Max Kabsch residual in meters before a frame is rejected.")
    parser.add_argument("--max-object-translation", type=float, default=0.15, help="Max allowed object pose translation jump per frame in meters.")
    parser.add_argument(
        "--reference-mode",
        choices=["first", "previous"],
        default="previous",
        help="Use the first frame or last valid frame as the Kabsch reference.",
    )
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=3,
        default=DEFAULT_WORKSPACE_MIN.tolist(),
        metavar=("X", "Y", "Z"),
        help="Robot-base workspace lower bound in meters.",
    )
    parser.add_argument(
        "--workspace-max",
        type=float,
        nargs=3,
        default=DEFAULT_WORKSPACE_MAX.tolist(),
        metavar=("X", "Y", "Z"),
        help="Robot-base workspace upper bound in meters.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Require a real controller adapter. Without an adapter this exits before motion; dry-run is the default.",
    )
    parser.add_argument(
        "--controller-adapter",
        help=(
            "Optional Python adapter as /path/to/file.py:function_name. The function receives a context dict "
            "with plan/report/command paths and must perform hardware-specific IK, planning, and execution."
        ),
    )
    parser.add_argument("--max-linear-speed", type=float, default=0.05, help="Dry-run Cartesian speed limit in m/s.")
    parser.add_argument("--max-angular-speed", type=float, default=0.25, help="Dry-run angular speed limit in rad/s.")
    return parser.parse_args()


def require_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def load_matrix(path: str | None, *, keys: tuple[str, ...], default: np.ndarray, name: str) -> np.ndarray:
    if path is None:
        return require_matrix(name, default)

    matrix_path = Path(path).expanduser().resolve()
    if not matrix_path.is_file():
        raise FileNotFoundError(f"{name} file not found: {matrix_path}")

    if matrix_path.suffix == ".npy":
        return require_matrix(name, np.load(matrix_path))

    if matrix_path.suffix == ".npz":
        data = np.load(matrix_path)
        for key in keys:
            if key in data:
                return require_matrix(name, data[key])
        if len(data.files) == 1:
            return require_matrix(name, data[data.files[0]])
        raise KeyError(f"{name} npz must contain one of {keys}, got {data.files}")

    with matrix_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    for key in keys:
        if key in payload:
            return require_matrix(name, payload[key])
    if isinstance(payload, list):
        return require_matrix(name, payload)
    raise KeyError(f"{name} JSON must contain one of {keys} or be a 4x4 list")


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = translation.astype(np.float32)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = (transform @ homogeneous.T).T[:, :3]
    return transformed.reshape(points.shape)


def kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid

    aligned = (rotation @ source.T).T + translation
    rms = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return make_transform(rotation, translation), rms


def initial_object_pose(coords_first: np.ndarray, visibs_first: np.ndarray) -> np.ndarray:
    points = coords_first[visibs_first.astype(bool)]
    if len(points) < 3:
        raise ValueError("Need at least 3 visible points to initialize the object frame")

    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    x_axis = vt[0]
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = vt[1]
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    return make_transform(np.stack([x_axis, y_axis, z_axis], axis=1), centroid)


def estimate_object_pose_sequence(
    coords: np.ndarray,
    visibs: np.ndarray,
    *,
    min_points: int,
    max_rms: float,
    max_object_translation: float,
    reference_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(f"coords_object must have shape (T, M, 3), got {coords.shape}")
    if visibs.shape != coords.shape[:2]:
        raise ValueError(f"visibs_object must have shape {coords.shape[:2]}, got {visibs.shape}")

    num_frames = coords.shape[0]
    reference_points = coords[0]
    reference_visible = visibs[0].astype(bool)
    object_pose0 = initial_object_pose(coords[0], reference_visible)
    poses = np.tile(object_pose0.astype(np.float32), (num_frames, 1, 1))
    valid = np.zeros(num_frames, dtype=bool)
    residuals = np.full(num_frames, np.inf, dtype=np.float32)
    valid[0] = bool(reference_visible.sum() >= min_points)
    residuals[0] = 0.0

    last_pose = object_pose0
    last_valid_frame = 0
    for frame_idx in range(1, num_frames):
        if reference_mode == "previous":
            reference_points = coords[last_valid_frame]
            reference_visible = visibs[last_valid_frame].astype(bool)
        common = reference_visible & visibs[frame_idx].astype(bool)
        if int(common.sum()) < min_points:
            poses[frame_idx] = last_pose
            continue

        transform, rms = kabsch_transform(reference_points[common], coords[frame_idx, common])
        if reference_mode == "previous":
            transform = transform @ last_pose
        else:
            transform = transform @ object_pose0
        residuals[frame_idx] = rms
        translation_jump = float(np.linalg.norm(transform[:3, 3] - last_pose[:3, 3]))
        if rms <= max_rms and translation_jump <= max_object_translation:
            poses[frame_idx] = transform
            valid[frame_idx] = True
            last_pose = transform
            last_valid_frame = frame_idx
        else:
            poses[frame_idx] = last_pose

    return poses, valid, residuals


def default_object_grasp(coords_first: np.ndarray, visibs_first: np.ndarray, first_object_pose_world: np.ndarray) -> np.ndarray:
    points = coords_first[visibs_first.astype(bool)]
    if len(points) < 3:
        raise ValueError("Need at least 3 visible points to infer a default grasp pose")

    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    x_axis = vt[0]
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = vt[1]
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    world_grasp = make_transform(np.stack([x_axis, y_axis, z_axis], axis=1), centroid)
    return invert_transform(first_object_pose_world) @ world_grasp


def resolve_object_grasp(
    grasp_pose_path: str | None,
    frame: str,
    first_object_pose_world: np.ndarray,
    t_base_world: np.ndarray,
    coords_first: np.ndarray,
    visibs_first: np.ndarray,
) -> np.ndarray:
    if grasp_pose_path is None:
        return default_object_grasp(coords_first, visibs_first, first_object_pose_world)

    grasp = load_matrix(
        grasp_pose_path,
        keys=("T_obj_grasp", "grasp_pose", "transform", "matrix"),
        default=np.eye(4, dtype=np.float32),
        name="grasp pose",
    )
    if frame == "object":
        return grasp
    if frame == "world":
        return invert_transform(first_object_pose_world) @ grasp
    if frame == "base":
        t_world_base = invert_transform(t_base_world)
        return invert_transform(first_object_pose_world) @ t_world_base @ grasp
    raise ValueError(f"Unsupported grasp pose frame: {frame}")


def make_pregrasp(grasp_pose: np.ndarray, offset: float) -> np.ndarray:
    pregrasp = grasp_pose.copy()
    approach_axis = grasp_pose[:3, 2]
    pregrasp[:3, 3] = grasp_pose[:3, 3] - approach_axis * offset
    return pregrasp


def build_ee_trajectory(
    object_poses_world: np.ndarray,
    t_base_world: np.ndarray,
    t_obj_grasp: np.ndarray,
    pregrasp_offset: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    object_poses_base = np.einsum("ij,tjk->tik", t_base_world, object_poses_world)
    ee_poses_base = np.einsum("tij,jk->tik", object_poses_base, t_obj_grasp)
    waypoints = {
        "pre_grasp": make_pregrasp(ee_poses_base[0], pregrasp_offset),
        "grasp": ee_poses_base[0],
        "goal": ee_poses_base[-1],
        "retreat": make_pregrasp(ee_poses_base[-1], pregrasp_offset),
    }
    return ee_poses_base, waypoints


def safety_check(
    ee_poses_base: np.ndarray,
    waypoints: dict[str, np.ndarray],
    *,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    max_frame_translation: float,
) -> SafetyReport:
    messages: list[str] = []
    positions = ee_poses_base[:, :3, 3]
    waypoint_positions = np.stack([pose[:3, 3] for pose in waypoints.values()], axis=0)
    all_positions = np.concatenate([positions, waypoint_positions], axis=0)

    outside = np.any((all_positions < workspace_min) | (all_positions > workspace_max), axis=1)
    if np.any(outside):
        messages.append("One or more EE waypoints are outside the configured workspace bounds.")

    if len(positions) > 1:
        jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        max_jump = float(jumps.max())
        if max_jump > max_frame_translation:
            messages.append(f"Max EE frame-to-frame translation {max_jump:.4f}m exceeds {max_frame_translation:.4f}m.")

    for idx, pose in enumerate(ee_poses_base):
        det = float(np.linalg.det(pose[:3, :3]))
        if not np.isfinite(det) or abs(det - 1.0) > 0.05:
            messages.append(f"EE pose {idx} has invalid rotation determinant {det:.4f}.")
            break

    return SafetyReport(ok=len(messages) == 0, messages=messages)


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def write_report(
    output_dir: Path,
    *,
    t_base_world: np.ndarray,
    t_obj_grasp: np.ndarray,
    object_valid: np.ndarray,
    residuals: np.ndarray,
    waypoints: dict[str, np.ndarray],
    safety: SafetyReport,
    execute_requested: bool,
) -> None:
    report: dict[str, Any] = {
        "dry_run": not execute_requested,
        "ready_for_execution": safety.ok and not execute_requested,
        "execution_blocked_reason": (
            "Real robot execution requires a hardware-specific controller adapter; this script only emits dry-run plans."
            if execute_requested
            else None
        ),
        "T_base_world": matrix_to_list(t_base_world),
        "T_obj_grasp": matrix_to_list(t_obj_grasp),
        "valid_object_pose_frames": int(object_valid.sum()),
        "total_frames": int(len(object_valid)),
        "max_kabsch_rms": float(np.nanmax(np.where(np.isfinite(residuals), residuals, np.nan))),
        "waypoints": {name: matrix_to_list(pose) for name, pose in waypoints.items()},
        "safety_ok": safety.ok,
        "safety_messages": safety.messages,
        "staged_validation": [
            "Inspect object_poses_base and ee_poses_base offline.",
            "Run robot dry-run/ghost trajectory with the saved waypoints.",
            "Execute pre_grasp to retreat at low speed with no object contact.",
            "Execute pre_grasp to grasp, then verify gripper closure and object motion.",
            "Execute grasp to goal only after collision checks pass.",
            "Use fresh perception before each segment for closed-loop correction.",
        ],
        "closed_loop_hooks": {
            "before_grasp": "re-segment target and regenerate pre_grasp/grasp if the object moved",
            "after_grasp": "verify gripper width/force and object follows the EE",
            "between_segments": "re-run perception or compare projected object flow before continuing",
        },
    }
    with (output_dir / "real_robot_plan_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_robot_commands(
    output_dir: Path,
    *,
    waypoints: dict[str, np.ndarray],
    max_linear_speed: float,
    max_angular_speed: float,
) -> Path:
    commands: dict[str, Any] = {
        "schema_version": "objflow_real_robot_dry_run_v1",
        "frame": "robot_base",
        "constraints": {
            "max_linear_speed_mps": float(max_linear_speed),
            "max_angular_speed_radps": float(max_angular_speed),
            "requires_collision_check": True,
            "requires_ik": True,
            "requires_operator_confirm": True,
        },
        "sequence": [
            {"type": "move_cartesian", "name": "pre_grasp", "target_pose": matrix_to_list(waypoints["pre_grasp"])},
            {"type": "move_cartesian", "name": "grasp", "target_pose": matrix_to_list(waypoints["grasp"])},
            {"type": "gripper", "name": "close_gripper", "command": "close"},
            {"type": "move_cartesian", "name": "goal", "target_pose": matrix_to_list(waypoints["goal"])},
            {"type": "move_cartesian", "name": "retreat", "target_pose": matrix_to_list(waypoints["retreat"])},
        ],
    }
    command_path = output_dir / "robot_command_sequence.json"
    with command_path.open("w", encoding="utf-8") as f:
        json.dump(commands, f, indent=2)
    return command_path


def write_validation_artifacts(output_dir: Path) -> tuple[Path, Path]:
    staged_validation = {
        "schema_version": "objflow_real_robot_validation_v1",
        "stages": [
            {
                "name": "offline_visualization",
                "goal": "Inspect object_poses_base and ee_poses_base without robot motion.",
                "pass_condition": "Poses are finite, inside workspace, and visually align with the object.",
            },
            {
                "name": "ghost_or_dry_run",
                "goal": "Send the command sequence to the robot planner without enabling actuators.",
                "pass_condition": "IK and collision checks pass for every waypoint.",
            },
            {
                "name": "pregrasp_retreat",
                "goal": "Move between pre_grasp and retreat at low speed with no object contact.",
                "pass_condition": "No collision, joint-limit, or workspace violation occurs.",
            },
            {
                "name": "static_grasp",
                "goal": "Execute pre_grasp to grasp and close the gripper.",
                "pass_condition": "The gripper closes on the object and lift remains stable.",
            },
            {
                "name": "grasp_to_goal",
                "goal": "Execute the post-grasp goal waypoint.",
                "pass_condition": "Object follows the gripper without slip and reaches the target region.",
            },
            {
                "name": "full_flow",
                "goal": "Track the smoothed end-effector trajectory from object flow.",
                "pass_condition": "The robot completes the motion within configured safety bounds.",
            },
        ],
    }
    closed_loop = {
        "schema_version": "objflow_closed_loop_hooks_v1",
        "hooks": [
            {
                "name": "before_grasp_relocalize",
                "trigger": "Immediately before moving from pre_grasp to grasp.",
                "action": "Capture a fresh RGB-D frame, re-run segmentation, and update grasp if object center shifted.",
            },
            {
                "name": "after_grasp_verify",
                "trigger": "After gripper close and before lift.",
                "action": "Check gripper width/force and optionally verify target mask moves with the gripper.",
            },
            {
                "name": "between_segments_replan",
                "trigger": "Before executing goal or any long Cartesian segment.",
                "action": "Re-estimate object pose and regenerate remaining waypoints if error exceeds tolerance.",
            },
            {
                "name": "safety_stop",
                "trigger": "IK failure, collision risk, tracking loss, or workspace violation.",
                "action": "Stop execution, retreat if safe, and request operator confirmation.",
            },
        ],
    }
    validation_path = output_dir / "staged_validation_checklist.json"
    closed_loop_path = output_dir / "closed_loop_hooks.json"
    with validation_path.open("w", encoding="utf-8") as f:
        json.dump(staged_validation, f, indent=2)
    with closed_loop_path.open("w", encoding="utf-8") as f:
        json.dump(closed_loop, f, indent=2)
    return validation_path, closed_loop_path


def run_controller_adapter(adapter_spec: str, context: dict[str, str]) -> None:
    if ":" not in adapter_spec:
        raise ValueError("--controller-adapter must have format /path/to/file.py:function_name")
    module_path_str, function_name = adapter_spec.split(":", 1)
    module_path = Path(module_path_str).expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"Controller adapter not found: {module_path}")

    spec = importlib.util.spec_from_file_location("objflow_controller_adapter", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import controller adapter: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    execute_fn = getattr(module, function_name)
    execute_fn(context)


def main() -> None:
    args = parse_args()
    object_npz = Path(args.object_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(object_npz)
    coords = np.asarray(data["coords_object"], dtype=np.float32)
    visibs = np.asarray(data["visibs_object"], dtype=bool)

    t_base_world = load_matrix(
        args.calibration,
        keys=("T_base_world", "T_base_camera", "transform", "matrix"),
        default=np.eye(4, dtype=np.float32),
        name="calibration transform",
    )
    object_poses_world, object_valid, residuals = estimate_object_pose_sequence(
        coords,
        visibs,
        min_points=args.min_points,
        max_rms=args.max_rms,
        max_object_translation=args.max_object_translation,
        reference_mode=args.reference_mode,
    )
    t_obj_grasp = resolve_object_grasp(
        args.grasp_pose,
        args.grasp_pose_frame,
        object_poses_world[0],
        t_base_world,
        coords[0],
        visibs[0],
    )
    ee_poses_base, waypoints = build_ee_trajectory(
        object_poses_world,
        t_base_world,
        t_obj_grasp,
        args.pregrasp_offset,
    )
    object_poses_base = np.einsum("ij,tjk->tik", t_base_world, object_poses_world)
    safety = safety_check(
        ee_poses_base,
        waypoints,
        workspace_min=np.asarray(args.workspace_min, dtype=np.float32),
        workspace_max=np.asarray(args.workspace_max, dtype=np.float32),
        max_frame_translation=args.max_frame_translation,
    )

    np.savez(
        output_dir / "real_robot_plan.npz",
        source_object_npz=np.array(str(object_npz)),
        T_base_world=t_base_world.astype(np.float32),
        T_obj_grasp=t_obj_grasp.astype(np.float32),
        object_poses_world=object_poses_world.astype(np.float32),
        object_poses_base=object_poses_base.astype(np.float32),
        ee_poses_base=ee_poses_base.astype(np.float32),
        object_pose_valid=object_valid.astype(bool),
        kabsch_rms=residuals.astype(np.float32),
        waypoint_names=np.array(list(waypoints.keys())),
        waypoint_poses=np.stack(list(waypoints.values()), axis=0).astype(np.float32),
    )
    plan_path = output_dir / "real_robot_plan.npz"
    report_path = output_dir / "real_robot_plan_report.json"
    command_path = write_robot_commands(
        output_dir,
        waypoints=waypoints,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
    )
    validation_path, closed_loop_path = write_validation_artifacts(output_dir)
    write_report(
        output_dir,
        t_base_world=t_base_world,
        t_obj_grasp=t_obj_grasp,
        object_valid=object_valid,
        residuals=residuals,
        waypoints=waypoints,
        safety=safety,
        execute_requested=args.execute,
    )

    print(f"Saved real-robot dry-run plan to: {output_dir / 'real_robot_plan.npz'}")
    print(f"Saved validation report to: {report_path}")
    print(f"Saved dry-run robot command sequence to: {command_path}")
    print(f"Saved staged validation checklist to: {validation_path}")
    print(f"Saved closed-loop hooks to: {closed_loop_path}")
    print(f"Valid object pose frames: {int(object_valid.sum())}/{len(object_valid)}")
    if safety.ok:
        print("Safety checks passed for configured workspace and trajectory jumps.")
    else:
        print("Safety checks failed:")
        for message in safety.messages:
            print(f"- {message}")

    if args.execute:
        if not args.controller_adapter:
            raise RuntimeError(
                "Real robot execution requires --controller-adapter /path/to/file.py:function_name after dry-run validation."
            )
        if not safety.ok:
            raise RuntimeError("Refusing execution because safety checks failed.")
        run_controller_adapter(
            args.controller_adapter,
            {
                "plan_npz": str(plan_path),
                "report_json": str(report_path),
                "command_json": str(command_path),
                "validation_json": str(validation_path),
                "closed_loop_json": str(closed_loop_path),
            },
        )


if __name__ == "__main__":
    main()
