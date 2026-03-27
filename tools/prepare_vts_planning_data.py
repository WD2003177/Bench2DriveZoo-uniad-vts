#!/usr/bin/env python3
"""Prepare VTS planning data into UniAD/mmdet3d compatible pkl infos.

This script converts one or more VTS collection sequences into NuScenes-style
info pkl files with:
- scene-level temporal chain: scene_token + prev/next.
- 6 camera records in info['cams'] with fixed intrinsics/extrinsics.
- extracted planning fields from json: gt_ego_traj, can_bus, command,
  ego2global_translation, ego2global_rotation.
- dummy perception/motion fields to avoid DataLoader KeyError.
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CAMERA_TO_FOLDER = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}

# Fixed calibration copied from code/uniad_b2d_agent.py (right-handed: X front, Y left, Z up).
LIDAR2CAM: Dict[str, np.ndarray] = {
    "CAM_FRONT": np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.24], [0.0, 1.0, 0.0, -1.19], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
    "CAM_FRONT_LEFT": np.array(
        [
            [0.57357644, 0.81915204, 0.0, -0.22517331],
            [0.0, 0.0, -1.0, -0.24],
            [-0.81915204, 0.57357644, 0.0, -0.82909407],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "CAM_FRONT_RIGHT": np.array(
        [
            [0.57357644, -0.81915204, 0.0, 0.22517331],
            [0.0, 0.0, -1.0, -0.24],
            [0.81915204, 0.57357644, 0.0, -0.82909407],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "CAM_BACK": np.array(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.24], [0.0, -1.0, 0.0, -1.61], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
    "CAM_BACK_LEFT": np.array(
        [
            [-0.34202014, 0.93969262, 0.0, -0.25388956],
            [0.0, 0.0, -1.0, -0.24],
            [-0.93969262, -0.34202014, 0.0, -0.49288953],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "CAM_BACK_RIGHT": np.array(
        [
            [-0.34202014, -0.93969262, 0.0, 0.25388956],
            [0.0, 0.0, -1.0, -0.24],
            [0.93969262, -0.34202014, 0.0, -0.49288953],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}

LIDAR2EGO = np.array(
    [[0.0, 1.0, 0.0, -0.39], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.84], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# Shared 1600x900 front-camera intrinsic used by the collector setup.
CAM_INTRINSIC = np.array(
    [[1142.51841, 0.0, 800.0], [0.0, 1142.51841, 450.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def matrix_to_quaternion_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion in [w, x, y, z]."""
    m = rot
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quaternion_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def build_fixed_cam_calibration() -> Dict[str, Dict[str, np.ndarray]]:
    calibs = {}
    for cam in CAMERAS:
        lidar2cam = LIDAR2CAM[cam]
        cam2lidar = np.linalg.inv(lidar2cam)
        cam2ego = LIDAR2EGO @ cam2lidar

        calibs[cam] = {
            "cam_intrinsic": CAM_INTRINSIC.astype(np.float32),
            "sensor2ego_translation": cam2ego[:3, 3].astype(np.float32),
            "sensor2ego_rotation": matrix_to_quaternion_wxyz(cam2ego[:3, :3]).astype(np.float32),
            "sensor2lidar_translation": cam2lidar[:3, 3].astype(np.float32),
            "sensor2lidar_rotation": cam2lidar[:3, :3].astype(np.float32),
            "cam2ego": cam2ego.astype(np.float32),
        }
    return calibs


FIXED_CAM_CALIB = build_fixed_cam_calibration()
LIDAR2EGO_TRANSLATION = LIDAR2EGO[:3, 3].astype(np.float32)
LIDAR2EGO_ROTATION = matrix_to_quaternion_wxyz(LIDAR2EGO[:3, :3]).astype(np.float32)


def normalize_can_bus(can_bus_raw) -> np.ndarray:
    can_bus = np.asarray(can_bus_raw, dtype=np.float32).reshape(-1) if can_bus_raw is not None else np.zeros(0)
    out = np.zeros((18,), dtype=np.float32)
    n = min(can_bus.shape[0], 18)
    if n > 0:
        out[:n] = can_bus[:n]
    return out


def normalize_translation(translation_raw, can_bus: np.ndarray) -> np.ndarray:
    if translation_raw is None:
        return can_bus[:3].astype(np.float32)
    t = np.asarray(translation_raw, dtype=np.float32).reshape(-1)
    out = np.zeros((3,), dtype=np.float32)
    n = min(t.shape[0], 3)
    if n > 0:
        out[:n] = t[:n]
    return out


def yaw_to_quaternion_wxyz(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def normalize_rotation(rotation_raw, can_bus: np.ndarray) -> np.ndarray:
    if rotation_raw is None:
        if abs(float(can_bus[16])) > 1e-8:
            return yaw_to_quaternion_wxyz(float(can_bus[16]))
        q = can_bus[3:7].astype(np.float32)
        n = float(np.linalg.norm(q))
        if n > 1e-8:
            return q / n
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    q = np.asarray(rotation_raw, dtype=np.float32).reshape(-1)
    out = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    n = min(q.shape[0], 4)
    if n > 0:
        out[:n] = q[:n]
    norm = float(np.linalg.norm(out))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return out / norm


def parse_timestamp_us(ts_raw, fallback_idx: int) -> int:
    if ts_raw is None:
        return int(fallback_idx * 100000)
    try:
        ts = float(ts_raw)
    except Exception:
        return int(fallback_idx * 100000)

    if ts > 1e14:
        return int(ts / 1000.0)  # ns -> us
    if ts > 1e12:
        return int(ts)  # likely us
    if ts > 1e10:
        return int(ts * 1000.0)  # likely ms -> us
    return int(ts * 1e6)  # sec -> us


def list_json_files(meta_dir: Path) -> List[Path]:
    files = sorted(meta_dir.glob("*.json"))

    def sort_key(p: Path) -> Tuple[int, str]:
        nums = re.findall(r"\d+", p.stem)
        idx = int(nums[-1]) if nums else int(1e18)
        return idx, p.stem

    return sorted(files, key=sort_key)


def make_scene_token(scene_dir: Path) -> str:
    return scene_dir.name


def resolve_image_path(scene_dir: Path, meta: dict, meta_file: Path, cam: str) -> str:
    img_dict = meta.get("images", None)
    if isinstance(img_dict, dict) and cam in img_dict:
        raw = Path(str(img_dict[cam]))
        if raw.is_absolute():
            return str(raw)
        return str((scene_dir / raw).resolve())

    cam_dir = scene_dir / CAMERA_TO_FOLDER[cam]
    stem = meta_file.stem
    candidate_stems = [stem]

    nums = re.findall(r"\d+", stem)
    if nums:
        idx = int(nums[-1])
        candidate_stems.extend([
            str(idx),
            f"{idx:04d}",
            f"{idx:05d}",
            f"{idx:06d}",
            f"frame_{idx:06d}",
        ])

    seen = set()
    uniq_stems = []
    for c in candidate_stems:
        if c not in seen:
            uniq_stems.append(c)
            seen.add(c)

    for c in uniq_stems:
        for ext in (".jpg", ".jpeg", ".png"):
            p = cam_dir / f"{c}{ext}"
            if p.exists():
                return str(p.resolve())

    # fallback path (even if not existing yet)
    return str((cam_dir / f"{stem}.jpg").resolve())


def parse_gt_ego_traj(meta: dict) -> np.ndarray:
    for key in ("gt_ego_traj", "expert_trajectory_for_pid", "expert_trajectory_global", "plan"):
        if key in meta:
            arr = np.asarray(meta[key], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr
    return np.zeros((0, 2), dtype=np.float32)


def discover_scene_dirs(data_root: Path) -> List[Path]:
    scenes: List[Path] = []

    # Case A: data_root itself is one collected scene.
    if (data_root / "meta").is_dir():
        scenes.append(data_root)

    # Case B: each direct child is one scene.
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and (child / "meta").is_dir():
            scenes.append(child)

    if scenes:
        uniq = []
        seen = set()
        for s in scenes:
            r = str(s.resolve())
            if r not in seen:
                seen.add(r)
                uniq.append(s)
        return sorted(uniq)

    # Case C: recursive search.
    for meta_dir in sorted(data_root.rglob("meta")):
        if meta_dir.is_dir() and any(meta_dir.glob("*.json")):
            scenes.append(meta_dir.parent)
    return sorted(scenes)


def build_dummy_fields(info: dict, past_steps: int, fut_steps: int) -> None:
    info["gt_bboxes_3d"] = np.zeros((0, 9), dtype=np.float32)
    info["gt_labels_3d"] = np.zeros((0,), dtype=np.int64)
    info["gt_names"] = np.array([], dtype=object)
    info["gt_inds"] = np.zeros((0,), dtype=np.int64)

    info["gt_past_traj"] = np.zeros((0, past_steps, 2), dtype=np.float32)
    info["gt_fut_traj"] = np.zeros((0, fut_steps, 2), dtype=np.float32)
    info["gt_past_traj_mask"] = np.zeros((0, past_steps, 2), dtype=np.float32)
    info["gt_fut_traj_mask"] = np.zeros((0, fut_steps, 2), dtype=np.float32)

    # Extra compatibility with NuScenes/B2D style dataset readers.
    info["gt_boxes"] = np.zeros((0, 7), dtype=np.float32)
    info["gt_velocity"] = np.zeros((0, 2), dtype=np.float32)
    info["gt_ids"] = np.zeros((0,), dtype=np.int64)
    info["num_points"] = np.zeros((0,), dtype=np.int64)
    info["num_lidar_pts"] = np.zeros((0,), dtype=np.int64)
    info["num_radar_pts"] = np.zeros((0,), dtype=np.int64)
    info["valid_flag"] = np.zeros((0,), dtype=bool)
    info["npc2world"] = np.zeros((0, 4, 4), dtype=np.float32)


def build_frame_info(scene_dir: Path, meta_file: Path, frame_idx: int, past_steps: int, fut_steps: int) -> dict:
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scene_token = make_scene_token(scene_dir)
    frame_token = str(meta.get("frame_token", f"{scene_token}_{frame_idx:06d}"))

    can_bus = normalize_can_bus(meta.get("can_bus"))
    ego2global_translation = normalize_translation(meta.get("ego2global_translation"), can_bus)
    ego2global_rotation = normalize_rotation(meta.get("ego2global_rotation"), can_bus)

    timestamp_us = parse_timestamp_us(meta.get("timestamp"), frame_idx)

    # Build world<->ego for B2D-style compatibility fields.
    ego2global = np.eye(4, dtype=np.float64)
    ego2global[:3, :3] = quaternion_wxyz_to_matrix(ego2global_rotation)
    ego2global[:3, 3] = ego2global_translation
    world2ego = np.linalg.inv(ego2global).astype(np.float32)

    world2lidar = np.linalg.inv(ego2global @ LIDAR2EGO).astype(np.float32)

    cams = {}
    sensors = {}
    for cam in CAMERAS:
        calib = FIXED_CAM_CALIB[cam]
        data_path = resolve_image_path(scene_dir, meta, meta_file, cam)

        cams[cam] = {
            "data_path": data_path,
            "cam_intrinsic": calib["cam_intrinsic"].copy(),
            "sensor2ego_translation": calib["sensor2ego_translation"].copy(),
            "sensor2ego_rotation": calib["sensor2ego_rotation"].copy(),
            "sensor2lidar_translation": calib["sensor2lidar_translation"].copy(),
            "sensor2lidar_rotation": calib["sensor2lidar_rotation"].copy(),
            "sample_data_token": f"{frame_token}_{cam}",
            "timestamp": timestamp_us,
            "ego2global_translation": ego2global_translation.copy(),
            "ego2global_rotation": ego2global_rotation.copy(),
        }

        cam2ego = calib["cam2ego"].copy()
        sensors[cam] = {
            "cam2ego": cam2ego,
            "intrinsic": calib["cam_intrinsic"].copy(),
            "world2cam": (np.linalg.inv(cam2ego) @ world2ego).astype(np.float32),
            "data_path": data_path,
        }

    sensors["LIDAR_TOP"] = {
        "lidar2ego": LIDAR2EGO.astype(np.float32).copy(),
        "world2lidar": world2lidar,
    }

    ego_yaw = float(can_bus[16])
    if abs(ego_yaw) < 1e-8:
        # Recover yaw from quaternion if can_bus yaw is absent.
        r = quaternion_wxyz_to_matrix(ego2global_rotation)
        ego_yaw = float(np.arctan2(r[1, 0], r[0, 0]))

    info = {
        "token": frame_token,
        "frame_token": frame_token,
        "scene_token": scene_token,
        "folder": scene_token,
        "frame_idx": int(frame_idx),
        "timestamp": int(timestamp_us),
        "prev": "",
        "next": "",
        "sweeps": [],
        "cams": cams,
        "sensors": sensors,
        "can_bus": can_bus,
        "command": int(meta.get("command", 2)),
        "gt_ego_traj": parse_gt_ego_traj(meta),
        "ego2global_translation": ego2global_translation,
        "ego2global_rotation": ego2global_rotation,
        "lidar2ego_translation": LIDAR2EGO_TRANSLATION.copy(),
        "lidar2ego_rotation": LIDAR2EGO_ROTATION.copy(),
        "ego_translation": ego2global_translation.copy(),
        "ego_yaw": float(ego_yaw),
        "ego_vel": np.array([float(can_bus[7]), float(can_bus[8]), float(can_bus[9])], dtype=np.float32),
        "ego_accel": np.array([float(can_bus[10]), float(can_bus[11]), float(can_bus[12])], dtype=np.float32),
        "ego_rotation_rate": np.array([float(can_bus[13]), float(can_bus[14]), float(can_bus[15])], dtype=np.float32),
        "world2ego": world2ego,
    }

    build_dummy_fields(info, past_steps=past_steps, fut_steps=fut_steps)
    return info


def convert_scene(scene_dir: Path, past_steps: int, fut_steps: int) -> List[dict]:
    meta_dir = scene_dir / "meta"
    json_files = list_json_files(meta_dir)
    if not json_files:
        return []

    infos = []
    for idx, meta_file in enumerate(json_files):
        infos.append(build_frame_info(scene_dir, meta_file, idx, past_steps=past_steps, fut_steps=fut_steps))

    # Hard requirement: fill temporal chain (prev/next) by frame_token.
    frame_tokens = [x["frame_token"] for x in infos]
    for i, info in enumerate(infos):
        info["prev"] = "" if i == 0 else frame_tokens[i - 1]
        info["next"] = "" if i == len(infos) - 1 else frame_tokens[i + 1]
    return infos


def split_scenes(scene_dirs: List[Path], train_ratio: float) -> Tuple[List[Path], List[Path]]:
    if not scene_dirs:
        return [], []

    n = len(scene_dirs)
    n_train = int(n * train_ratio)
    if n > 1:
        n_train = max(1, min(n_train, n - 1))
    else:
        n_train = 1

    train_scenes = scene_dirs[:n_train]
    val_scenes = scene_dirs[n_train:]
    return train_scenes, val_scenes


def dump_infos(infos: List[dict], out_path: Path) -> None:
    payload = {
        "metadata": {
            "dataset": "VTS",
            "version": "v1.0-vts-planning",
        },
        "infos": infos,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VTS planning pkl infos with dummy perception fields.")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of VTS collections.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for pkl files.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio by scenes. Default: 0.8")
    parser.add_argument("--past-steps", type=int, default=4, help="Dummy gt_past_traj time steps. Default: 4")
    parser.add_argument("--fut-steps", type=int, default=12, help="Dummy gt_fut_traj time steps. Default: 12")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {data_root}")

    scene_dirs = discover_scene_dirs(data_root)
    if not scene_dirs:
        raise FileNotFoundError(f"No scenes found under: {data_root}")

    train_scene_dirs, val_scene_dirs = split_scenes(scene_dirs, train_ratio=args.train_ratio)

    train_infos: List[dict] = []
    val_infos: List[dict] = []

    for scene_dir in train_scene_dirs:
        train_infos.extend(convert_scene(scene_dir, past_steps=args.past_steps, fut_steps=args.fut_steps))
    for scene_dir in val_scene_dirs:
        val_infos.extend(convert_scene(scene_dir, past_steps=args.past_steps, fut_steps=args.fut_steps))

    train_path = out_dir / "vts_infos_train.pkl"
    val_path = out_dir / "vts_infos_val.pkl"
    dump_infos(train_infos, train_path)
    dump_infos(val_infos, val_path)

    print(f"Found scenes: total={len(scene_dirs)}, train={len(train_scene_dirs)}, val={len(val_scene_dirs)}")
    print(f"Generated infos: train={len(train_infos)}, val={len(val_infos)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
