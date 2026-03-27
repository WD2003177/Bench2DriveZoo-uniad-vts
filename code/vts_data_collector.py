import datetime
import json
import math
import pathlib
import uuid

import cv2
import numpy as np
from chassis.proto.chassis_messages_pb2 import VehicleControl
from pid_controller import PIDController
from planner import RoutePlanner
from pyquaternion import Quaternion


class VTSPlanningCollector:
    CAMERA_KEYS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    CAMERA_DIRS = {
        "CAM_FRONT": "rgb_front",
        "CAM_FRONT_LEFT": "rgb_front_left",
        "CAM_FRONT_RIGHT": "rgb_front_right",
        "CAM_BACK": "rgb_back",
        "CAM_BACK_LEFT": "rgb_back_left",
        "CAM_BACK_RIGHT": "rgb_back_right",
    }

    def __init__(
        self,
        save_root="code/output",
        save_every_n_ticks=2,
        pid_waypoint_count=6,
        jpg_quality=95,
    ):
        self.pidcontroller = PIDController(clip_delta=0.1, turn_KP=0.1)
        self.route_planner = RoutePlanner(4.0, 50.0)
        self.save_every_n_ticks = max(1, int(save_every_n_ticks))
        self.pid_waypoint_count = max(2, int(pid_waypoint_count))
        self.jpg_quality = int(jpg_quality)

        self.step = -1
        self.initialized = False

        now = datetime.datetime.now()
        run_name = "_".join(
            map(lambda x: "%02d" % x, (now.month, now.day, now.hour, now.minute, now.second))
        )
        self.save_path = pathlib.Path(save_root) / run_name
        self.save_path.mkdir(parents=True, exist_ok=False)

        self.image_dirs = {}
        for cam, folder_name in self.CAMERA_DIRS.items():
            folder = self.save_path / folder_name
            folder.mkdir(parents=True, exist_ok=False)
            self.image_dirs[cam] = folder
        self.meta_dir = self.save_path / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=False)

    def init(self, global_plan):
        if not global_plan:
            raise ValueError("global_plan is empty, cannot initialize collector.")
        self.route_planner.set_route(global_plan)
        self.route_planner.first_hit = True
        self.initialized = True

    def reset(self):
        self.step = -1
        self.initialized = False

    def run_step(self, tick_data):
        if not self.initialized:
            raise RuntimeError("Collector is not initialized. Please call init(route) first.")

        self.step += 1
        state = self._parse_tick_data(tick_data)

        near_node, near_command = self.route_planner.run_step(state["pos_xy"], state["heading"])
        command = int(near_command.value if hasattr(near_command, "value") else near_command)
        if command < 0:
            command = 4
        command -= 1

        target_for_pid = self._build_target_for_pid(
            state["pos_xy"], state["heading"], np.asarray(near_node, dtype=np.float32)
        )

        gt_ego_traj, waypoints_for_pid = self._build_expert_trajectory_for_pid(
            state["pos_xy"], state["heading"], state["speed"], self.pid_waypoint_count
        )

        steer_traj, throttle_traj, brake_traj, metadata = self.pidcontroller.control_pid(
            waypoints_for_pid,
            np.array(state["speed"], dtype=np.float32),
            target_for_pid,
        )

        if brake_traj < 0.05:
            brake_traj = 0.0
        if throttle_traj > brake_traj:
            brake_traj = 0.0
        if state["speed"] > 5.0:
            throttle_traj = 0.0

        steer = np.clip(float(steer_traj), -1.0, 1.0)
        throttle = np.clip(float(throttle_traj), 0.0, 0.75)
        brake = np.clip(float(brake_traj), 0.0, 1.0)

        control_cmd = self._build_vts_control(steer, throttle, brake, metadata)

        can_bus, ego_quat = self._build_can_bus(state)
        ego2global_translation = [
            float(state["pos_xy"][0]),
            float(state["pos_xy"][1]),
            float(state["pos_z"]),
        ]
        ego2global_rotation = [float(v) for v in ego_quat]

        if self.step % self.save_every_n_ticks == 0:
            self._save_frame(
                state=state,
                can_bus=can_bus,
                command=command,
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
                target_for_pid=target_for_pid,
                gt_ego_traj=gt_ego_traj,
                waypoints_for_pid=waypoints_for_pid,
                control=dict(
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                    desired_speed=float(metadata["desired_speed"]),
                    acceleration=float(metadata["delta"] / 0.1),
                ),
            )

        return control_cmd

    def _parse_tick_data(self, tick_data):
        images = {cam: tick_data[cam] for cam in self.CAMERA_KEYS}

        imu = np.asarray(tick_data["IMU"], dtype=np.float32)
        pos_xy = np.asarray(tick_data["POS"], dtype=np.float32)
        speed = float(np.asarray(tick_data["SPEED"], dtype=np.float32).reshape(-1)[0])

        heading = float(imu[-1]) if imu.shape[0] >= 7 else 0.0
        acceleration = imu[:3] if imu.shape[0] >= 3 else np.zeros(3, dtype=np.float32)
        angular_velocity = imu[3:6] if imu.shape[0] >= 6 else np.zeros(3, dtype=np.float32)

        if math.isnan(heading):
            heading = 0.0
            acceleration = np.zeros(3, dtype=np.float32)
            angular_velocity = np.zeros(3, dtype=np.float32)

        return {
            "timestamp": float(tick_data.get("timestamp", 0.0)),
            "images": images,
            "pos_xy": pos_xy,
            "pos_z": float(tick_data.get("POS_Z", 0.0)),
            "speed": speed,
            "heading": heading,
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
        }

    def _build_target_for_pid(self, pos_xy, heading, target_global_xy):
        delta = target_global_xy - pos_xy
        local_xy = self._world_to_ego_xy(delta, heading)
        return np.array([local_xy[1], local_xy[0]], dtype=np.float32)

    def _build_expert_trajectory_for_pid(self, pos_xy, heading, speed, waypoint_count):
        """
        Build time-indexed future trajectory.
        Returns:
          - gt_ego_traj: [X forward, Y left], for UniAD training GT.
          - waypoints_for_pid: [Y left, X forward], for PID controller.
        """
        dt = 0.5
        planning_speed = max(float(speed), 2.0)

        route_points = []
        for route_point, _ in list(self.route_planner.route):
            route_point = np.asarray(route_point, dtype=np.float32)
            local_xy = self._world_to_ego_xy(route_point - pos_xy, heading)
            if local_xy[0] < -1.0:
                continue
            route_points.append(route_point)

        gt_points = []
        if len(route_points) < 1:
            for i in range(1, waypoint_count + 1):
                gt_points.append(np.array([planning_speed * i * dt, 0.0], dtype=np.float32))
        else:
            polyline = [np.asarray(pos_xy, dtype=np.float32)]
            for p in route_points:
                if np.linalg.norm(p - polyline[-1]) > 1e-4:
                    polyline.append(p)

            if len(polyline) < 2:
                for i in range(1, waypoint_count + 1):
                    gt_points.append(np.array([planning_speed * i * dt, 0.0], dtype=np.float32))
            else:
                polyline = np.stack(polyline, axis=0)
                seg = polyline[1:] - polyline[:-1]
                seg_len = np.linalg.norm(seg, axis=1)
                cumdist = np.concatenate([[0.0], np.cumsum(seg_len)])
                total_len = float(cumdist[-1])

                for i in range(1, waypoint_count + 1):
                    target_dist = planning_speed * i * dt
                    if target_dist <= total_len and total_len > 1e-6:
                        idx = int(np.searchsorted(cumdist, target_dist, side="right") - 1)
                        idx = np.clip(idx, 0, len(seg_len) - 1)
                        seg_denom = max(float(seg_len[idx]), 1e-6)
                        ratio = (target_dist - float(cumdist[idx])) / seg_denom
                        interp_pt = polyline[idx] + ratio * (polyline[idx + 1] - polyline[idx])
                    else:
                        if len(gt_points) > 0:
                            interp_local = gt_points[-1] + np.array([planning_speed * dt, 0.0], dtype=np.float32)
                            gt_points.append(interp_local.astype(np.float32))
                            continue
                        interp_pt = polyline[-1]

                    local_xy = self._world_to_ego_xy(interp_pt - pos_xy, heading)
                    gt_points.append(np.array([local_xy[0], local_xy[1]], dtype=np.float32))

        gt_ego_traj = np.stack(gt_points, axis=0).astype(np.float32)
        waypoints_for_pid = np.zeros_like(gt_ego_traj)
        waypoints_for_pid[:, 0] = gt_ego_traj[:, 1]
        waypoints_for_pid[:, 1] = gt_ego_traj[:, 0]
        return gt_ego_traj, waypoints_for_pid

    def _world_to_ego_xy(self, world_delta_xy, heading):
        rotation_matrix = np.array(
            [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]],
            dtype=np.float32,
        )
        return rotation_matrix @ world_delta_xy

    def _build_can_bus(self, state):
        can_bus = np.zeros(18, dtype=np.float32)
        quat = list(Quaternion(axis=[0, 0, 1], radians=state["heading"]))

        can_bus[0] = float(state["pos_xy"][0])
        can_bus[1] = float(state["pos_xy"][1])
        can_bus[2] = float(state["pos_z"])
        can_bus[3:7] = np.asarray(quat, dtype=np.float32)
        can_bus[7] = float(state["speed"])
        can_bus[10:13] = state["acceleration"].astype(np.float32)
        can_bus[13:16] = state["angular_velocity"].astype(np.float32)
        can_bus[16] = float(state["heading"])
        can_bus[17] = float(state["heading"] / np.pi * 180.0)

        return can_bus, quat

    def _build_vts_control(self, steer, throttle, brake, metadata):
        vts_control = VehicleControl()
        vts_control.driving_control.driving_Mode = 2
        vts_control.driving_control.gear_control.gear_mode = 0
        vts_control.driving_control.gear_control.target_gear_position = 1
        vts_control.acceleration = float(metadata["delta"] / 0.1)
        vts_control.speed = float(metadata["desired_speed"])
        vts_control.steering_control.target_steering_wheel_angle = float(steer * 100.0)

        _ = throttle
        _ = brake
        return vts_control

    def _save_frame(
        self,
        state,
        can_bus,
        command,
        ego2global_translation,
        ego2global_rotation,
        target_for_pid,
        gt_ego_traj,
        waypoints_for_pid,
        control,
    ):
        frame_name = f"frame_{self.step:06d}"
        image_rel_paths = {}

        for cam in self.CAMERA_KEYS:
            img = state["images"][cam]
            img_file = self.image_dirs[cam] / f"{frame_name}.jpg"
            cv2.imwrite(str(img_file), img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality])
            image_rel_paths[cam] = f"{self.CAMERA_DIRS[cam]}/{frame_name}.jpg"

        meta = {
            "timestamp": float(state["timestamp"]),
            "frame_token": str(uuid.uuid4()),
            "step": int(self.step),
            "can_bus": [float(v) for v in can_bus.tolist()],
            "command": int(command),
            "ego2global_translation": [float(v) for v in ego2global_translation],
            "ego2global_rotation": [float(v) for v in ego2global_rotation],
            "target_for_pid": [float(v) for v in target_for_pid.tolist()],
            "gt_ego_traj": [[float(p[0]), float(p[1])] for p in gt_ego_traj.tolist()],
            "expert_trajectory_for_pid": [[float(p[0]), float(p[1])] for p in waypoints_for_pid.tolist()],
            "control": {
                "steer": float(control["steer"]),
                "throttle": float(control["throttle"]),
                "brake": float(control["brake"]),
                "speed": float(control["desired_speed"]),
                "acceleration": float(control["acceleration"]),
            },
            "images": image_rel_paths,
        }

        meta_file = self.meta_dir / f"{frame_name}.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
