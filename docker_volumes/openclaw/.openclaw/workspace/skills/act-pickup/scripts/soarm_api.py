#!/usr/bin/env python3

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from flask import Flask, jsonify, request

try:
    import pinocchio as pin
except ImportError:  # pinocchio only powers the optional XYZ end-effector readout
    pin = None

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from physicalai.capture import CameraType, ColorMode, SharedCamera
from physicalai.inference import InferenceModel


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REFERENCES_DIR = SKILL_DIR / "references"
URDF_PATH = REFERENCES_DIR / "so101_new_calib.urdf"
CALIBRATION_PATH = Path(os.getenv("SOARM_CALIBRATION", str(REFERENCES_DIR / "robot_calibration.json")))
CONFIG_PATH = Path(os.getenv("SOARM_CONFIG", str(REFERENCES_DIR / "config.yaml")))

EE_FRAME = "gripper_frame_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM_JOINTS + ["gripper"]
DEFAULT_SPEED = 0.2
STEP_DELAY_S = 0.05

# ACT pick pipeline config (mirrors ~/workspace/finetune/pipeline/main.py).
# Model settings, camera list, and loop tuning live in config.yaml.
def _load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh) or {}
    model = dict(cfg.get("model") or {})
    model_path = Path(model.get("model_path", "references/openvino"))
    if not model_path.is_absolute():
        model_path = SKILL_DIR / model_path
    model["model_path"] = str(model_path)
    model.setdefault("policy_name", "act")
    model.setdefault("backend", "openvino")
    model.setdefault("device", "GPU")
    cfg["model"] = model
    cfg["cameras"] = cfg.get("cameras") or []
    cfg["pick"] = cfg.get("pick") or {}
    return cfg


CONFIG = _load_config()
MODEL_CONFIG = CONFIG["model"]
CAMERA_CONFIGS = CONFIG["cameras"]
CONTROL_FREQUENCY = int(CONFIG["pick"].get("control_frequency", 30))  # Hz
ACTION_QUEUE_THRESHOLD = int(CONFIG["pick"].get("action_queue_threshold", 1))
PICK_MAX_STEPS = int(CONFIG["pick"].get("max_steps", 30 * CONTROL_FREQUENCY))  # ~30 s cap

_DRIVER_TO_CAMERA_TYPE = {
    "usb_camera": CameraType.UVC,
    "realsense": CameraType.REALSENSE,
}


def _load_calibration_ranges(path: Path) -> dict[str, tuple[int, int]]:
    with open(path) as fh:
        calibration = json.load(fh)
    return {name: (int(val["range_min"]), int(val["range_max"])) for name, val in calibration.items()}


def build_shared_camera(config: dict[str, Any]) -> SharedCamera:
    camera_type = _DRIVER_TO_CAMERA_TYPE[config["driver"]]
    kwargs: dict[str, Any] = {k: v for k, v in config["payload"].items() if v is not None}
    if camera_type == CameraType.UVC:
        kwargs["device"] = config["fingerprint"]
    else:
        kwargs["serial_number"] = config["serial_number"]
    return SharedCamera(
        camera_type,
        color_mode=ColorMode.RGB,
        validate_on_connect=True,
        overwrite_settings=True,
        idle_timeout=5.0,
        **kwargs,
    )


class PickTaskManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.last_result: dict | None = None
        self._stop = threading.Event()

    def status(self) -> dict:
        with self.lock:
            return {"running": self.running, "last_result": self.last_result}

    def stop(self) -> bool:
        with self.lock:
            if not self.running:
                return False
            self._stop.set()
            return True

    def start(self, worker) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.last_result = None
            self._stop.clear()
            stop_event = self._stop

        def runner() -> None:
            try:
                result = worker(stop_event)
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": str(exc)}
            with self.lock:
                self.running = False
                self.last_result = result

        threading.Thread(target=runner, daemon=True).start()
        return True


class ActPickEngine:
    """Runs the ACT inference pick loop, mirroring the reference pipeline."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cameras: dict[str, SharedCamera] = {}
        self.model: InferenceModel | None = None
        self.ready = False
        self.error: str | None = None

    def ensure_ready(self) -> None:
        with self.lock:
            if self.ready:
                return
            try:
                for cfg in CAMERA_CONFIGS:
                    cam = build_shared_camera(cfg)
                    cam.connect()
                    self.cameras["images." + cfg["name"]] = cam
                    time.sleep(1.0)  # camera warmup
                self.model = InferenceModel(
                    export_dir=MODEL_CONFIG["model_path"],
                    policy_name=MODEL_CONFIG["policy_name"],
                    backend=MODEL_CONFIG["backend"],
                    device=MODEL_CONFIG["device"],
                )
                self.ready = True
                self.error = None
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                self.close()
                raise

    def close(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001, S110
                pass
        self.cameras = {}
        self.model = None
        self.ready = False

    def _read_inputs(self, controller: SoArmController) -> dict[str, np.ndarray]:
        inputs: dict[str, np.ndarray] = {}
        state = controller.read_state_normalized()
        inputs["state"] = state[np.newaxis]  # [6] -> [1, 6]
        for key, cam in self.cameras.items():
            frame = cam.read_latest()
            inputs[key] = np.ascontiguousarray(
                frame.data[..., ::-1].transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255
            )
        return inputs

    def run(self, controller: SoArmController, stop_event: threading.Event) -> dict:
        self.ensure_ready()

        result_queue: queue.Queue = queue.Queue()
        current_inputs: dict[str, np.ndarray] = {}
        inference_busy = [False]
        infer_stop = threading.Event()

        def run_inference() -> None:
            if result_queue.qsize() < ACTION_QUEUE_THRESHOLD and not inference_busy[0] and current_inputs:
                inference_busy[0] = True
                try:
                    actions = self.model.predict_action_chunk(dict(current_inputs))
                    for action in actions:
                        result_queue.put(action)
                finally:
                    inference_busy[0] = False

        def infer_loop() -> None:
            while not infer_stop.is_set():
                try:
                    run_inference()
                except Exception as exc:  # noqa: BLE001
                    print(f"Inference error: {exc}")
                time.sleep(0.01)

        infer_thread = threading.Thread(target=infer_loop, daemon=True)
        infer_thread.start()

        step = 0
        with controller.lock:
            try:
                while not stop_event.is_set() and step < PICK_MAX_STEPS:
                    loop_start = time.time()
                    current_inputs = self._read_inputs(controller)
                    try:
                        action = result_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    controller.send_action_normalized(np.asarray(action, dtype=np.float32))
                    step += 1
                    elapsed = time.time() - loop_start
                    time.sleep(max(0.0, (1.0 / CONTROL_FREQUENCY) - elapsed))
            finally:
                infer_stop.set()
                infer_thread.join(timeout=1.0)

        return {
            "ok": True,
            "message": "抓取任务完成",
            "steps": step,
            "stopped": stop_event.is_set(),
        }


class SoArmController:
    def __init__(self, port: str, robot_id: str, skip_calibration: bool):
        self.lock = threading.Lock()
        self.has_fk = pin is not None
        if self.has_fk:
            self.model = pin.buildModelFromUrdf(str(URDF_PATH))
            self.data = self.model.createData()
            self.frame_id = self.model.getFrameId(EE_FRAME)
        self.calibration_ranges = _load_calibration_ranges(CALIBRATION_PATH)
        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=port,
                id=robot_id,
                disable_torque_on_disconnect=True,
                use_degrees=True,
            )
        )
        self.robot.connect(calibrate=not skip_calibration)

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()

    def _fk_position(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id].translation.copy()

    def _gripper_pct_to_rad(self, gripper_pct: float) -> float:
        lower = self.model.lowerPositionLimit[5]
        upper = self.model.upperPositionLimit[5]
        return lower + (gripper_pct / 100.0) * (upper - lower)

    def _observation_to_q(self, observation: dict[str, float]) -> np.ndarray:
        q = np.zeros(self.model.nq)
        for i, joint in enumerate(ARM_JOINTS):
            q[i] = np.deg2rad(observation[f"{joint}.pos"])
        q[5] = self._gripper_pct_to_rad(observation["gripper.pos"])
        return np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

    def _read_state(self) -> tuple[dict[str, float], np.ndarray | None]:
        observation = self.robot.get_observation()
        if not self.has_fk:
            return observation, None
        q = self._observation_to_q(observation)
        xyz = self._fk_position(q)
        return observation, xyz

    @staticmethod
    def _xyz_list(xyz: np.ndarray | None) -> list | None:
        return xyz.tolist() if xyz is not None else None

    # --- ACT bridge: raw servo ticks <-> normalized units ([-100,100], gripper [0,100]) ---
    def read_state_normalized(self) -> np.ndarray:
        ticks = self.robot.bus.sync_read("Present_Position", normalize=False)
        state = np.empty(len(JOINTS), dtype=np.float32)
        for i, name in enumerate(JOINTS):
            lo, hi = self.calibration_ranges[name]
            rng = hi - lo
            if rng <= 0:
                state[i] = 0.0
                continue
            t = float(np.clip(ticks[name], lo, hi))
            if name == "gripper":
                state[i] = float(np.clip((t - lo) / rng * 100.0, 0.0, 100.0))
            else:
                state[i] = float(np.clip((t - lo) / rng * 200.0 - 100.0, -100.0, 100.0))
        return state

    def send_action_normalized(self, action: np.ndarray) -> None:
        goal: dict[str, int] = {}
        for i, name in enumerate(JOINTS):
            lo, hi = self.calibration_ranges[name]
            rng = hi - lo
            value = float(action[i])
            if name == "gripper":
                clamped = float(np.clip(value, 0.0, 100.0))
                tick = round(lo + clamped / 100.0 * rng)
            else:
                clamped = float(np.clip(value, -100.0, 100.0))
                tick = round(lo + (clamped + 100.0) / 200.0 * rng)
            goal[name] = int(np.clip(tick, lo, hi))
        self.robot.bus.sync_write("Goal_Position", goal, normalize=False)

    def _send_smooth_action(
        self, start: dict[str, float], target: dict[str, float], speed: float
    ) -> tuple[dict[str, float], int]:
        if speed <= 0:
            raise ValueError("speed must be > 0.")

        arm_delta = max(abs(target[f"{joint}.pos"] - start[f"{joint}.pos"]) for joint in ARM_JOINTS)
        gripper_delta = abs(target["gripper.pos"] - start["gripper.pos"])
        arm_step_deg = 8.0 * speed
        gripper_step_pct = 10.0 * speed
        steps = max(
            1,
            int(np.ceil(arm_delta / arm_step_deg)) if arm_step_deg > 0 else 1,
            int(np.ceil(gripper_delta / gripper_step_pct)) if gripper_step_pct > 0 else 1,
        )

        sent = target
        for i in range(1, steps + 1):
            alpha = i / steps
            action = {
                f"{joint}.pos": float(
                    start[f"{joint}.pos"] + alpha * (target[f"{joint}.pos"] - start[f"{joint}.pos"])
                )
                for joint in JOINTS
            }
            sent = self.robot.send_action(action)
            if i < steps:
                time.sleep(STEP_DELAY_S)
        return sent, steps

    def status(self) -> dict:
        with self.lock:
            observation, xyz = self._read_state()
            return {
                "connected": self.robot.is_connected,
                "joints": observation,
                "xyz": self._xyz_list(xyz),
            }

    def move_joints(self, angles: list[float], sleep_s: float, speed: float) -> dict:
        if len(angles) != 6:
            raise ValueError("angles must contain 6 values.")
        if not 0.0 <= angles[5] <= 100.0:
            raise ValueError("gripper must be in [0, 100].")

        target = {f"{joint}.pos": float(value) for joint, value in zip(JOINTS, angles, strict=True)}
        with self.lock:
            before, before_xyz = self._read_state()
            sent, steps = self._send_smooth_action(before, target, speed)
            if sleep_s > 0:
                time.sleep(sleep_s)
            after, after_xyz = self._read_state()
            return {
                "before": before,
                "before_xyz": self._xyz_list(before_xyz),
                "sent": sent,
                "speed": speed,
                "steps": steps,
                "after": after,
                "after_xyz": self._xyz_list(after_xyz),
            }


def create_app() -> Flask:
    port = os.getenv("SOARM_PORT", "/dev/ttyACM0")
    robot_id = os.getenv("SOARM_ID", "openclaw_soarm")
    skip_calibration = os.getenv("SOARM_SKIP_CALIBRATION", "1") != "0"
    controller = SoArmController(port=port, robot_id=robot_id, skip_calibration=skip_calibration)
    atexit.register(controller.close)

    pick_engine = ActPickEngine()
    atexit.register(pick_engine.close)
    pick_manager = PickTaskManager()

    app = Flask(__name__)
    app.config["controller"] = controller

    @app.get("/healthz")
    def healthz():
        return jsonify(
            {
                "ok": True,
                "connected": controller.robot.is_connected,
                "pick_ready": pick_engine.ready,
                "pick_error": pick_engine.error,
            }
        )

    @app.get("/joints")
    def joints():
        return jsonify(controller.status())

    @app.get("/status")
    def pick_status():
        return jsonify(pick_manager.status())

    @app.post("/pick")
    def pick():
        started = pick_manager.start(lambda stop_event: pick_engine.run(controller, stop_event))
        if not started:
            return jsonify({"ok": False, "message": "任务正在运行中"}), 409
        return jsonify({"ok": True, "message": "抓取任务已启动"})

    @app.post("/pick/stop")
    def pick_stop():
        stopped = pick_manager.stop()
        return jsonify({"ok": True, "stopping": stopped})

    @app.post("/move/joints")
    def move_joints():
        payload = request.get_json(force=True, silent=False) or {}
        angles = payload.get("angles")
        sleep_s = float(payload.get("sleep", 2.0))
        speed = float(payload.get("speed", DEFAULT_SPEED))
        return jsonify(controller.move_joints(angles, sleep_s, speed))

    @app.post("/disconnect")
    def disconnect():
        controller.close()
        return jsonify({"ok": True, "connected": False})

    @app.errorhandler(Exception)
    def handle_error(exc: Exception):
        return jsonify({"ok": False, "error": str(exc)}), 400

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("SOARM_API_HOST", "127.0.0.1")
    port = int(os.getenv("SOARM_API_PORT", "8000"))
    app.run(host=host, port=port, threaded=True, use_reloader=False)
