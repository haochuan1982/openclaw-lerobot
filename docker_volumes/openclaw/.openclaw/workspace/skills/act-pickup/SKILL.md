---
name: act-pickup
description: Control the robotic arm through the OpenClaw SOARM API. Use this skill when reading current joint state, moving by joint angles, running an ACT-model pick task, disconnecting the arm, or handling SOARM robot-control requests.
---

# 🦐 SOARM Control Skill

Use the existing SOARM API to control the robotic arm directly.

## ⚙️ Configuration Notes

**Default Local Setup (when on same machine):**
- API Base URL: `http://localhost:8000`

## 🔍 Key APIs & Examples

### Read Current State

```bash
curl -sS http://localhost:8000/joints
```

Returns current joint values and XYZ end-effector position.

---

### Move To a Position By Joint Angles

```bash
curl -sS -X POST http://localhost:8000/move/joints \
  -H 'Content-Type: application/json' \
  -d '{"angles":[0,0,0,0,0,0]}'
```

**Parameter Notes:**
- Joints order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`
- First 5 joints use **degrees (deg)**
- Gripper uses **0-100** range

**Fixed Positions:**

| Name | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---:|---:|---:|---:|---:|---:|
| `initial` | 0 | -104 | 95 | 65 | -95 | 10 |
| `top_down` | 0 | -50 | 30 | 90 | -95 | 70 |

**Examples:**

Return to `initial`:

```bash
curl -sS -X POST http://localhost:8000/move/joints \
  -H 'Content-Type: application/json' \
  -d '{"angles":[0,-104,95,65,-95,10]}'
```

Return to `top_down`:

```bash
curl -sS -X POST http://localhost:8000/move/joints \
  -H 'Content-Type: application/json' \
  -d '{"angles":[0,-50,30,90,-95,70]}'
```

---

### Trigger a Pick Task (ACT model)

```bash
curl -sS -X POST http://localhost:8000/pick
```

Runs the ACT inference pick pipeline (ports `~/workspace/finetune/pipeline/main.py`):
reads normalized joint state + 4 camera frames (`top`, `head`, `right`, `left`) at
30 Hz, feeds them to the OpenVINO ACT policy, and streams the predicted action chunks
back to the arm until the object is picked or the step cap is reached.

Returns:

- `ok`: true if the task was accepted
- `message`: `抓取任务已启动`
- Returns HTTP `409` if another pick task is already running

**Note:** On the first pick the server lazily connects the 4 cameras and loads the
ACT model, so the first request takes a few seconds longer.

---

### Stop the Running Pick Task

```bash
curl -sS -X POST http://localhost:8000/pick/stop
```

Signals the active pick loop to stop. `stopping` is `true` if a task was running.

---

### Pick Task Status

```bash
curl -sS http://localhost:8000/status
```

Returns `{ "running": bool, "last_result": {...} }`. `last_result` includes `steps`
executed and whether the loop was `stopped` early.

---

## 🐙 Quick Commands I Can Run

### Return to initial position
```bash
curl -sS -X POST http://localhost:8000/move/joints \
  -H 'Content-Type: application/json' \
  -d '{"angles":[0,-104,95,65,-95,10]}'
```

### Return to top-down position
```bash
curl -sS -X POST http://localhost:8000/move/joints \
  -H 'Content-Type: application/json' \
  -d '{"angles":[0,-50,30,90,-95,70]}'
```

### Read current position
```bash
curl -sS http://localhost:8000/joints
```

### Disconnect the arm
```bash
curl -sS -X POST http://localhost:8000/disconnect
```

---

## 🛠️ Setup Notes
When pairing your SOARM device with OpenClaw:

1. **Organize the skill directory**

    ```text
    act-pickup/
    ├── references/
    │   ├── config.yaml              # model + camera + loop config (read at startup)
    │   ├── robot_calibration.json   # servo tick ranges used to normalize ACT state
    │   ├── so101_new_calib.urdf     # from TheRobotStudio (optional XYZ readout)
    │   └── openvino/                # exported ACT policy (act.xml / act.bin / manifest.json)
    ├── scripts/
    │   ├── soarm_api.py
    │   └── start_server.sh
    └── SKILL.md
    ```
2. **Prepare the lerobot env** (needs `lerobot`, `physicalai`, `openvino`, `pyyaml`, `flask`, and optionally `pinocchio` for the XYZ readout)
3. **Edit `references/config.yaml`** to set the ACT model settings and the 4 camera
   inputs — `top` (RealSense `serial_number`), `head`/`right`/`left` (USB `fingerprint`
   device paths) — plus loop tuning. Override its location with `SOARM_CONFIG`.
4. **Launch the server** from the skill root:
    ```bash
    cd ~/workspace/openclaw/openclaw-lerobot/docker_volumes/openclaw/.openclaw/workspace/skills/act-pickup
    ./scripts/start_server.sh
    ```
   Then, in OpenClaw, ask it to pick up the object — the skill calls `POST /pick`.

---

