# Vision-Guided Robotic Insertion with YOLOv11-OBB and UR5e

> Final Year Project — Nanyang Technological University  
> A closed-loop image-based visual servoing (IBVS) system that autonomously aligns and inserts a USB-A plug into its port using a pretrained CNN detector and a 6-DOF collaborative robot.

<!-- Replace these badges with live ones if you host the repo publicly -->
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Robot](https://img.shields.io/badge/Robot-UR5e-red)
![Camera](https://img.shields.io/badge/Camera-RealSense%20D435i-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Classical peg-in-hole insertion using a USBA port and PLUG through a **perception-driven closed loop** built on three ideas:

1. **YOLOv11-OBB** fine-tuned on a port dataset provides oriented bounding boxes — giving the controller angular information that axis-aligned detectors cannot.
2. A **Kalman-filtered IBVS controller** drives the error between the current and desired image features to zero in real time, without needing absolute 3-D pose.
3. A **three-thread asynchronous architecture** decouples camera, controller, and robot communication so that slow inference cannot stall the 100 Hz control loop.

The task scope is restricted to **4 DOF** (x, y, z translation + yaw), under the planar-plug assumption between the gripped plug and the port face.

> **Demo:** `docs/demo.mp4` — full convergence + insertion sequence  
> **Demo:** `docs/demo_overlay.mp4` — camera overlay with desired positions

---

## Key Features

- Real-time YOLOv11-OBB inference on GPU with **adaptive SEARCH → TRACK strategy**
  - *SEARCH*: SAHI-style tiled inference for small / distant targets
  - *TRACK*: adaptive ROI cropping around the last detection for low-latency tracking
- **Classical IBVS** with a four-feature vector `s = [u, v, ln(σ), θ]ᵀ` and analytically derived reduced interaction matrix
- **Kalman filter** smoothing on `[u, v, w, h, sin θ, cos θ]` with sin/cos angle encoding to avoid wrap-around discontinuities
- **Area-based depth estimation** — avoids stereo bias from the D435i's depth module
- **State machine**: `SERVO → OFFSET → APPROACH → DONE` handles the blind-insertion phase after visual convergence
- **100 Hz RTDE streaming** with heartbeat watchdog failsafe on the UR5e
- **Full-state CSV logging** + **annotated MP4 recording** per run, for post-hoc analysis
- **Three-state state machine:** SERVO: Closed-loop vision-based control --> OFFSET: Open-loop lateral camera principal axis to TCP error adjustment --> APPROACH: Open-loop depth adjustment

---

## System Architecture

```
┌──────────────────────┐    Detection    ┌────────────────────────┐   6-DOF cmd   ┌─────────────────┐
│   VisionThread       │ ───────────────▶│  ControllerThread      │──────────────▶│ StreamerThread  │
│  (VisionProducer)    │   latest_det    │  (VisualServoCtrl)     │   latest_cmd  │ (RTDEStreamer)  │
│                      │                 │                        │               │                 │
│ • RealSense capture  │                 │ • Kalman filter        │               │ • RTDE 100 Hz   │
│ • YOLOv11-OBB infer  │                 │ • IBVS control law     │               │ • Watchdog      │
│ • SEARCH / TRACK mode│                 │ • TCP-frame transform  │               │ • speedl() cmd  │
│ • MP4 recording      │                 │ • State machine        │               │                 │
└──────────────────────┘                 └────────────────────────┘               └─────────────────┘
        ↓ native FPS                              ↓ 100 Hz                                ↓ 100 Hz
```

All inter-thread data flow is through **single-slot, lock-protected `LatestValue` buffers** — only the most recent value is kept, stale data is silently dropped. Cooperative shutdown is via a shared `threading.Event`.

---

## Hardware Requirements

| Component | Model / Spec | Notes |
|---|---|---|
| Robot | Universal Robots **UR5e** | 6-DOF, e-Series, 5 kg payload |
| Camera | Intel **RealSense D435i** | RGB stream only; depth is estimated from OBB area |
| GPU | CUDA-capable (NVIDIA) | Developed on RTX 5060 Laptop |
| Host OS | Windows 10/11 | Uses `cv2.CAP_DSHOW` backend |
| Network | Direct Ethernet to UR5e | Robot IP default `169.254.194.220` |

Gripper-mounted eye-in-hand mount and plug fixture are documented in the FYP report (Chapter 1.3).

---

## Software Requirements

Tested on Python 3.10+.

```bash
pip install -r requirements.txt
```

Core dependencies:

- `ultralytics` — YOLOv11-OBB inference
- `torch` (CUDA build) — GPU backend
- `opencv-python` — camera capture, annotation, MP4 writer
- `numpy`, `scipy` — Kalman filter math, rotation utilities
- `pyrealsense2` — (if migrating off the OpenCV DSHOW backend)
- The RTDE Python library from Universal Robots (vendored under `rtde/`)

---

## Installation

```bash
# 1. Clone
git clone https://github.com/<user>/<repo>.git
cd <repo>

# 2. Create venv (recommended)
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Obtain the fine-tuned weights
#    Place best.pt at:  models/saved_runs/train/weights/best.pt
#    (See the Fine-Tuning section below if you want to retrain.)
```

---

## Configuration

### Robot-side (UR5e)

1. On the UR teach pendant, load the provided URScript program that runs `speedl()` on the six `input_double_register_*` values and watches the heartbeat register.
2. Confirm the robot's IP and ensure it is reachable from the host:
   ```bash
   ping 169.254.194.220
   ```
3. Start the URScript program on the UR5e **before** launching the host-side script.

### Host-side (controller)

All user-tunable parameters live at the top of `main/main.py`:

```python
ROBOT_IP     = "169.254.194.220" # May change for every robot session
RECIPE_PATH  = "control_loop_configuration.xml"
RATE_HZ      = 100
IMAGE_WIDTH  = 1280         # Supported: 424x240, 640x480, 1280x720, 1920x1080
IMAGE_HEIGHT = 720          # All pixel-domain gains auto-scale from the reference 640x480
```

The RTDE recipe file (`control_loop_configuration.xml`) declares three register groups:

- **`state`** — subscribes to `actual_TCP_pose` for base-frame velocity rotation
- **`setp`** — six `input_double_register` slots for the 6-DOF velocity command
- **`watchdog`** — command-valid flag + monotonic heartbeat counter

---

## Usage

Run the full pipeline:

```bash
cd main
python main.py
```

**On startup:**
- The RealSense warms up for 30 frames to stabilise exposure.
- A live OpenCV window titled `OBB-Detection` opens on the main thread, showing:
  - SEARCH-mode tile grid (magenta) *or* TRACK-mode ROI (green)
  - Oriented bounding box + angle label on the detected port
  - Desired-feature rectangle at the image centre

**Controls:**
- `q` — stop all threads, close the robot session cleanly, and save logs

**Expected behaviour:**
1. `SERVO` — the arm aligns the port within the image dead-zone
2. `OFFSET` — blind lateral move to the camera-to-TCP offset
3. `APPROACH` — forward plunge until the plug is seated
4. `DONE` — zero velocity held indefinitely until `q`

---

## Project Structure

```
.
├── main/
│   ├── main.py                    # Entry point — edit hardware config here
│   ├── app.py                     # VisualServoApp: thread orchestration
│   ├── vision.py                  # VisionProducer: camera + YOLO + recording
│   ├── controller.py              # VisualServoController: IBVS + state machine
│   ├── rtde_streamer.py           # RTDEStreamer: 100 Hz write + watchdog
│   ├── filters.py                 # DetectionKalmanFilter (12-state CV model)
│   ├── inference_strategy.py      # TiledSearch + ROITracker
│   ├── annotation.py              # Overlay rendering for the live feed
│   ├── buffers.py                 # LatestValue single-slot thread-safe buffer
│   ├── camera_config.py           # D435i intrinsics lookup by resolution
│   ├── detection_types.py         # Detection dataclass
│   ├── controller_logger.py       # Per-cycle CSV writer (all state variables)
│   ├── utils.py                   # clamp, wrap_to_pi, rotvec_to_matrix, vel_tcp_to_base
│   └── control_loop_configuration.xml   # RTDE recipe
├── rtde/                          # Vendored UR RTDE Python client
├── models/saved_runs/train/weights/best.pt   # YOLOv11-OBB fine-tuned weights
├── ur_programs/                   # URScript running on the UR5e teach pendant
├── notebooks/
│   └── yolo_obb_finetune.ipynb    # Training / evaluation on the port dataset
├── recorded_session/              # Annotated MP4s written per run (gitignored)
├── logging_archive/               # Per-cycle controller CSVs (gitignored)
├── docs/
│   ├── figures/                   # Architecture diagrams, experimental plots
│   └── report.pdf                 # Full FYP report
├── requirements.txt
└── README.md
```

---

## Key Control Parameters

Defaults measured at the reference resolution (640×480) and automatically rescaled for other resolutions via a single focal-length ratio.

| Parameter | Symbol | Default | Units | Role |
|---|---|---|---|---|
| IBVS proportional gain | λ | 0.5 | s⁻¹ | Convergence rate |
| Desired depth | Z_d | 0.117 | m | Calibration distance at which `w_d, h_d` were measured |
| Desired OBB width | w_d,ref | 62.5 | px @640×480 | Reference desired width |
| Desired OBB height | h_d,ref | 24.44 | px @640×480 | Reference desired height |
| Dead-zone (u, v) | δ_u, δ_v | 1.0 | px @640×480 | Pixel error tolerance |
| Dead-zone (scale) | δ_σ | 0.01 | — | Log-scale tolerance (~1% size) |
| Dead-zone (angle) | δ_θ | 2.0 | deg | Orientation tolerance |
| Max lateral velocity | v_xy,max | 0.05 | m/s | Clamp on v_x, v_y |
| Max approach velocity | v_z,max | 0.03 | m/s | Clamp on v_z |
| Max yaw rate | ω_z,max | 0.6 | rad/s | Clamp on ω_z |
| Confidence threshold | c_min | 0.2 | — | Minimum YOLO confidence |
| Staleness threshold | t_stale | 0.2 | s | Max detection age |
| Convergence dwell | t_dwell | 0.5 | s | Time in dead-zone before state transition |
| TCP offset (x, y) | d_x, d_y | −0.0385, −0.0335 | m | Camera-to-TCP lateral offset |
| Control rate | f_ctrl | 100 | Hz | RTDE / controller loop |
| Camera frame rate | f_cam | 15 | Hz | At 1280×720 |

Full parameter list with rationales is in Chapter 4.9 of the report.

---

## YOLOv11-OBB Fine-Tuning

The detector is fine-tuned from Ultralytics' DOTAv1 pretrained weights on a 6-class port dataset (1,609 images: USB-A, USB-C, HDMI, Ethernet, VGA, Display Port), relabelled with oriented bounding boxes.

To reproduce:

```bash
jupyter notebook notebooks/yolo_obb_finetune.ipynb
```

Key training hyperparameters: `epochs=100`, `batch=-1` (auto), `imgsz=640`, `workers=2`.  
Hardware: NVIDIA RTX 5060 Laptop.

---

## Outputs

Every run writes two artefacts:

- **`recorded_session/<timestamp>.mp4`** — the annotated live feed (OBB, ROI / tile grid, mode label, desired rectangle).
- **`logging_archive/controller_log_<timestamp>.csv`** — one row per control cycle. Columns include raw detection, Kalman-filtered state, depth estimate, feature error, reduced interaction matrix diagnostics (det, cond), camera- and TCP-frame velocities, and the final clamped command.

These feed directly into the plotting notebooks under `notebooks/` for convergence, Kalman performance, and repeatability analysis.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RTDE start failed` | UR URScript not running, or wrong IP / port | Start the URScript on the pendant first; verify `ping` |
| Robot halts immediately with protective stop | Watchdog heartbeat not incrementing | Check `StreamerThread` is alive; no upstream stall |
| YOLO returns no detections in SEARCH | Weights not found, or object outside FOV | Confirm `best.pt` path; check camera pose |
| Camera opens but shows a black frame | DSHOW backend warmup not completed | Wait 30 frames; confirm D435i is not in use by another process |
| Controller runs but robot does not move | Command-valid flag stuck at 0 | Inspect `watchdog` register values in RTDE trace |
| Large steady-state pixel error | Dead-zone too tight vs. detector noise | Raise `δ_u`, `δ_v`, or increase KF measurement noise |

---

## Known Limitations

- The planar-plug assumption is required — the controller does not command pitch or roll of the plug relative to the port face.
- Single-target tracking only; multiple identical ports in view are handled by highest-confidence selection, not identity association.
- Depth is inferred from OBB area, so accuracy degrades if the port's apparent size deviates from the calibration (e.g. occlusion, extreme viewing angle).
- Insertion is open-loop after visual convergence (no force feedback); jamming is not detected.

Future work directions are discussed in Chapter 6 of the report.

---

## Citation

Key references that informed the design (full list in the report):

- J. Guo *et al.*, "CNN-Based Robot Control for an Eye-in-Hand Camera," *IEEE Trans. Syst. Man Cybern. Syst.*, vol. 53, no. 8, 2023.
- F. Chaumette and S. Hutchinson, "Visual servo control. I. Basic approaches," *IEEE RAM*, 2006.
- F. C. Akyon *et al.*, "Slicing Aided Hyper Inference" (SAHI), 2022.
- Y. Zhang *et al.*, "ByteTrack," ECCV 2022.
- Ultralytics, *YOLO Oriented Bounding Boxes Documentation*, 2025.

---

## Acknowledgements

- FYP supervisor: **Cheah Chien Chern**, NTU
- Ultralytics for the YOLOv11-OBB framework and DOTAv1 pretrained weights
- Universal Robots for the RTDE Python client
- Port dataset: Kaggle (relabelled for OBB annotation)

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).  
The YOLOv11 weights and RTDE client retain their respective upstream licenses.

---

## Contact

**Rafie Hanania Hertrian** — hananiarafie@gmail.com  
Final Year Project, Electrical and Electronic Engineering, Nanyang Technological University
