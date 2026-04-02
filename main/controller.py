"""
Classical Image-Based Visual Servoing (IBVS) controller.

Control law:   v_cmd = -lambda * L_red^-1 * e

Feature vector:  s = [u, v, ln_sigma, theta]
  - u, v      : bounding box center (pixels)
  - ln_sigma  : log scale ratio  ln(sqrt(w*h) / sqrt(w_d*h_d))
  - theta     : OBB orientation (radians)

Interaction matrix L_red is the 4x4 sub-matrix of the full
6-column interaction matrix, keeping only the columns for the
4 DOF we command: (vx, vy, vz, wz).

Depth Z is estimated from bounding box area using the thin-lens
model.  Errors in Z only affect velocity magnitude, not direction.

Resolution scaling
------------------
All pixel-domain defaults (w_d, h_d, dead_u, dead_v) are specified
at the *reference* resolution (640x480).  The ``resolution_scale``
parameter (ratio of current fx to reference fx, supplied by app.py
via camera_config) automatically scales them to the active resolution.
Intrinsics (fx, fy, cx, cy) are passed in directly — no hardcoded
defaults.

State machine
-------------
SERVO    — IBVS: centre object at image centre + match scale & angle
OFFSET   — blind lateral move to align TCP with target
APPROACH — blind forward motion for final insertion
DONE     — zero velocity, task complete
"""

import time
import math
import threading
import numpy as np
from typing import Optional, Tuple

from detection_types import Detection
from utils import clamp, wrap_to_pi
from filters import DetectionKalmanFilter
from controller_logger import ControllerLogger

cmd6 = Tuple[float, float, float, float, float, float]

# ---- Reference-resolution defaults (640x480) ----
_REF_W_D    = 62.5        # desired OBB width  (pixels @ 640x480)
_REF_H_D    = 24.4375     # desired OBB height (pixels @ 640x480)
_REF_DEAD_U = 1.0       # pixel dead-zone    (pixels @ 640x480)
_REF_DEAD_V = 1.0

# ---- State machine states ----
_STATE_SERVO    = 0
_STATE_OFFSET   = 1      # CHANGED — new state
_STATE_APPROACH = 2
_STATE_DONE     = 3


class VisualServoController(threading.Thread):

    def __init__(
        self,
        det_in,
        cmd_out,
        stop_event: threading.Event,
        img_w: int,
        img_h: int,
        # ---- Camera intrinsics (REQUIRED) ----
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        # ---- Resolution scale factor (from camera_config) ----
        resolution_scale: float = 1.0,
        # ---- Extrinsic: camera frame -> TCP frame (3x3 rotation) ----
        R_cam_to_tcp: np.ndarray = None,
        # ---- Controller parameters ----
        rate_hz: float = 100.0,
        conf_min: float = 0.2,
        stale_s: float = 0.2,
        # ---- Camera-to-TCP lateral offset (meters, in camera frame) ----
        # Used during OFFSET phase to shift TCP over target after centering
        tcp_offset_x: float = -0.0385,
        tcp_offset_y: float = -0.0335,
        theta_d: float = 0.0,
        # ---- Desired bounding box size at target distance ----
        w_d: float = None,
        h_d: float = None,
        # ---- Depth calibration ----
        Z_d: float = 0.117,
        # ---- IBVS gain lambda ----
        lam: float = 0.5,
        # ---- Dead-zones ----
        dead_u: float = None,
        dead_v: float = None,
        dead_theta: float = math.radians(2.0),
        dead_scale: float = 0.01,
        # ---- Velocity limits ----
        vxy_max: float = 0.05,
        vz_max: float = 0.05,
        wz_max: float = 0.6,
        # ---- Kalman filter tuning ----
        kf_sigma_accel_pos: float = 150.0,
        # Continuous acceleration noise
        kf_sigma_accel_size: float = 80.0,
        kf_sigma_accel_angle: float = 3.0,
        kf_sigma_meas_pos: float = 15.0,
        kf_sigma_meas_size: float = 15.0,
        kf_sigma_meas_angle: float = 0.1,
        camera_fps: float = 30.0,
        # ---- Final approach parameters ----
        approach_distance_m: float = 0.01,
        approach_speed: float = 0.02,
        offset_speed: float = 0.02,          # CHANGED — lateral speed during OFFSET
        converge_dwell_s: float = 0.5,
    ):
        threading.Thread.__init__(self, daemon=True, name="ControllerThread")
        self.det_in = det_in
        self.cmd_out = cmd_out
        self.stop_event = stop_event
        self.rate_hz = rate_hz
        self.conf_min = conf_min
        self.stale_s = stale_s
        self.img_w = img_w
        self.img_h = img_h

        s = resolution_scale

        # Intrinsics (passed in, no defaults)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Extrinsic — 6x6 velocity rotation block
        if R_cam_to_tcp is None:
            R = np.array([
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1],
            ], dtype=float)
        else:
            R = np.asarray(R_cam_to_tcp, dtype=float)

        self._R_cam_to_tcp_3x3 = R                                # CHANGED — keep 3x3 for OFFSET

        d = np.linalg.det(R)
        self.T_cam_to_tcp = np.block([
            [R,                np.zeros((3, 3))],
            [np.zeros((3, 3)), d * R           ],
        ])

        # TCP offset — used in OFFSET phase only (not during SERVO) # CHANGED
        self.tcp_offset_x = tcp_offset_x
        self.tcp_offset_y = tcp_offset_y

        self.theta_d = theta_d
        self.w_d     = w_d if w_d is not None else _REF_W_D * s
        self.h_d     = h_d if h_d is not None else _REF_H_D * s
        self.area_d  = self.w_d * self.h_d

        # Depth calibration
        self.Z_d = Z_d

        # Gain
        self.lam = lam

        # Dead-zones
        self.dead_u     = dead_u if dead_u is not None else _REF_DEAD_U * s
        self.dead_v     = dead_v if dead_v is not None else _REF_DEAD_V * s
        self.dead_theta = dead_theta
        self.dead_scale = dead_scale

        # Velocity limits
        self.vxy_max = vxy_max
        self.vz_max  = vz_max
        self.wz_max  = wz_max

        # Kalman filter
        self.kf = DetectionKalmanFilter(
            dt=1.0 / camera_fps,
            sigma_accel_pos=kf_sigma_accel_pos,
            sigma_accel_size=kf_sigma_accel_size,
            sigma_accel_angle=kf_sigma_accel_angle,
            sigma_meas_pos=kf_sigma_meas_pos,
            sigma_meas_size=kf_sigma_meas_size,
            sigma_meas_angle=kf_sigma_meas_angle,
        )
        self._last_tick_t: float = 0.0
        self._last_det_stamp: float = 0.0

        # ---- State machine ----
        self.approach_distance_m = approach_distance_m
        self.approach_speed      = approach_speed
        self.offset_speed        = offset_speed                    # CHANGED
        self.converge_dwell_s    = converge_dwell_s

        self._state: int              = _STATE_SERVO
        self._converge_start_t: float = 0.0
        self._offset_start_t: float   = 0.0                       # CHANGED
        self._offset_duration: float  = 0.0                       # CHANGED
        self._offset_cmd: cmd6        = (0, 0, 0, 0, 0, 0)        # CHANGED
        self._approach_start_t: float = 0.0

        # Initialise command buffer
        self.cmd_out.set((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Log effective config
        print(f"[Controller] resolution_scale={s:.3f}  "
              f"fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
        print(f"[Controller] w_d={self.w_d:.1f}  h_d={self.h_d:.1f}  "
              f"dead_u={self.dead_u:.1f}  dead_v={self.dead_v:.1f}  "
              f"camera_fps={camera_fps:.1f}")
        print(f"[Controller] tcp_offset=({tcp_offset_x*100:.1f}, "
              f"{tcp_offset_y*100:.1f}) cm  "
              f"approach={approach_distance_m*100:.1f} cm @ "
              f"{approach_speed:.3f} m/s  "
              f"offset_speed={offset_speed:.3f} m/s  "
              f"dwell={converge_dwell_s:.2f} s")

    # ------------------------------------------------------------------
    # Depth estimator
    # ------------------------------------------------------------------
    def _estimate_Z(self, area: float) -> float:
        if area <= 0:
            return self.Z_d
        return self.Z_d * math.sqrt(self.area_d / area)

    # ------------------------------------------------------------------
    # 4x4 reduced interaction matrix
    # ------------------------------------------------------------------
    def _build_L_reduced(
        self, u_px: float, v_px: float, Z: float
    ) -> np.ndarray:
        fx, fy = self.fx, self.fy
        up = u_px - self.cx
        vp = v_px - self.cy

        Lu = np.array([-fx / Z, 0.0, up / Z, vp * fx / fy])
        Lv = np.array([0.0, -fy / Z, vp / Z, -up * fy / fx])
        L_sigma = np.array([0.0, 0.0, -1.0 / Z, 0.0])
        L_theta = np.array([0.0, 0.0, 0.0, -1.0])

        return np.vstack([Lu, Lv, L_sigma, L_theta])

    # ------------------------------------------------------------------
    # Compute OFFSET phase velocity and duration                    # CHANGED
    # ------------------------------------------------------------------
    def _prepare_offset(self) -> None:
        """
        Compute the TCP-frame velocity command and duration for the
        lateral OFFSET phase.

        The offset [tcp_offset_x, tcp_offset_y, 0] is in camera frame.
        Rotate it to TCP frame, then normalise to offset_speed.
        """
        offset_cam = np.array([self.tcp_offset_x, self.tcp_offset_y, 0.0])
        offset_tcp = self._R_cam_to_tcp_3x3 @ offset_cam

        dist = np.linalg.norm(offset_tcp)
        if dist < 1e-6:
            self._offset_duration = 0.0
            self._offset_cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return

        self._offset_duration = dist / self.offset_speed

        # Unit direction scaled by speed
        v = (offset_tcp / dist) * self.offset_speed
        self._offset_cmd = (v[0], v[1], v[2], 0.0, 0.0, 0.0)

        print(f"[Controller] OFFSET prepared: "
              f"direction_tcp=({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}) m/s  "
              f"duration={self._offset_duration:.2f} s  "
              f"distance={dist*100:.1f} cm")

    # ------------------------------------------------------------------
    # Thread entry
    # ------------------------------------------------------------------
    def run(self):
        dt = 1.0 / self.rate_hz
        next_t = time.time()
        logger = ControllerLogger(log_dir="../logging_archive")

        try:
            while not self.stop_event.is_set():
                now = time.time()
                det: Optional[Detection] = self.det_in.get()
                cmd = self.compute_cmd(det, now, logger)
                self.cmd_out.set(cmd)
                print(cmd)

                next_t += dt
                sleep_s = next_t - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.time()
        finally:
            logger.close()

    # ------------------------------------------------------------------
    # Core control computation
    # ------------------------------------------------------------------
    def compute_cmd(
        self, det: Optional[Detection], now: float, logger: ControllerLogger
    ) -> cmd6:

        # ---- Helper for logging blind phases (no vision data) ----  # CHANGED
        def _log_blind(cmd):
            logger.log(
                now=now, state=self._state, det=det, ok=False,
                u_f=None, v_f=None, w_f=None, h_f=None, area=None,
                sin_f=None, cos_f=None, theta_f=None,
                Z_est=None, u_d=None, v_d=None, ln_sigma=None,
                e=None, L_red=None, v_cam=None, v_tcp=None,
                cmd=cmd,
            )

        # ---- OFFSET phase: blind lateral move ----
        if self._state == _STATE_OFFSET:
            elapsed = now - self._offset_start_t
            if elapsed >= self._offset_duration:
                self._state = _STATE_APPROACH
                self._approach_start_t = now
                print(f"[Controller] OFFSET complete → APPROACH "
                      f"({self.approach_distance_m*100:.1f} cm at "
                      f"{self.approach_speed:.3f} m/s)")
                cmd = (0.0, 0.0, self.approach_speed, 0.0, 0.0, 0.0)
                _log_blind(cmd)
                return cmd
            _log_blind(self._offset_cmd)
            return self._offset_cmd

        # ---- APPROACH phase: blind forward motion ----
        if self._state == _STATE_APPROACH:
            elapsed = now - self._approach_start_t
            needed  = self.approach_distance_m / self.approach_speed
            if elapsed >= needed:
                self._state = _STATE_DONE
                print("[Controller] APPROACH complete → DONE")
                cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                _log_blind(cmd)
                return cmd
            cmd = (0.0, 0.0, self.approach_speed, 0.0, 0.0, 0.0)
            _log_blind(cmd)
            return cmd

        # ---- DONE phase: hold zero ----
        if self._state == _STATE_DONE:
            cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            _log_blind(cmd)
            return cmd

        # ---- SERVO phase: IBVS with target = image centre ----
        vx = vy = vz = wx = wy = wz = 0.0

        ok = (
            det is not None
            and det.conf >= self.conf_min
            and (now - det.t) <= self.stale_s
        )

        # ---- 1. Kalman predict + update ----
        dt_pred = now - self._last_tick_t if self._last_tick_t > 0 else None
        self._last_tick_t = now
        if self.kf.initialised:
            self.kf.predict(dt=dt_pred)

        if ok and det.t != self._last_det_stamp:
            self.kf.update(det)
            self._last_det_stamp = det.t

        if not self.kf.initialised:
            cmd = (0,0,0,0,0,0)
            logger.log(
                now=now, state=self._state, det=det, ok=ok,
                u_f=None, v_f=None, w_f=None, h_f=None, area=None,
                sin_f=None, cos_f=None, theta_f=None,
                Z_est=None, u_d=None, v_d=None, ln_sigma=None,
                e=None, L_red=None, v_cam=None, v_tcp=None,
                cmd=cmd,
            )
            return cmd

        u_f, v_f, w_f, h_f, theta_f = self.kf.state()

        s_sin = self.kf.x[4]
        s_cos = self.kf.x[5]

        # ---- 2. Compute current feature values ----
        area = w_f * h_f

        if area > 0 and self.area_d > 0:
            ln_sigma = 0.5 * math.log(area / self.area_d)
        else:
            ln_sigma = 0.0

        # ---- 3. Estimate depth from area ----
        Z_est = self._estimate_Z(area)

        # ---- 4. Desired pixel = image centre ----                 # CHANGED
        u_d = self.cx
        v_d = self.cy

        # ---- 5. Compute feature error  e = s - s*  ----
        e_u     = u_f - u_d
        e_v     = v_f - v_d
        e_sigma = ln_sigma
        e_theta = wrap_to_pi(theta_f - self.theta_d)

        # Check raw convergence BEFORE zeroing by dead-zone
        all_converged = (
            abs(e_u)     < self.dead_u
            and abs(e_v)     < self.dead_v
            and abs(e_sigma) < self.dead_scale
            and abs(e_theta) < self.dead_theta
        )

        if abs(e_u) < self.dead_u:
            e_u = 0.0
        if abs(e_v) < self.dead_v:
            e_v = 0.0
        if abs(e_sigma) < self.dead_scale:
            e_sigma = 0.0
        if abs(e_theta) < self.dead_theta:
            e_theta = 0.0

        # ---- Convergence dwell check ----
        if all_converged:
            if self._converge_start_t == 0.0:
                self._converge_start_t = now
                print("[Controller] Errors in dead-zone — dwell timer started")
            elif (now - self._converge_start_t) >= self.converge_dwell_s:
                # ---- Transition: SERVO → OFFSET ----              # CHANGED
                self._prepare_offset()
                self._state = _STATE_OFFSET
                self._offset_start_t = now
                print("[Controller] SERVO converged → OFFSET")
                return self._offset_cmd
        else:
            if self._converge_start_t != 0.0:
                self._converge_start_t = 0.0

        e = np.array([e_u, e_v, e_sigma, e_theta])

        # ---- 6. Build 4x4 reduced interaction matrix ----
        # Evaluate L at the midpoint (s + s*) / 2.0
        u_mid = (u_f + u_d) / 2.0
        v_mid = (v_f + v_d) / 2.0
        Z_mid = (Z_est + self.Z_d) / 2.0
        L_red = self._build_L_reduced(u_mid, v_mid, Z_mid)

        # ---- 7. Classical IBVS control law ----
        try:
            v_cam = -self.lam * np.linalg.solve(L_red, e)
        except np.linalg.LinAlgError:
            v_cam = -self.lam * (np.linalg.pinv(L_red) @ e)

        # ---- 8. Map camera-frame velocity -> TCP-frame velocity ----
        v_cam_6 = np.array([
            v_cam[0], v_cam[1], v_cam[2],
            0.0, 0.0, v_cam[3],
        ])
        v_tcp = self.T_cam_to_tcp @ v_cam_6

        # ---- 9. Clamp to velocity limits ----
        vx = clamp(v_tcp[0], -self.vxy_max, self.vxy_max)
        vy = clamp(v_tcp[1], -self.vxy_max, self.vxy_max)
        vz = clamp(v_tcp[2], -self.vz_max,  self.vz_max)
        wx = 0.0
        wy = 0.0
        wz = clamp(v_tcp[5], -self.wz_max,  self.wz_max)

        cmd = (vx, vy, vz, wx, wy, wz)

        logger.log(
            now=now, state=self._state, det=det, ok=ok,
            u_f=u_f, v_f=v_f, w_f=w_f, h_f=h_f, area=area,
            sin_f=s_sin, cos_f=s_cos, theta_f=theta_f,
            Z_est=Z_est, u_d=u_d, v_d=v_d, ln_sigma=ln_sigma,
            e=e, L_red=L_red, v_cam=v_cam, v_tcp=v_tcp,
            cmd=cmd,
        )

        return cmd