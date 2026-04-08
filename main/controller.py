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

Wrist-invariant OFFSET and APPROACH
-------------------------------------
Both the OFFSET and APPROACH phases compute their TCP-frame command
by pre-rotating the desired base-frame vector using the live wrist pose
at the moment of transition. The streamer's vel_tcp_to_base then cancels
this back to the intended base-frame direction, making both phases
independent of wrist orientation.

  OFFSET  : desired direction in base frame derived from the physical
             camera-to-TCP offset vector rotated by the live wrist pose.
  APPROACH: desired direction is always base-frame +z (port insertion axis).
"""

import time
import math
import threading
import numpy as np
from typing import Optional, Tuple

from detection_types import Detection
from utils import clamp, wrap_to_pi, rotvec_to_matrix
from filters import DetectionKalmanFilter
from controller_logger import ControllerLogger

cmd6 = Tuple[float, float, float, float, float, float]

# ---- Reference-resolution defaults (640x480) ----
_REF_W_D    = 62.5        # desired OBB width  (pixels @ 640x480)
_REF_H_D    = 24.4375     # desired OBB height (pixels @ 640x480)
_REF_DEAD_U = 0.5         # pixel dead-zone    (pixels @ 640x480)
_REF_DEAD_V = 0.5

# ---- State machine states ----
_STATE_SERVO    = 0
_STATE_OFFSET   = 1
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
        # ---- Shared TCP pose from streamer (LatestValue) ----
        tcp_pose_in=None,
        # ---- Camera-to-TCP lateral offset (meters, in camera frame) ----
        # Physical measurement: camera optical centre to TCP
        tcp_offset_x: float = -0.03573,
        tcp_offset_y: float = -0.0348,
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
        dead_theta: float = math.radians(1.5),
        dead_scale: float = 0.01,
        # ---- Velocity limits ----
        vxy_max: float = 0.05,
        vz_max: float = 0.05,
        wz_max: float = 0.6,
        # ---- Kalman filter tuning ----
        kf_sigma_accel_pos: float = 150.0,
        kf_sigma_accel_size: float = 80.0,
        kf_sigma_accel_angle: float = 3.0,
        kf_sigma_meas_pos: float = 15.0,
        kf_sigma_meas_size: float = 15.0,
        kf_sigma_meas_angle: float = 0.1,
        camera_fps: float = 30.0,
        # ---- Final approach parameters ----
        approach_distance_m: float = 0.013,
        approach_speed: float = 0.02,
        offset_speed: float = 0.02,
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

        self._R_cam_to_tcp_3x3 = R

        d = np.linalg.det(R)
        self.T_cam_to_tcp = np.block([
            [R,                np.zeros((3, 3))],
            [np.zeros((3, 3)), d * R           ],
        ])

        # Shared live TCP pose from streamer
        self.tcp_pose_in = tcp_pose_in

        # TCP offset — physical camera-to-TCP displacement in camera frame
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
        self.offset_speed        = offset_speed
        self.converge_dwell_s    = converge_dwell_s

        self._state: int              = _STATE_SERVO
        self._converge_start_t: float = 0.0
        self._offset_start_t: float   = 0.0
        self._offset_duration: float  = 0.0
        self._offset_cmd: cmd6        = (0, 0, 0, 0, 0, 0)
        self._approach_start_t: float = 0.0
        self._approach_cmd: cmd6      = (0, 0, 0, 0, 0, 0)

        # Initialise command buffer
        self.cmd_out.set((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Log effective config
        print(f"[Controller] resolution_scale={s:.3f}  "
              f"fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
        print(f"[Controller] w_d={self.w_d:.1f}  h_d={self.h_d:.1f}  "
              f"dead_u={self.dead_u:.1f}  dead_v={self.dead_v:.1f}  "
              f"camera_fps={camera_fps:.1f}")
        print(f"[Controller] tcp_offset=({tcp_offset_x*100:.3f}, "
              f"{tcp_offset_y*100:.3f}) cm  "
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

        Lu      = np.array([-fx / Z, 0.0,     up / Z,  vp * fx / fy])
        Lv      = np.array([0.0,    -fy / Z,  vp / Z, -up * fy / fx])
        L_sigma = np.array([0.0,     0.0,    -1.0 / Z,  0.0         ])
        L_theta = np.array([0.0,     0.0,     0.0,      -1.0        ])

        return np.vstack([Lu, Lv, L_sigma, L_theta])

    # ------------------------------------------------------------------
    # Helper: get R_tcp_to_base from shared TCP pose
    # ------------------------------------------------------------------
    def _get_R_tcp_to_base(self) -> Optional[np.ndarray]:
        """
        Returns the 3x3 rotation matrix R_tcp_to_base from the live
        TCP pose, or None if the pose is not yet available.
        """
        tcp_pose = self.tcp_pose_in.get() if self.tcp_pose_in is not None else None
        if tcp_pose is None:
            return None
        rx, ry, rz = tcp_pose[3], tcp_pose[4], tcp_pose[5]
        return rotvec_to_matrix(rx, ry, rz)

    # ------------------------------------------------------------------
    # Compute OFFSET phase velocity and duration
    # ------------------------------------------------------------------
    def _prepare_offset(self) -> None:
        """
        Compute the TCP-frame velocity command and duration for the
        lateral OFFSET phase.

        The physical camera-to-TCP offset [tcp_offset_x, tcp_offset_y, 0]
        is expressed in camera frame. It is rotated to TCP frame via
        R_cam_to_tcp, then to base frame via R_tcp_to_base. The base-frame
        vector is then pre-rotated back to TCP frame by R_tcp_to_base.T so
        that when the streamer applies vel_tcp_to_base, the result is the
        correct base-frame direction regardless of wrist orientation.
        """
        R_tcp_to_base = self._get_R_tcp_to_base()

        # camera frame -> TCP frame
        offset_cam = np.array([self.tcp_offset_x, self.tcp_offset_y, 0.0])
        offset_tcp_raw = self._R_cam_to_tcp_3x3 @ offset_cam

        if R_tcp_to_base is None:
            print("[Controller] WARNING: no TCP pose for OFFSET, "
                  "falling back to TCP-frame offset")
            offset_tcp = offset_tcp_raw
        else:
            # TCP frame -> base frame (wrist-invariant world direction)
            offset_base = R_tcp_to_base @ offset_tcp_raw
            # Pre-rotate back to TCP frame so streamer cancels to offset_base
            offset_tcp = R_tcp_to_base.T @ offset_base
            print(f"[Controller] OFFSET base-frame direction: "
                  f"({offset_base[0]*100:.2f}, {offset_base[1]*100:.2f}, "
                  f"{offset_base[2]*100:.2f}) cm")

        dist = np.linalg.norm(offset_tcp)
        if dist < 1e-6:
            self._offset_duration = 0.0
            self._offset_cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return

        self._offset_duration = dist / self.offset_speed
        v = (offset_tcp / dist) * self.offset_speed
        self._offset_cmd = (v[0], v[1], v[2], 0.0, 0.0, 0.0)

        print(f"[Controller] OFFSET prepared: "
              f"direction_tcp=({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}) m/s  "
              f"duration={self._offset_duration:.2f} s  "
              f"distance={dist*100:.2f} cm")

    # ------------------------------------------------------------------
    # Compute APPROACH phase velocity command
    # ------------------------------------------------------------------
    def _prepare_approach(self) -> cmd6:
        """
        Compute the TCP-frame velocity command for the APPROACH phase
        such that the robot always moves along base-frame +z (the port
        insertion axis), regardless of wrist orientation.

        Pre-rotates the desired base-frame +z vector by R_tcp_to_base.T
        so that the streamer's vel_tcp_to_base cancels it back to base +z.
        """
        R_tcp_to_base = self._get_R_tcp_to_base()

        if R_tcp_to_base is None:
            print("[Controller] WARNING: no TCP pose for APPROACH, "
                  "falling back to TCP-frame +z")
            return (0.0, 0.0, self.approach_speed, 0.0, 0.0, 0.0)

        # Desired motion: pure base-frame +z (port insertion axis)
        v_base = np.array([0.0, 0.0, -self.approach_speed])

        # Pre-rotate to TCP frame so streamer maps it back to base +z
        v_tcp = R_tcp_to_base.T @ v_base

        print(f"[Controller] APPROACH prepared: "
              f"direction_tcp=({v_tcp[0]:.4f}, {v_tcp[1]:.4f}, {v_tcp[2]:.4f}) m/s")

        return (float(v_tcp[0]), float(v_tcp[1]), float(v_tcp[2]), 0.0, 0.0, 0.0)

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
                self._approach_cmd = self._prepare_approach()
                self._state = _STATE_APPROACH
                self._approach_start_t = now
                print(f"[Controller] OFFSET complete → APPROACH "
                      f"({self.approach_distance_m*100:.1f} cm at "
                      f"{self.approach_speed:.3f} m/s)")
                _log_blind(self._approach_cmd)
                return self._approach_cmd
            _log_blind(self._offset_cmd)
            return self._offset_cmd

        # ---- APPROACH phase: blind forward motion along base +z ----
        if self._state == _STATE_APPROACH:
            elapsed = now - self._approach_start_t
            needed  = self.approach_distance_m / self.approach_speed
            if elapsed >= needed:
                self._state = _STATE_DONE
                print("[Controller] APPROACH complete → DONE")
                cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                _log_blind(cmd)
                return cmd
            _log_blind(self._approach_cmd)
            return self._approach_cmd

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
            cmd = (0, 0, 0, 0, 0, 0)
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

        # ---- 4. Desired pixel = image centre ----
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