"""
Classical Image-Based Visual Servoing (IBVS) controller.

Control law:   v_cmd = -λ · L_red⁻¹ · e

Feature vector:  s = [u, v, ln_σ, θ]
  - u, v    : bounding box center (pixels)
  - ln_σ    : log scale ratio  ln(√(w·h) / √(w*·h*))
               using log linearises the depth relationship
  - θ       : OBB orientation (radians)

Interaction matrix L_red is the 4×4 sub-matrix of the full
6-column interaction matrix, keeping only the columns for the
4 DOF we command: (vx, vy, vz, ωz).

Depth Z is estimated from bounding box area using the thin-lens
model.  Errors in Z only affect velocity magnitude, not direction —
IBVS convergence is guaranteed regardless of depth accuracy.

Reference: Chaumette & Hutchinson, "Visual Servo Control,
Part I: Basic Approaches", IEEE RAM, 2006.
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


class VisualServoController(threading.Thread):

    def __init__(
        self,
        det_in,
        cmd_out,
        stop_event: threading.Event,
        img_w: int,
        img_h: int,
        # ---- Camera intrinsics ----
        fx: float = 618.072,
        fy: float = 618.201,
        cx: float = 318.662,
        cy: float = 240.939,
        # ---- Extrinsic: camera frame → TCP frame (3×3 rotation) ----
        R_cam_to_tcp: np.ndarray = None,
        # ---- Controller parameters ----
        rate_hz: float = 100.0,
        conf_min: float = 0.2,
        stale_s: float = 0.2,
        # ---- Desired feature values ----
        u_d: float = None,              # image center by default
        v_d: float = None,
        theta_d: float = 0.0,          # desired OBB angle (rad)
        # ---- Desired bounding box size at target distance ----
        w_d: float = 35.0,             # desired OBB width  (pixels)
        h_d: float = 35.0,             # desired OBB height (pixels)
        # ---- Depth calibration ----
        Z_d: float = 0.15,             # depth (m) at which w_d, h_d were measured
        # ---- IBVS gain λ  ----
        lam: float = 0.5,
        # ---- Dead-zones ----
        dead_u: float = 2.0,           # pixels
        dead_v: float = 2.0,
        dead_theta: float = math.radians(2.0),
        dead_scale: float = 0.05,      # ln-scale (≈ 5% size tolerance)
        # ---- Velocity limits ----
        vxy_max: float = 0.05,
        vz_max: float = 0.03,
        wz_max: float = 0.6,
        # ---- Kalman filter tuning ----
        kf_sigma_pos: float = 2.0,
        kf_sigma_vel: float = 5.0,
        kf_sigma_meas_pos: float = 5.0,
        kf_sigma_meas_size: float = 5.0,
        kf_sigma_meas_angle: float = 0.1,
        camera_fps: float = 30.0,
    ):
        threading.Thread.__init__(self, daemon=True, name="ControllerThread")
        self.det_in = det_in
        self.cmd_out = cmd_out
        self.stop_event = stop_event
        self.rate_hz = rate_hz
        self.conf_min = conf_min
        self.stale_s = stale_s

        # Intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx if cx is not None else img_w / 2.0
        self.cy = cy if cy is not None else img_h / 2.0

        # Extrinsic — 6×6 velocity rotation block
        if R_cam_to_tcp is None:
            R = np.array([
                [ 0, -1,  0],
                [-1,  0,  0],
                [ 0,  0,  1],
            ], dtype=float)
        else:
            R = np.asarray(R_cam_to_tcp, dtype=float)
        self.T_cam_to_tcp = np.block([
            [R,                np.zeros((3, 3))],
            [np.zeros((3, 3)), R               ],
        ])   # 6×6

        # Desired features
        self.u_d     = u_d if u_d is not None else img_w / 2.0
        self.v_d     = v_d if v_d is not None else img_h / 2.0
        self.theta_d = theta_d
        self.w_d     = w_d
        self.h_d     = h_d
        self.area_d  = w_d * h_d

        # Depth calibration
        self.Z_d = Z_d

        # Gain
        self.lam = lam

        # Dead-zones
        self.dead_u     = dead_u
        self.dead_v     = dead_v
        self.dead_theta = dead_theta
        self.dead_scale = dead_scale

        # Velocity limits
        self.vxy_max = vxy_max
        self.vz_max  = vz_max
        self.wz_max  = wz_max

        # Kalman filter (replaces 6 separate EMA filters)
        self.kf = DetectionKalmanFilter(
            dt=1.0 / camera_fps,
            sigma_pos=kf_sigma_pos,
            sigma_vel=kf_sigma_vel,
            sigma_meas_pos=kf_sigma_meas_pos,
            sigma_meas_size=kf_sigma_meas_size,
            sigma_meas_angle=kf_sigma_meas_angle,
        )
        self._last_det_t: float = 0.0

        # Initialise command buffer
        self.cmd_out.set((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # Depth estimator — thin-lens approximation from bounding box area
    #
    #   Z = Z_d · √(area_d / area)
    #
    # Errors in Z only scale the velocity magnitude, not its direction.
    # IBVS converges regardless of depth accuracy (Chaumette, 2006).
    # ------------------------------------------------------------------
    def _estimate_Z(self, area: float) -> float:
        if area <= 0:
            return self.Z_d
        return self.Z_d * math.sqrt(self.area_d / area)

    # ------------------------------------------------------------------
    # 4×4 reduced interaction matrix
    #
    #   Features:  s = [u, v, ln_σ, θ]
    #   Commands:  v = [vx, vy, vz, ωz]   (columns 0,1,2,5 of full L)
    # ------------------------------------------------------------------
    def _build_L_reduced(
        self, u_px: float, v_px: float, Z: float
    ) -> np.ndarray:
        fx, fy = self.fx, self.fy
        up = u_px - self.cx
        vp = v_px - self.cy

        Lu = np.array([
            -fx / Z,
             0.0,
             up / Z,
             vp * fx / fy,
        ])

        Lv = np.array([
             0.0,
            -fy / Z,
             vp / Z,
            -up * fy / fx,
        ])

        L_sigma = np.array([
             0.0,
             0.0,
            -1.0 / Z,
             0.0,
        ])

        L_theta = np.array([
             0.0,
             0.0,
             0.0,
            -1.0,
        ])

        return np.vstack([Lu, Lv, L_sigma, L_theta])   # 4×4

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

        vx = vy = vz = wx = wy = wz = 0.0

        ok = (
            det is not None
            and det.conf >= self.conf_min
            and (now - det.t) <= self.stale_s
        )

        if not ok:
            if self.kf.initialised:
                dt_pred = now - self._last_det_t if self._last_det_t > 0 else None
                self.kf.predict(dt=dt_pred)
            cmd = (vx, vy, vz, wx, wy, wz)
            logger.log(
                now=now, det=det, ok=ok,
                u_f=None, v_f=None, w_f=None, h_f=None, area=None,
                sin_f=None, cos_f=None, theta_f=None,
                Z_est=None, ln_sigma=None,
                e=None, L_red=None, v_cam=None, v_tcp=None,
                cmd=cmd,
            )
            return cmd

        # ---- 1. Kalman predict + update ----
        dt_pred = now - self._last_det_t if self._last_det_t > 0 else None
        self._last_det_t = now

        self.kf.predict(dt=dt_pred)
        self.kf.update(det)

        u_f, v_f, w_f, h_f, theta_f = self.kf.state()

        s_sin = self.kf.x[4]
        s_cos = self.kf.x[5]

        # ---- 2. Compute current feature values ----
        area = w_f * h_f

        if area > 0 and self.area_d > 0:
            ln_sigma = 0.5 * math.log(area / self.area_d)
        else:
            ln_sigma = 0.0

        # ---- 3. Compute feature error  e = s - s*  ----
        e_u     = u_f - self.u_d
        e_v     = v_f - self.v_d
        e_sigma = ln_sigma
        e_theta = wrap_to_pi(theta_f - self.theta_d)

        if abs(e_u) < self.dead_u:
            e_u = 0.0
        if abs(e_v) < self.dead_v:
            e_v = 0.0
        if abs(e_sigma) < self.dead_scale:
            e_sigma = 0.0
        if abs(e_theta) < self.dead_theta:
            e_theta = 0.0

        e = np.array([e_u, e_v, e_sigma, e_theta])

        # ---- 4. Estimate depth from area ----
        Z_est = self._estimate_Z(area)

        # ---- 5. Build 4×4 reduced interaction matrix ----
        L_red = self._build_L_reduced(u_f, v_f, Z_est)

        # ---- 6. Classical IBVS control law ----
        try:
            v_cam = -self.lam * np.linalg.solve(L_red, e)
        except np.linalg.LinAlgError:
            v_cam = -self.lam * (np.linalg.pinv(L_red) @ e)

        # ---- 7. Map camera-frame velocity → TCP-frame velocity ----
        v_cam_6 = np.array([
            v_cam[0],
            v_cam[1],
            v_cam[2],
            0.0,
            0.0,
            v_cam[3],
        ])
        v_tcp = self.T_cam_to_tcp @ v_cam_6

        # ---- 8. Clamp to velocity limits ----
        vx = clamp(v_tcp[0], -self.vxy_max, self.vxy_max)
        vy = clamp(v_tcp[1], -self.vxy_max, self.vxy_max)
        vz = clamp(v_tcp[2], -self.vz_max,  self.vz_max)
        wx = 0.0
        wy = 0.0
        wz = clamp(v_tcp[5], -self.wz_max,  self.wz_max)

        cmd = (vx, vy, vz, wx, wy, wz)

        logger.log(
            now=now, det=det, ok=ok,
            u_f=u_f, v_f=v_f, w_f=w_f, h_f=h_f, area=area,
            sin_f=s_sin, cos_f=s_cos, theta_f=theta_f,
            Z_est=Z_est, ln_sigma=ln_sigma,
            e=e, L_red=L_red, v_cam=v_cam, v_tcp=v_tcp,
            cmd=cmd,
        )

        return cmd