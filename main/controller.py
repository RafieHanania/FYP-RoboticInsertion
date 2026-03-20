import time
import math
import threading
import numpy as np
from typing import Optional, Tuple

from detection_types import Detection
from utils import clamp, wrap_to_pi, EMA
from controller_logger_J import ControllerLogger

cmd6 = Tuple[float, float, float, float, float, float]


class VisualServoController(threading.Thread):
    """
    Image-Based Visual Servo (IBVS) controller combining the full interaction
    matrix (image Jacobian) with the region-based potential energy control law
    from Guo et al. (IEEE T-SMC, 2023).

    --- Changes from Guo et al. ---

    [KEPT] Task variable vector Γ = [u, v, w, h] — bounding box center and
           dimensions directly from the detector, as in eq. (3) of the paper.

    [KEPT] Two-feature-point formulation: center xc = [uc, vc] and top-left
           xl = [ul, vl], with the P-matrix mapping to Γ (eq. 6).

    [KEPT] Region-based potential energy function P_T and its gradient ε as
           the error signal (eqs. 8–16), replacing the fixed-point pixel error.
           This means the controller only corrects when Γ is OUTSIDE the
           desired region, and is zero inside — tolerating aspect ratio changes.

    [CHANGED] Control output is end-effector Cartesian velocity v_cam, not
              joint velocity q̇_r as in eq. (17). This suits velocity-controlled
              manipulators (e.g. UR in speedJ/speedL mode).

    [CHANGED] The interaction matrix L* is built in pixel space (not joint
              space), so the full manipulator Jacobian Jr(q) is not required.
              The paper's J* = P Z⁻¹ A(q) is replaced with L* = P · L_pixel,
              where L_pixel is the standard 4x6 pixel-space image Jacobian.

    [CHANGED] EMA filter is used instead of the paper's LSTM network for
              smoothing bounding box chattering. LSTM gives better accuracy
              (Fig. 6 of paper) but requires offline training data. EMA is a
              practical drop-in with no training requirement.

    [ADDED]   Decoupled wz control for object orientation theta — not present
              in the paper, which does not address in-plane rotation.

    [ADDED]   Depth Z is estimated from bounding box area (thin-lens model)
              rather than a depth sensor. The paper uses an Intel RealSense
              D435i for RGB-D. If a depth sensor is available, pass Z directly.

    [ADDED]   Camera-to-TCP rotation R_cam_to_tcp to support arbitrary camera
              mounting angles. The paper assumes a fixed eye-in-hand mount.

    --- Coordinate Convention ---
    Camera looking down along robot Y-axis, object below TCP.
    Pixel u → robot X, pixel v → robot Z, area/depth → robot Y.
    """

    def __init__(
        self,
        det_in,
        cmd_out,
        stop_event: threading.Event,
        img_w: int,
        img_h: int,
        # Camera intrinsics (need to check again)
        fx: float = 618.072, 
        fy: float = 618.201, 
        cx: float = 318.662,
        cy: float = 240.939,
        # Extrinsic: rotation from camera frame to robot TCP frame (3x3)
        R_cam_to_tcp: np.ndarray = None,
        rate_hz: float = 100.0,
        conf_min: float = 0.2,
        stale_s: float = 0.2,
        # --- Region targets [KEPT from Guo et al. eq. (8)] ---
        # Center desired position (pixels)
        u_d: float = None,   # defaults to img_w / 2
        v_d: float = None,   # defaults to img_h / 2
        # Acceptable center deadband radius (pixels) — eu, ev in the paper
        dead_u: float = 5.0,
        dead_v: float = 5.0,
        # Acceptable bounding box size range (pixels)
        # [CHANGED] replaces single area_d with min/max region bounds
        w_min: float = 20.0,
        w_max: float = 50.0,
        h_min: float = 20.0,
        h_max: float = 50.0,
        # Orientation deadband
        dead_theta: float = math.radians(2.0),
        # Servo gain lambda [KEPT as alpha in the paper eq. (17)]
        lam: float = 0.5,
        # Potential energy gains k1..k6 [KEPT from paper eq. (9)]
        k1: float = 3e-4,   # center u
        k2: float = 3e-4,   # center v
        k3: float = 1e-4,   # w upper bound
        k4: float = 1e-4,   # w lower bound
        k5: float = 1e-4,   # h upper bound
        k6: float = 1e-4,   # h lower bound
        k_approach: float = 0.1, # decoupled depth gain
        # Potential energy exponent N [KEPT from paper eq. (9)], must be > 2
        N: int = 2,
        # Decoupled theta gain [ADDED — not in paper]
        k_theta: float = 0.5,
        # Velocity limits
        vxy_max: float = 0.03,
        vz_max: float = 0.05,
        wz_max: float = 0.6,
        # Depth calibration [ADDED — paper uses RealSense depth]
        Z_d: float = 0.15,


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

        # Extrinsic — build 6x6 velocity transform
        if R_cam_to_tcp is None:
            R = np.array([
                [0,  -1,  0],
                [-1,  0,  0],
                [0, 0,  1],
            ], dtype=float)
        else:
            R = R_cam_to_tcp
        self.T_cam_to_tcp = np.block([
            [R,               np.zeros((3, 3))],
            [np.zeros((3, 3)), R              ]
        ])

        # Region targets [KEPT from Guo et al.]
        self.u_d = u_d if u_d is not None else img_w / 2.0
        self.v_d = v_d if v_d is not None else img_h / 2.0
        self.dead_u = dead_u
        self.dead_v = dead_v
        self.w_min = w_min
        self.w_max = w_max
        self.h_min = h_min
        self.h_max = h_max

        # Potential energy parameters [KEPT from Guo et al. eq. (9)]
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.k_approach = k_approach
        self.N  = N

        # Gains and limits
        self.lam     = lam
        self.k_theta = k_theta
        self.vxy_max = vxy_max
        self.vz_max  = vz_max
        self.wz_max  = wz_max

        # Theta setpoint [ADDED]
        self.theta_d = 0.0
        self.dead_theta = dead_theta

        # Depth calibration [ADDED — paper uses RGB-D sensor]
        self.Z_d    = Z_d
        # area at desired distance — set from calibration log
        self.area_d = (w_min + w_max) / 2.0 * (h_min + h_max) / 2.0

        # [CHANGED] EMA filters instead of LSTM (paper Section IV-A)
        self.ema_u   = EMA(0.2)
        self.ema_v   = EMA(0.2)
        self.ema_w   = EMA(0.3)
        self.ema_h   = EMA(0.3) 
        self.ema_sin = EMA(0.3)
        self.ema_cos = EMA(0.3)

        self.cmd_out.set((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # Depth estimator [ADDED — paper uses RealSense D435i depth channel]
    # ------------------------------------------------------------------
    def _estimate_Z(self, area: float) -> float:
        """
        Z = Z_d * sqrt(area_d / area)  —  thin-lens inverse-square relation.
        Calibrate Z_d and area_d together from a static hold at known distance.
        """
        if area <= 0:
            return self.Z_d
        return self.Z_d * math.sqrt(self.area_d / area)

    # ------------------------------------------------------------------
    # Pixel-space interaction matrix for one point
    # ------------------------------------------------------------------
    def _L_pixel(self, u_px: float, v_px: float, Z: float) -> np.ndarray:
        """
        2x6 interaction matrix for a single pixel-coordinate feature point.
        Derived by scaling the normalized-coordinate L by diag(fx, fy).

        [CHANGED from paper] Paper uses joint-space Jacobian Jr(q).
        Here we stay in Cartesian camera-frame velocity space.
        """
        xn = (u_px - self.cx) / self.fx
        yn = (v_px - self.cy) / self.fy
        fx, fy = self.fx, self.fy

        Lu = np.array([
            -fx/Z,  0,       (u_px - self.cx)/Z,
            (u_px - self.cx)*(v_px - self.cy)/fy,
            -fx*(1 + xn**2),
            (v_px - self.cy)*fx/fy
        ])
        Lv = np.array([
            0,     -fy/Z,   (v_px - self.cy)/Z,
            fy*(1 + yn**2),
            -(u_px - self.cx)*(v_px - self.cy)/fx,
            -(u_px - self.cx)*fy/fx
        ])
        return np.vstack([Lu, Lv])   # 2x6

    # ------------------------------------------------------------------
    # Task interaction matrix L* [KEPT structure from paper eq. (4)–(7)]
    # ------------------------------------------------------------------
    def _task_interaction_matrix(
        self, uc: float, vc: float, ul: float, vl: float, Z: float
    ) -> np.ndarray:
        """
        Build the 4x6 task interaction matrix L* for Γ = [u, v, w, h].

        Two feature points used [KEPT from paper]:
          xc = (uc, vc)  — bounding box center
          xl = (ul, vl)  — bounding box top-left corner

        The P matrix [KEPT from paper eq. (6)]:
          Γ = P * [uc, vc, ul, vl]ᵀ
          P = [[1, 0,  0,  0],
               [0, 1,  0,  0],
               [2, 0, -2,  0],
               [0, 2,  0, -2]]

        Task interaction matrix [KEPT from paper eq. (7)]:
          L* = P · L_full
        where L_full is the 4x6 stacked pixel interaction matrix
        for both feature points.

        [CHANGED] Paper computes J* = P Z⁻¹ A(q) in joint space.
        Here L_full is purely in Cartesian camera-velocity space.
        """
        P = np.array([
            [1,  0,  0,  0],
            [0,  1,  0,  0],
            [2,  0, -2,  0],
            [0,  2,  0, -2],
        ], dtype=float)

        L_center  = self._L_pixel(uc, vc, Z)    # 2x6
        L_topleft = self._L_pixel(ul, vl, Z)    # 2x6
        L_full    = np.vstack([L_center, L_topleft])  # 4x6

        return P @ L_full   # 4x6

    # ------------------------------------------------------------------
    # Region-based potential energy gradient [KEPT from paper eqs. (8)–(16)]
    # ------------------------------------------------------------------
    def _region_error(self, u: float, v: float, w: float, h: float) -> np.ndarray:
        """
        Compute ε = ∂P_T/∂Γ, the gradient of total potential energy P_T
        with respect to the task variable Γ = [u, v, w, h].

        Objective functions [KEPT from paper eq. (8)]:
          f1 = (u - ud)² - eu²  ≤ 0
          f2 = (v - vd)² - ev²  ≤ 0
          f3 = w - w_max        ≤ 0
          f4 = w_min - w        ≤ 0
          f5 = h - h_max        ≤ 0
          f6 = h_min - h        ≤ 0

        Potential energy [KEPT from paper eq. (12)]:
          P_T = P1 + P2 + (P3 + P4)*(P5 + P6)

        [CHANGED] P1, P2 use simplified quadratic form (N=2 special case)
        since center control is a point target (eu, ev act as deadband).
        Size terms P3..P6 use general exponent N from eq. (9).
        """
        N = self.N

        # --- Center errors (simplified quadratic, acts as deadband) ---
        f1 = (u - self.u_d)**2 - self.dead_u**2
        f2 = (v - self.v_d)**2 - self.dead_v**2

        dP1_du = self.k1 * max(0.0, f1)**(N-1) * 2*(u - self.u_d) if f1 > 0 else 0.0
        dP2_dv = self.k2 * max(0.0, f2)**(N-1) * 2*(v - self.v_d) if f2 > 0 else 0.0

        # --- Size constraint functions ---
        f3 = w - self.w_max   # too wide
        f4 = self.w_min - w   # too narrow
        f5 = h - self.h_max   # too tall
        f6 = self.h_min - h   # too short

        P3 = self.k3 * max(0.0, f3)**N / N
        P4 = self.k4 * max(0.0, f4)**N / N
        P5 = self.k5 * max(0.0, f5)**N / N
        P6 = self.k6 * max(0.0, f6)**N / N

        dP3_dw = self.k3 * max(0.0, f3)**(N-1) * (+1) if f3 > 0 else 0.0
        dP4_dw = self.k4 * max(0.0, f4)**(N-1) * (-1) if f4 > 0 else 0.0
        dP5_dh = self.k5 * max(0.0, f5)**(N-1) * (+1) if f5 > 0 else 0.0
        dP6_dh = self.k6 * max(0.0, f6)**(N-1) * (-1) if f6 > 0 else 0.0

        # --- Total gradient [KEPT from paper eq. (15)] ---
        # dP_T/dw = (dP3/dw + dP4/dw) * (P5 + P6)
        # dP_T/dh = (P3 + P4) * (dP5/dh + dP6/dh)
        dPT_du = dP1_du
        dPT_dv = dP2_dv
        dPT_dw = (dP3_dw + dP4_dw) * (P5 + P6)
        dPT_dh = (dP5_dh + dP6_dh) * (P3 + P4) 

        return np.array([dPT_du, dPT_dv, dPT_dw, dPT_dh])

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
    # Main control computation
    # ------------------------------------------------------------------
    def compute_cmd(
        self, det: Optional[Detection], now: float, logger: ControllerLogger
    ) -> cmd6:

        # print(f"[DEBUG] u_d={self.u_d:.1f}  v_d={self.v_d:.1f}  img setpoint check")
        vx = vy = vz = wx = wy = wz = 0.0

        ok = (
            det is not None
            and det.conf >= self.conf_min
            and (now - det.t) <= self.stale_s
        )

        if not ok:
            cmd = (vx, vy, vz, wx, wy, wz)
            logger.log(
                    now=now, det=det, ok=ok,
                    u_f=None, v_f=None, w_f=None, h_f=None, area=None,
                    ul=None, vl=None, Z_est=None,
                    s=None, c=None, theta_f=None,
                    eps=None, e_th=None, v_tcp=None,
                    cmd=cmd,
                )
            return cmd

        # --- Filter measurements [CHANGED: EMA instead of LSTM] ---
        u_f = self.ema_u.update(det.u)
        v_f = self.ema_v.update(det.v)
        w_f = self.ema_w.update(det.w)
        h_f = self.ema_h.update(det.h)

        s = self.ema_sin.update(math.sin(det.theta))
        c = self.ema_cos.update(math.cos(det.theta))
        norm = math.sqrt(s**2 + c**2)
        if norm > 1e-9:
            s /= norm
            c /= norm
        theta_f = math.atan2(s, c)

        # --- Derive top-left corner from center + size ---
        # [KEPT from paper] two feature points: center and top-left
        ul = u_f - w_f / 2.0
        vl = v_f - h_f / 2.0

        # --- Depth estimate from area [ADDED — paper uses depth sensor] ---
        area = w_f * h_f
        Z_est = self._estimate_Z(area)

        # --- Task interaction matrix L* [KEPT structure, CHANGED space] ---
        L_star = self._task_interaction_matrix(u_f, v_f, ul, vl, Z_est)  # 4x6

        # --- Region-based potential energy gradient [KEPT from paper] ---
        eps = self._region_error(u_f, v_f, w_f, h_f)   # 4x1

        # --- Camera-frame velocity [CHANGED: Cartesian, not joint space] ---
        L_pinv = np.linalg.pinv(L_star)                # 6x4
        v_cam  = -self.lam * (L_pinv @ eps)            # 6x1

        # --- Rotate to TCP frame [ADDED] ---
        v_tcp = self.T_cam_to_tcp @ v_cam

        # # --- Decoupled depth (approach) control [ADDED] ---
        # # Target area from desired w/h midpoints
        # w_target = (self.w_min + self.w_max) / 2.0
        # h_target = (self.h_min + self.h_max) / 2.0
        # area_target = w_target * h_target
        # area_error = (area_target - area) / area_target  # normalized, positive = too far

        # if abs(area_error) < 0.05:  # 5% deadband
        #     area_error = 0.0

        # v_approach = clamp(self.k_approach * area_error, -self.vz_max, self.vz_max)

        # # Override the approach axis (vy in TCP frame for your setup)
        # v_tcp[1] = v_approach

        vx = clamp(v_tcp[0], -self.vxy_max, self.vxy_max)
        vy = clamp(v_tcp[1], -self.vxy_max, self.vxy_max)
        vz = clamp(v_tcp[2], -self.vz_max,  self.vz_max)

        # --- Decoupled wz from theta [ADDED — not in paper] ---
        e_th = wrap_to_pi(self.theta_d - theta_f)
        if abs(e_th) < self.dead_theta:
            e_th = 0.0

        # wz = 0.1
        wz = clamp(self.k_theta * e_th, -self.wz_max, self.wz_max)

        cmd = (vx, vy, vz, wx, wy, wz)
        logger.log(
                now=now, det=det, ok=ok,
                u_f=u_f, v_f=v_f, w_f=w_f, h_f=h_f, area=area,
                ul=ul, vl=vl, Z_est=Z_est,
                s=s, c=c, theta_f=theta_f,
                eps=eps, e_th=e_th, v_tcp=v_tcp,
                cmd=cmd,
            )
        
        
        return cmd