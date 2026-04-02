import time 
import math
import threading
from typing import Optional, Tuple

from detection_types import Detection 
from utils import clamp, wrap_to_pi, EMA
from main.controller_logger import ControllerLogger

cmd6 = Tuple[float, float, float, float, float, float]

class VisualServoController(threading.Thread):
    """
    Image-Based Visual Servo (IBVS) controller for a robotic manipulator
    with a downward-facing camera aligned along the robot's Y-axis.

    The controller runs at a fixed rate in a background thread, consuming
    Detection objects and publishing 6-DOF velocity commands (vx, vy, vz, wx, wy, wz).

    --- Coordinate Convention ---
    The camera is mounted looking down along the robot Y-axis, with the target
    object placed directly below the TCP. Axes are mapped as follows:

        Image u (horizontal)  →  robot X  (vx)
        Image v (vertical)    →  robot Z  (vz)
        Bounding box area     →  robot Y  (vy, depth/approach axis)
        Object angle theta    →  robot Wz (wz, yaw rotation)

    --- Control Law ---
    Four decoupled proportional controllers run in parallel:

        vx  = -k_u    * e_u       e_u    = u_filtered - u_desired   [pixels]
        vz  = -k_v    * e_v       e_v    = v_filtered - v_desired   [pixels]
        vy  = +k_area * e_area    e_area = log(area / area_desired)  [log-ratio]
        wz  =  k_theta * e_theta  e_theta = wrap(theta_desired - theta_filtered) [rad]

    Area error uses a log-ratio so that the gain is symmetric: the same gain
    produces the same speed whether the object is 2x too close or 2x too far.
    The sign of vy is positive because the object is below the TCP — a positive
    area error (object too close) must move the arm away (+Y), not toward it.

    --- Robustness Features ---
    - Detections are rejected if confidence < conf_min or older than stale_s.
    - Deadbands suppress small errors to prevent jitter near the setpoint.
    - EMA filters smooth noisy pixel and angle measurements before use.
    - All velocity outputs are clamped to safe hardware limits.

    --- Limitations ---
    This is a decoupled axis-aligned controller. It assumes the camera and
    robot axes are aligned as described above. For arbitrary camera mounting
    angles, a full interaction matrix (image Jacobian) approach is required.
    """

    def __init__(
        self, 
        det_in,
        cmd_out,
        stop_event: threading.Event,
        img_w: int,
        img_h: int,
        rate_hz: float = 100.0,
        conf_min: float = 0.2,
        stale_s: float = 0.2,
        dead_px: float = 2.0,
        dead_theta: float = math.radians(2.0),
        dead_area : float = 0.05 # log-ratio deadband (5% area difference)
    ):
        threading.Thread.__init__(self, daemon=True, name="ControllerThread")
        self.det_in = det_in
        self.cmd_out = cmd_out
        self.stop_event = stop_event
        self.rate_hz = rate_hz

        self.conf_min = conf_min
        self.stale_s = stale_s
        self.dead_px = dead_px
        self.dead_theta = dead_theta
        self.dead_area = dead_area

        # Targets (tune later)
        self.u_d = img_w / 2.0
        self.v_d = img_h / 2.0
        self.area_d = 4500.0
        self.theta_d = 0.0
    
        # Limits
        self.vxy_max = 0.03
        self.vz_max = 0.03
        self.wz_max = 0.6

        # Gains (start small)
        self.k_u = 0.0006
        self.k_v = 0.0006
        self.k_area = 0.0000133 # vz_max / area_d
        self.k_theta = 0.01
        # k_area vz_max is reached when object is 2x or 0.5x the target area
        # log(2) = 0.693 -> k_area = vz_max / log(2)
        self.k_area = self.vz_max / math.log(2) 


        # Filters
        self.ema_u = EMA(0.2)
        self.ema_v = EMA(0.2)
        self.ema_w = EMA(0.3)
        self.ema_h = EMA(0.3)
        self.ema_sin = EMA(0.3)
        self.ema_cos = EMA(0.3)

        self.cmd_out.set((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def run(self):
        dt = 1.0 / self.rate_hz
        next_t = time.time()

        logger = ControllerLogger("controller_log.csv")

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

    def compute_cmd(self, det: Optional[Detection], now: float, logger: ControllerLogger) -> cmd6:
        vx = vy = vz = wx = wy = wz = 0.0

        ok = (
            det is not None and
            det.conf >= self.conf_min and
            (now - det.t) <= self.stale_s
        )

        if not ok:
            cmd = (vx, vy, vz, wx, wy, wz)
            # --- no-detection / stale path ---
            logger.log(
                now=now, det=det, ok=ok,
                u_f=None, v_f=None, w_f=None, h_f=None, area=None,
                Z_est=None,
                s=None, c=None, theta_f=None,
                e_u=None, e_v=None, e_w=None, e_h=None, e_th=None,
                cmd=cmd,
            )
            return cmd

        u_f = self.ema_u.update(det.u)
        v_f = self.ema_v.update(det.v)

        area = self.ema_w.update(det.w * det.h)

        s = self.ema_sin.update(math.sin(det.theta))
        c = self.ema_cos.update(math.cos(det.theta))
        theta_f = math.atan2(s, c)

        e_u = u_f - self.u_d
        e_v = v_f - self.v_d
        e_area = math.log(area / self.area_d)
        e_th = wrap_to_pi(self.theta_d - theta_f)

        if abs(e_u) < self.dead_px: e_u = 0.0
        if abs(e_v) < self.dead_px: e_v = 0.0
        if abs(e_area) < self.dead_area: e_area = 0.0
        if abs(e_th) < self.dead_theta: e_th = 0.0

        vx = clamp(-self.k_u * e_u, -self.vxy_max, self.vxy_max)
        # target object alligned at y-axis and under the TCP
        vz = clamp(-self.k_v * e_v, -self.vxy_max, self.vxy_max)
        vy = clamp(self.k_area * e_area, -self.vz_max, self.vz_max)
        wz = clamp(self.k_theta * e_th, -self.wz_max, self.wz_max)

        cmd = (vx, vy, vz, wx, wy, wz)

        # --- normal control path ---
        logger.log(
            now=now, det=det, ok=ok,
            u_f=u_f, v_f=v_f, w_f=w_f, h_f=h_f, area=area,
            Z_est=Z_est,
            s=s, c=c, theta_f=theta_f,
            e_u=eps[0], e_v=eps[1], e_w=eps[2], e_h=eps[3], e_th=e_th,
            cmd=cmd,
        )

        return cmd
        return (0.0, 0.0, 0.01, 0.0, 0.0, 0.0)  # does robot move toward or away?

