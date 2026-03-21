import csv
import os
from datetime import datetime
from typing import Optional

from detection_types import Detection

LOG_DIR = "../logging_archive"


class ControllerLogger:
    def __init__(self, log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(log_dir, f"controller_log_{timestamp}.csv")

        self.csv_path = csv_path
        self.csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self._closed = False

        self.csv_writer.writerow([
            # --- Timing ---
            "time",
            # --- Raw detection ---
            "det_valid",
            "det_conf",
            "det_t",
            "det_u",
            "det_v",
            "det_w",
            "det_h",
            "det_area",
            "det_theta",
            # --- Filtered task variables Γ = [u, v, w, h] ---
            "u_f",
            "v_f",
            "w_f",
            "h_f",
            "area",         # w_f * h_f
            # --- Derived geometry ---
            "ul",           # top-left u = u_f - w_f/2
            "vl",           # top-left v = v_f - h_f/2
            "Z_est",        # thin-lens depth estimate
            # --- Filtered orientation ---
            "sin_f",
            "cos_f",
            "theta_f",
            # --- Potential energy gradient eps = dP_T/dGamma ---
            "eps_u",        # eps[0]: center u gradient
            "eps_v",        # eps[1]: center v gradient
            "eps_w",        # eps[2]: width bounds gradient
            "eps_h",        # eps[3]: height bounds gradient
            # --- Decoupled theta error (post-deadband) ---
            "e_th",
            # --- Pre-clamp TCP velocities (to diagnose saturation) ---
            "v_tcp_x",
            "v_tcp_y",
            "v_tcp_z",
            "v_tcp_wx",
            "v_tcp_wy",
            "v_tcp_wz",
            # --- Final clamped output ---
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
        ])
        self.csv_file.flush()

    def log(
        self,
        now: float,
        det: Optional[Detection],
        ok: bool,
        u_f: Optional[float],
        v_f: Optional[float],
        w_f: Optional[float],
        h_f: Optional[float],
        area: Optional[float],
        ul: Optional[float],
        vl: Optional[float],
        Z_est: Optional[float],
        s: Optional[float],
        c: Optional[float],
        theta_f: Optional[float],
        eps: Optional[object],  # np.ndarray shape (4,) or None
        e_th: Optional[float],
        v_tcp: Optional[object],  # np.ndarray shape (6,) or None
        cmd: tuple,
    ):
        if self._closed:
            raise RuntimeError("Cannot log to a closed ControllerLogger")

        vx, vy, vz, wx, wy, wz = cmd

        self.csv_writer.writerow([
            now,
            ok,
            None if det is None else det.conf,
            None if det is None else det.t,
            None if det is None else det.u,
            None if det is None else det.v,
            None if det is None else det.w,
            None if det is None else det.h,
            None if det is None else det.w * det.h,
            None if det is None else det.theta,
            u_f,
            v_f,
            w_f,
            h_f,
            area,
            ul,
            vl,
            Z_est,
            s,
            c,
            theta_f,
            None if eps is None else eps[0],
            None if eps is None else eps[1],
            None if eps is None else eps[2],
            None if eps is None else eps[3],
            e_th,
            None if v_tcp is None else v_tcp[0],
            None if v_tcp is None else v_tcp[1],
            None if v_tcp is None else v_tcp[2],
            None if v_tcp is None else v_tcp[3],
            None if v_tcp is None else v_tcp[4],
            None if v_tcp is None else v_tcp[5],
            vx,
            vy,
            vz,
            wx,
            wy,
            wz,
        ])
        self.csv_file.flush()

    def close(self):
        if not self._closed:
            self.csv_file.flush()
            self.csv_file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False