import csv
import os
import numpy as np
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
            # --- Filtered measurements ---
            "u_f",
            "v_f",
            "w_f",
            "h_f",
            "area",
            "sin_f",
            "cos_f",
            "theta_f",
            # --- Depth estimate (from area) ---
            "Z_est",
            # --- Feature values  s = [u, v, ln_sigma, theta] ---
            "s_u",
            "s_v",
            "s_ln_sigma",
            "s_theta",
            # --- Feature errors  e = s - s*  (post-deadband) ---
            "e_u",
            "e_v",
            "e_sigma",
            "e_theta",
            # --- L_red diagnostics ---
            "L_red_det",
            "L_red_cond",
            # --- Camera-frame velocity (4-DOF, pre-rotation) ---
            "v_cam_vx",
            "v_cam_vy",
            "v_cam_vz",
            "v_cam_wz",
            # --- TCP-frame velocity (6-DOF, pre-clamp) ---
            "v_tcp_vx",
            "v_tcp_vy",
            "v_tcp_vz",
            "v_tcp_wx",
            "v_tcp_wy",
            "v_tcp_wz",
            # --- Final clamped command ---
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
        # Filtered measurements
        u_f: Optional[float],
        v_f: Optional[float],
        w_f: Optional[float],
        h_f: Optional[float],
        area: Optional[float],
        sin_f: Optional[float],
        cos_f: Optional[float],
        theta_f: Optional[float],
        # Depth (area-based estimate)
        Z_est: Optional[float],
        # Feature values
        ln_sigma: Optional[float],
        # Feature errors (post-deadband)
        e: Optional[np.ndarray],
        # L_red diagnostics
        L_red: Optional[np.ndarray],
        # Camera-frame velocity
        v_cam: Optional[np.ndarray],
        # TCP-frame velocity (pre-clamp)
        v_tcp: Optional[np.ndarray],
        # Final command
        cmd: tuple,
    ):
        if self._closed:
            raise RuntimeError("Cannot log to a closed ControllerLogger")

        vx, vy, vz, wx, wy, wz = cmd

        # L_red diagnostics
        if L_red is not None:
            try:
                L_det = np.linalg.det(L_red)
            except Exception:
                L_det = None
            try:
                L_cond = np.linalg.cond(L_red)
            except Exception:
                L_cond = None
        else:
            L_det = None
            L_cond = None

        self.csv_writer.writerow([
            now,
            # Raw detection
            ok,
            None if det is None else det.conf,
            None if det is None else det.t,
            None if det is None else det.u,
            None if det is None else det.v,
            None if det is None else det.w,
            None if det is None else det.h,
            None if det is None else det.w * det.h,
            None if det is None else det.theta,
            # Filtered
            u_f,
            v_f,
            w_f,
            h_f,
            area,
            sin_f,
            cos_f,
            theta_f,
            # Depth
            Z_est,
            # Feature values
            u_f,
            v_f,
            ln_sigma,
            theta_f,
            # Feature errors
            None if e is None else e[0],
            None if e is None else e[1],
            None if e is None else e[2],
            None if e is None else e[3],
            # L_red diagnostics
            L_det,
            L_cond,
            # Camera-frame velocity
            None if v_cam is None else v_cam[0],
            None if v_cam is None else v_cam[1],
            None if v_cam is None else v_cam[2],
            None if v_cam is None else v_cam[3],
            # TCP-frame velocity
            None if v_tcp is None else v_tcp[0],
            None if v_tcp is None else v_tcp[1],
            None if v_tcp is None else v_tcp[2],
            None if v_tcp is None else v_tcp[3],
            None if v_tcp is None else v_tcp[4],
            None if v_tcp is None else v_tcp[5],
            # Final command
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