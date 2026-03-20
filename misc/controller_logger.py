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
            "time",
            "det_valid",
            "det_conf",
            "det_t",
            "det_u",
            "det_v",
            "det_w",
            "det_h",
            "det_theta",
            "u_filtered",
            "v_filtered",
            "area_filtered",
            "sin_filtered",
            "cos_filtered",
            "theta_filtered",
            "e_u",
            "e_v",
            "e_area",
            "e_theta",
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
        area: Optional[float],
        s: Optional[float],
        c: Optional[float],
        theta_f: Optional[float],
        e_u: Optional[float],
        e_v: Optional[float],
        e_area: Optional[float],
        e_th: Optional[float],
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
            None if det is None else det.theta,
            u_f,
            v_f,
            area,
            s,
            c,
            theta_f,
            e_u,
            e_v,
            e_area,
            e_th,
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
        return False  # do not suppress exceptions