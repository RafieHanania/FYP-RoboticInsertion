import traceback
import time
import threading
from typing import Optional
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
import math
import os

from detection_types import Detection
from buffers import LatestValue

MODEL_PATH = '../models/saved_runs/train/weights/best.pt'

# Target box: USB-A port aspect ratio 12:4.5 area = 1000px^2
_TARGET_RATIO = 12.0 / 4.5
_TARGET_AREA  = 1000.0
TARGET_W = int(round(math.sqrt(_TARGET_AREA * _TARGET_RATIO)))
TARGET_H = int(round(math.sqrt(_TARGET_AREA / _TARGET_RATIO)))

RECORD_DIR = "../recorded_session"


class VisionProducer(threading.Thread):
    """
    Provide inference using YOLO OBB.
    """

    def __init__(self, det_out, stop_event: threading.Event, img_w: int, img_h: int):
        threading.Thread.__init__(self, daemon=True, name="VisionThread")
        self.stop_event = stop_event
        self.det_out = det_out
        self.img_w = img_w
        self.img_h = img_h
        self.t0 = time.time()
        self.model = YOLO(MODEL_PATH)
        self.latest_frame = LatestValue()

        self._writer: Optional[cv2.VideoWriter] = None

    def _init_writer(self, frame: np.ndarray, fps: float) -> None:
        os.makedirs(RECORD_DIR, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(self.t0))
        path = os.path.join(RECORD_DIR, f'session_{timestamp}.mp4')

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        print(f"[VisionProducer] Recording to {path}  ({w}x{h} @ {fps:.1f} fps)")

    def _draw_target_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw a centered target rectangle representing the desired USB-A port size"""
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        hw, hh = TARGET_W // 2, TARGET_H // 2

        x1, y1 = cx - hw, cy - hh
        x2, y2 = cx + hw, cy + hh

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        GAP, ARM = 2, 6
        cv2.line(frame, (cx - ARM - GAP, cy), (cx - GAP, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx + GAP, cy), (cx + ARM, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx, cy - ARM - GAP), (cx, cy - GAP), (0, 255, 255), 1)
        cv2.line(frame, (cx, cy + GAP), (cx, cy + ARM + GAP), (0, 255, 255), 1)

        return frame

    @staticmethod
    def _canonicalize_obb(w, h, theta):
        """Collapse all equivalent OBB representations to |θ| ≤ 45°."""
        theta = (theta + math.pi / 2) % math.pi - math.pi / 2

        if theta > math.pi / 4:
            w, h = h, w
            theta -= math.pi / 2
        elif theta < -math.pi / 4:
            w, h = h, w
            theta += math.pi / 2

        return w, h, theta

    def predict(self, frame):
        results = self.model.predict(
            source=frame,
            classes=[2],
            device=0,
            verbose=False
        )
        return results

    def run(self):
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_h)
            time.sleep(2)

            # Warmup
            for _ in range(30):
                cap.read()

            if not cap.isOpened():
                print("Failed to open Camera")
                return

            fps = 30.0

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("failed to read from camera")
                    continue

                results = self.predict(frame)

                for result in results:
                    annotated = result.plot(
                        conf=False,
                        line_width=1,
                        labels=False
                    )
                    annotated = self._draw_target_overlay(annotated)

                    if result.obb is not None and len(result.obb) > 0:
                        xywhr_np = result.obb.xywhr.cpu().numpy()
                        conf_np = result.obb.conf.cpu().numpy()
                        for (u, v, w, h, theta), conf in zip(xywhr_np, conf_np):
                            w, h, theta = self._canonicalize_obb(w, h, theta)

                            angle_deg = math.degrees(theta)
                            cv2.putText(annotated, f"{angle_deg:.1f} deg",
                                        (int(u), int(v) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 1, cv2.LINE_AA)

                            det = Detection(
                                t=time.time(),
                                u=u, v=v, w=w, h=h,
                                theta=theta, conf=conf,
                            )
                            self.det_out.set(det)

                    self.latest_frame.set(annotated)

                    if self._writer is None:
                        self._init_writer(annotated, fps)
                    self._writer.write(annotated)

        except BaseException as e:
            print(f"[VisionProducer] CRASHED: {e}")
            traceback.print_exc()

        finally:
            if self._writer is not None:
                self._writer.release()
            cap.release()