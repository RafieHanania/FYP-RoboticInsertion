"""
Vision producer thread — camera capture, inference orchestration,
and video recording.

Delegates detection logic to ``inference_strategy`` (TiledSearch,
ROITracker) and frame drawing to ``annotation``.

The public interface consumed by ``app.py`` is unchanged:
    VisionProducer(det_out, stop_event, img_w, img_h)
    .start()
    .latest_frame   — LatestValue buffer for cv2.imshow on main thread
"""

import traceback
import time
import threading
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from detection_types import Detection
from buffers import LatestValue
from camera_config import max_fps
from inference_strategy import TiledSearch, ROITracker
from annotation import annotate_frame

MODEL_PATH = '../models/saved_runs/train/weights/best.pt'
RECORD_DIR = "../recorded_session"


class VisionProducer(threading.Thread):
    """
    YOLO OBB inference with two-mode adaptive strategy.

    SEARCH — tiled inference for initial acquisition of small /
             distant objects.
    TRACK  — adaptive ROI crop for fast, high-resolution tracking.
    """

    def __init__(self, det_out, stop_event: threading.Event,
                 img_w: int, img_h: int,
                 # ---- Tiled SEARCH tuning ----
                 tile_size: int = 640,
                 tile_overlap: float = 0.25,
                 # ---- ROI TRACK tuning ----
                 roi_margin_mult: float = 3.0,
                 roi_min_half: int = 150,
                 roi_max_misses: int = 10,
                 roi_recheck_interval: int = 30):
        threading.Thread.__init__(self, daemon=True, name="VisionThread")
        self.stop_event = stop_event
        self.det_out = det_out
        self.img_w = img_w
        self.img_h = img_h
        self.t0 = time.time()
        self.latest_frame = LatestValue()

        # ---- YOLO model ----
        self.model = YOLO(MODEL_PATH)

        # ---- Inference strategies ----
        self._searcher = TiledSearch(img_w, img_h, tile_size, tile_overlap)
        self._tracker = ROITracker(img_w, img_h,
                                   margin_mult=roi_margin_mult,
                                   min_half=roi_min_half,
                                   max_misses=roi_max_misses,
                                   recheck_interval=roi_recheck_interval)

        # ---- State ----
        self._mode = "SEARCH"
        self._writer: Optional[cv2.VideoWriter] = None

    # ------------------------------------------------------------------
    # YOLO predict (single entry point for all inference calls)
    # ------------------------------------------------------------------
    def predict(self, frame: np.ndarray):
        return self.model.predict(
            source=frame,
            classes=[2],
            device=0,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------
    def _init_writer(self, frame: np.ndarray, fps: float) -> None:
        os.makedirs(RECORD_DIR, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(self.t0))
        path = os.path.join(RECORD_DIR, f'session_{timestamp}.mp4')

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        print(f"[VisionProducer] Recording to {path}  ({w}x{h} @ {fps:.1f} fps)")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        try:
            cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_h)
            time.sleep(2)

            # Warmup
            for _ in range(30):
                cap.read()

            if not cap.isOpened():
                print("Failed to open Camera")
                return

            fps = max_fps(self.img_w, self.img_h)

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("failed to read from camera")
                    continue

                det: Optional[Detection] = None
                roi: Optional[Tuple[int, int, int]] = None

                # ======================================================
                # SEARCH — tiled inference
                # ======================================================
                if self._mode == "SEARCH":
                    det = self._searcher.search(frame, self.predict)

                    if det is not None:
                        self._mode = "TRACK"
                        self._tracker.reset(det)
                        print(f"[Vision] SEARCH → TRACK  "
                              f"(u={det.u:.0f}, v={det.v:.0f}, "
                              f"conf={det.conf:.2f})")

                # ======================================================
                # TRACK — adaptive ROI crop
                # ======================================================
                else:
                    det, roi = self._tracker.track(frame, self.predict)

                    if self._tracker.lost:
                        self._mode = "SEARCH"
                        print(f"[Vision] TRACK → SEARCH  "
                              f"({self._tracker.miss_count} consecutive misses)")

                # ======================================================
                # Publish detection
                # ======================================================
                if det is not None:
                    self.det_out.set(det)

                # ======================================================
                # Annotate & record
                # ======================================================
                annotated = annotate_frame(
                    frame, self._mode,
                    det=det,
                    roi=roi,
                    tiles=self._searcher.tiles,
                    miss_count=self._tracker.miss_count,
                    max_misses=self._tracker.max_misses,
                )
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