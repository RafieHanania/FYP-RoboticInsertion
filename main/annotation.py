"""
Frame annotation helpers for VisionProducer.

All functions take a frame (np.ndarray) and draw on it in-place or
return a modified copy.  No state — purely rendering utilities.
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detection_types import Detection


# ---- USB-A target overlay sizing ----
_TARGET_RATIO = 12.0 / 4.5
_TARGET_AREA  = 1000.0
TARGET_W = int(round(math.sqrt(_TARGET_AREA * _TARGET_RATIO)))
TARGET_H = int(round(math.sqrt(_TARGET_AREA / _TARGET_RATIO)))


def draw_target_overlay(frame: np.ndarray) -> np.ndarray:
    """
    Draw a centred target rectangle representing the desired
    USB-A port size, with a semi-transparent fill and crosshair.
    """
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


def draw_obb(frame: np.ndarray, det: Detection,
             color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2) -> None:
    """Draw the oriented bounding box polygon and angle label."""
    half_w, half_h = det.w / 2.0, det.h / 2.0
    cos_t, sin_t = math.cos(det.theta), math.sin(det.theta)

    corners = []
    for dx, dy in [(-half_w, -half_h), (half_w, -half_h),
                   (half_w, half_h), (-half_w, half_h)]:
        rx = cos_t * dx - sin_t * dy + det.u
        ry = sin_t * dx + cos_t * dy + det.v
        corners.append([int(round(rx)), int(round(ry))])

    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True,
                  color=color, thickness=thickness)

    angle_deg = math.degrees(det.theta)
    cv2.putText(frame, f"{angle_deg:.1f} deg",
                (int(det.u), int(det.v) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)


def draw_roi(frame: np.ndarray,
             roi: Tuple[int, int, int],
             color: Tuple[int, int, int] = (255, 255, 0),
             thickness: int = 1) -> None:
    """Draw the ROI crop rectangle."""
    rx1, ry1, rside = roi
    cv2.rectangle(frame, (rx1, ry1), (rx1 + rside, ry1 + rside),
                  color, thickness)


def draw_tile_grid(frame: np.ndarray,
                   tiles: List[Tuple[int, int, int]],
                   color: Tuple[int, int, int] = (255, 0, 255),
                   thickness: int = 1) -> None:
    """Draw the SEARCH-mode tile grid."""
    for tx, ty, ts in tiles:
        cv2.rectangle(frame, (tx, ty), (tx + ts, ty + ts),
                      color, thickness)


def draw_mode_label(frame: np.ndarray, mode: str,
                    miss_count: int = 0, max_misses: int = 0,
                    n_tiles: int = 0) -> None:
    """Draw the current mode label in the top-left corner."""
    if mode == "TRACK":
        label = f"TRACK  miss={miss_count}/{max_misses}"
        color = (0, 255, 0)
    else:
        label = f"SEARCH  ({n_tiles} tiles)"
        color = (0, 0, 255)

    cv2.putText(frame, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA)


def annotate_frame(frame: np.ndarray,
                   mode: str,
                   det: Optional[Detection] = None,
                   roi: Optional[Tuple[int, int, int]] = None,
                   tiles: Optional[List[Tuple[int, int, int]]] = None,
                   miss_count: int = 0,
                   max_misses: int = 0) -> np.ndarray:
    """
    Compose all overlays onto a copy of the frame.

    This is the single entry point called by VisionProducer each cycle.
    """
    annotated = frame.copy()
    annotated = draw_target_overlay(annotated)

    if roi is not None:
        draw_roi(annotated, roi)

    if det is not None:
        draw_obb(annotated, det)

    n_tiles = len(tiles) if tiles else 0
    draw_mode_label(annotated, mode,
                    miss_count=miss_count,
                    max_misses=max_misses,
                    n_tiles=n_tiles)

    if mode == "SEARCH" and tiles:
        draw_tile_grid(annotated, tiles)

    return annotated