"""
Inference strategies for YOLO OBB detection.

Provides two inference modes used by VisionProducer:

  TiledSearch  — overlapping tile grid for initial acquisition of
                 small / distant objects.  Runs full-frame first,
                 falls back to per-tile inference only when needed.

  ROITracker   — adaptive square crop around last known position for
                 fast, high-resolution tracking once an object has
                 been acquired.

Both classes operate on raw frames and return ``Detection`` objects
in full-frame coordinates.  The YOLO model is passed in — neither
class owns it.

Coordinate remapping
--------------------
All crops are **square** so the resize scaling is uniform and the
OBB orientation angle is preserved exactly.
"""

import math
import time
from typing import List, Optional, Tuple

import numpy as np

from detection_types import Detection


# =====================================================================
# OBB helpers
# =====================================================================

def canonicalize_obb(w: float, h: float, theta: float
                     ) -> Tuple[float, float, float]:
    """
    Canonicalize OBB so that w >= h, with theta in (-90°, +90°].

    This ensures a consistent convention where *w* is always the
    longer (USB-A lengthwise) dimension.
    """
    theta = (theta + math.pi / 2) % math.pi - math.pi / 2

    if h > w:
        w, h = h, w
        theta += math.pi / 2 if theta <= 0 else -math.pi / 2

    return w, h, theta


def remap_to_full_frame(u_crop: float, v_crop: float,
                        w_crop: float, h_crop: float,
                        theta_crop: float,
                        x1: int, y1: int, side: int,
                        model_input_w: int, model_input_h: int
                        ) -> Tuple[float, float, float, float, float]:
    """
    Map OBB detection from model-input-of-crop space back to
    full-frame pixel coordinates.

    Because the crop is square and YOLO's internal letterboxing
    preserves aspect ratio, the effective scale is uniform and
    theta is preserved exactly.
    """
    scale_x = side / model_input_w
    scale_y = side / model_input_h

    u_full = u_crop * scale_x + x1
    v_full = v_crop * scale_y + y1
    w_full = w_crop * scale_x
    h_full = h_crop * scale_y
    theta_full = theta_crop

    return u_full, v_full, w_full, h_full, theta_full


def extract_best(results, x1: int = 0, y1: int = 0,
                 side: Optional[int] = None) -> Optional[Detection]:
    """
    Extract the highest-confidence OBB detection from YOLO results.

    Parameters
    ----------
    results : ultralytics Results list
    x1, y1  : top-left of crop in full-frame pixels (0 if full-frame)
    side    : crop side length.  If *None* the results are assumed to
              already be in full-frame coordinates.

    Returns
    -------
    Detection in full-frame coordinates, or None.
    """
    for result in results:
        if result.obb is None or len(result.obb) == 0:
            continue

        xywhr_np = result.obb.xywhr.cpu().numpy()
        conf_np  = result.obb.conf.cpu().numpy()

        best_idx = int(conf_np.argmax())
        u, v, w, h, theta = xywhr_np[best_idx]
        conf = float(conf_np[best_idx])

        w, h, theta = canonicalize_obb(w, h, theta)

        if side is not None:
            model_h, model_w = result.orig_shape   # (H, W)
            u, v, w, h, theta = remap_to_full_frame(
                u, v, w, h, theta,
                x1, y1, side, model_w, model_h,
            )

        return Detection(
            t=time.time(),
            u=float(u), v=float(v),
            w=float(w), h=float(h),
            theta=float(theta), conf=conf,
        )

    return None


# =====================================================================
# TiledSearch
# =====================================================================

class TiledSearch:
    """
    Overlapping-tile inference for initial acquisition of small objects.

    At construction the tile grid is pre-computed from the frame size,
    tile size, and overlap fraction.  Calling ``search()`` first tries
    a cheap full-frame pass; only if that fails does it iterate over
    tiles.
    """

    def __init__(self, img_w: int, img_h: int,
                 tile_size: int = 640,
                 overlap: float = 0.25):
        self.img_w = img_w
        self.img_h = img_h
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiles = self._generate_tiles(img_w, img_h, tile_size, overlap)

        print(f"[TiledSearch] {len(self.tiles)} tiles "
              f"({tile_size}px, {overlap:.0%} overlap) "
              f"for {img_w}x{img_h}")

    # ----- tile generation -----

    @staticmethod
    def _generate_tiles(img_w: int, img_h: int,
                        tile_size: int, overlap: float
                        ) -> List[Tuple[int, int, int]]:
        """
        Pre-compute (x1, y1, side) square tiles covering the frame.
        """
        if img_w <= tile_size and img_h <= tile_size:
            side = min(img_w, img_h)
            return [(0, 0, side)]

        stride = max(1, int(tile_size * (1.0 - overlap)))

        xs = list(range(0, img_w - tile_size + 1, stride))
        if not xs or xs[-1] + tile_size < img_w:
            xs.append(max(0, img_w - tile_size))

        ys = list(range(0, img_h - tile_size + 1, stride))
        if not ys or ys[-1] + tile_size < img_h:
            ys.append(max(0, img_h - tile_size))

        return list(dict.fromkeys(
            (x, y, tile_size) for y in ys for x in xs
        ))

    # ----- main entry point -----

    def search(self, frame: np.ndarray, predict_fn) -> Optional[Detection]:
        """
        Attempt to detect the target in *frame*.

        Parameters
        ----------
        frame      : full-resolution camera frame (np.ndarray).
        predict_fn : callable(frame) → ultralytics Results list.

        Returns
        -------
        Highest-confidence Detection in full-frame coords, or None.
        """
        # Pass 1 — cheap full-frame attempt
        best = extract_best(predict_fn(frame))
        if best is not None:
            return best

        # Pass 2 — tiled inference
        candidates: List[Detection] = []
        for tx, ty, tside in self.tiles:
            tile_img = frame[ty:ty + tside, tx:tx + tside]
            det = extract_best(predict_fn(tile_img),
                               x1=tx, y1=ty, side=tside)
            if det is not None:
                candidates.append(det)

        if not candidates:
            return None
        return max(candidates, key=lambda d: d.conf)


# =====================================================================
# ROITracker
# =====================================================================

class ROITracker:
    """
    Adaptive square-ROI tracker for fast, high-resolution inference
    once a target has been acquired.

    Maintains a simple state machine:
      - Tracks consecutive misses and signals when to fall back
        to SEARCH.
      - Periodically triggers a full-frame recheck to guard against
        ROI drift.

    The actual mode switching lives in VisionProducer; this class
    only provides the ROI geometry and miss bookkeeping.
    """

    def __init__(self, img_w: int, img_h: int,
                 margin_mult: float = 3.0,
                 min_half: int = 150,
                 max_misses: int = 10,
                 recheck_interval: int = 30):
        self.img_w = img_w
        self.img_h = img_h
        self.margin_mult = margin_mult
        self.min_half = min_half
        self.max_misses = max_misses
        self.recheck_interval = recheck_interval

        self.last_det: Optional[Detection] = None
        self.miss_count: int = 0
        self.frame_count: int = 0

    # ----- public API -----

    def reset(self, det: Detection) -> None:
        """Initialise / re-initialise tracking from a SEARCH detection."""
        self.last_det = det
        self.miss_count = 0
        self.frame_count = 0

    @property
    def should_recheck(self) -> bool:
        """True when a periodic full-frame recheck is due."""
        return self.frame_count % self.recheck_interval == 0

    @property
    def lost(self) -> bool:
        """True when consecutive misses exceed threshold."""
        return self.miss_count >= self.max_misses

    def record_hit(self, det: Detection) -> None:
        self.last_det = det
        self.miss_count = 0

    def record_miss(self) -> None:
        self.miss_count += 1

    def tick(self) -> None:
        """Call once per frame to advance the recheck counter."""
        self.frame_count += 1

    # ----- ROI geometry -----

    def compute_roi(self) -> Tuple[int, int, int]:
        """
        Compute a square crop (x1, y1, side) centred on the last
        detection.  Square ensures uniform scaling → theta preserved.
        """
        det = self.last_det
        half = max(det.w, det.h) * self.margin_mult
        half = max(half, self.min_half)
        half = int(math.ceil(half))

        cu = int(round(det.u))
        cv = int(round(det.v))

        x1 = max(0, cu - half)
        y1 = max(0, cv - half)
        x2 = min(self.img_w, cu + half)
        y2 = min(self.img_h, cv + half)

        side = min(x2 - x1, y2 - y1)

        x1 = max(0, cu - side // 2)
        y1 = max(0, cv - side // 2)
        if x1 + side > self.img_w:
            x1 = self.img_w - side
        if y1 + side > self.img_h:
            y1 = self.img_h - side

        return x1, y1, side

    # ----- single-frame inference -----

    def track(self, frame: np.ndarray, predict_fn
              ) -> Tuple[Optional[Detection], Optional[Tuple[int, int, int]]]:
        """
        Run one tracking cycle.

        Returns (det, roi) where *roi* is ``(x1, y1, side)`` when a
        crop was used, or None for full-frame recheck frames.
        """
        self.tick()
        roi: Optional[Tuple[int, int, int]] = None

        if self.should_recheck:
            results = predict_fn(frame)
            det = extract_best(results)
        else:
            x1, y1, side = self.compute_roi()
            roi = (x1, y1, side)
            crop = frame[y1:y1 + side, x1:x1 + side]
            results = predict_fn(crop)
            det = extract_best(results, x1=x1, y1=y1, side=side)

        if det is not None:
            self.record_hit(det)
        else:
            self.record_miss()

        return det, roi