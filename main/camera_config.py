"""
RealSense D435i camera intrinsics lookup.

All values measured from the actual device.  Keyed by (width, height).
To add a new resolution, just add another entry to INTRINSICS.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float   # ppx
    cy: float   # ppy


# ---- Measured intrinsics per resolution ----
INTRINSICS: Dict[Tuple[int, int], CameraIntrinsics] = {
    (640, 480): CameraIntrinsics(
        fx=618.07,  fy=618.20,
        cx=318.66,  cy=240.94,
    ),
    (1280, 720): CameraIntrinsics(
        fx=927.11,  fy=927.30,
        cx=637.99,  cy=361.41,
    ),
    (1920, 1080): CameraIntrinsics(
        fx=1390.66, fy=1390.95,
        cx=956.99,  cy=542.11,
    ),
    (424, 240): CameraIntrinsics(
        fx=309.04,  fy=309.10,
        cx=211.33,  cy=120.47,
    ),
}

# ---- Reference resolution used when w_d / h_d were originally tuned ----
REF_RESOLUTION = (640, 480)


def get_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """Return intrinsics for the given resolution, or raise ValueError."""
    key = (width, height)
    if key not in INTRINSICS:
        supported = ", ".join(f"{w}x{h}" for w, h in sorted(INTRINSICS))
        raise ValueError(
            f"No intrinsics for {width}x{height}. "
            f"Supported: {supported}"
        )
    return INTRINSICS[key]


def scale_factor(width: int, height: int) -> float:
    """
    Focal-length ratio  fx_new / fx_ref.

    Pixel-space quantities that represent a fixed physical size
    (w_d, h_d, dead_px, KF measurement noise, etc.) should be
    multiplied by this factor when switching resolution.
    """
    ref = INTRINSICS[REF_RESOLUTION]
    cur = get_intrinsics(width, height)
    return cur.fx / ref.fx


def max_fps(width: int, height: int) -> float:
    """Best-case FPS for the given resolution (D435i RGB sensor)."""
    _FPS: Dict[Tuple[int, int], float] = {
        (424,  240):  60.0,
        (640,  480):  30.0,
        (1280, 720):  15.0,
        (1920, 1080): 8.0,
    }
    return _FPS.get((width, height), 30.0)