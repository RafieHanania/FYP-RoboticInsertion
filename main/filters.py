"""
Kalman filter for smoothing OBB detection measurements.

State vector (12):
    [u, v, w, h, sin(θ), cos(θ), u̇, v̇, ẇ, ḣ, sin̊(θ), cos̊(θ)]

Measurement vector (6):
    [u, v, w, h, sin(θ), cos(θ)]

Motion model: constant-velocity (position += velocity * dt)
Observation model: direct read-off of positions (H = [I₆ | 0₆])

Usage
-----
    kf = DetectionKalmanFilter(dt=1/30)

    # Each frame:
    if detection_valid:
        kf.predict()
        kf.update(det)
    else:
        kf.predict()          # coast on velocity model

    u, v, w, h, theta = kf.state()
"""

import math
import numpy as np
from typing import Optional, Tuple

from detection_types import Detection


class DetectionKalmanFilter:
    """
    Linear Kalman filter for OBB detection state.

    Parameters
    ----------
    dt : float
        Nominal time step between predictions (1 / camera_fps).
    sigma_pos : float
        Process noise std-dev on position states (pixels / step).
        Higher → trusts measurements more (less smoothing).
    sigma_vel : float
        Process noise std-dev on velocity states (pixels / step²).
        Higher → allows faster velocity changes.
    sigma_meas_pos : float
        Measurement noise std-dev for u, v (pixels).
    sigma_meas_size : float
        Measurement noise std-dev for w, h (pixels).
    sigma_meas_angle : float
        Measurement noise std-dev for sin(θ), cos(θ) (unitless).
    """

    N_STATE = 12
    N_MEAS  = 6

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        sigma_pos: float = 2.0,
        sigma_vel: float = 5.0,
        sigma_meas_pos: float = 5.0,
        sigma_meas_size: float = 5.0,
        sigma_meas_angle: float = 0.1,
    ):
        self.dt = dt

        # ---- State vector x = [pos(6), vel(6)] ----
        self.x = np.zeros(self.N_STATE)
        self.x[5] = 1.0   # cos(θ) = 1

        # ---- Covariance ----
        self.P = np.eye(self.N_STATE) * 100.0

        # ---- Transition matrix F (constant-velocity) ----
        self.F = np.eye(self.N_STATE)
        self.F[:6, 6:] = np.eye(6) * dt

        # ---- Process noise Q ----
        q_pos = sigma_pos ** 2
        q_vel = sigma_vel ** 2
        self.Q = np.diag(
            [q_pos] * 6 + [q_vel] * 6
        )

        # ---- Observation matrix H ----
        self.H = np.zeros((self.N_MEAS, self.N_STATE))
        self.H[:6, :6] = np.eye(6)

        # ---- Measurement noise R ----
        self.R = np.diag([
            sigma_meas_pos ** 2,
            sigma_meas_pos ** 2,
            sigma_meas_size ** 2,
            sigma_meas_size ** 2,
            sigma_meas_angle ** 2,
            sigma_meas_angle ** 2,
        ])

        self._initialised = False

    def _init_from_detection(self, det: Detection) -> None:
        s = math.sin(det.theta)
        c = math.cos(det.theta)
        self.x[:6] = [det.u, det.v, det.w, det.h, s, c]
        self.x[6:] = 0.0
        self.P = np.diag(
            [10.0] * 6 + [50.0] * 6
        )
        self._initialised = True

    def predict(self, dt: Optional[float] = None) -> None:
        if dt is not None and dt != self.dt:
            F = np.eye(self.N_STATE)
            F[:6, 6:] = np.eye(6) * dt
        else:
            F = self.F

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        self._normalise_sincos()

    def update(self, det: Detection) -> None:
        if not self._initialised:
            self._init_from_detection(det)
            return

        s = math.sin(det.theta)
        c = math.cos(det.theta)
        z = np.array([det.u, det.v, det.w, det.h, s, c])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I_KH = np.eye(self.N_STATE) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self._normalise_sincos()

    def state(self) -> Tuple[float, float, float, float, float]:
        u, v, w, h, s, c = self.x[:6]
        theta = math.atan2(s, c)
        return u, v, w, h, theta

    def velocities(self) -> Tuple[float, float, float, float]:
        return tuple(self.x[6:10])

    @property
    def initialised(self) -> bool:
        return self._initialised

    def _normalise_sincos(self) -> None:
        s = self.x[4]
        c = self.x[5]
        norm = math.hypot(s, c)
        if norm > 1e-9:
            self.x[4] = s / norm
            self.x[5] = c / norm
            self.x[10] = self.x[10] / norm
            self.x[11] = self.x[11] / norm