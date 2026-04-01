"""
Kalman filter for smoothing OBB detection measurements.

State vector (12):
    [u, v, w, h, sin(θ), cos(θ), u̇, v̇, ẇ, ḣ, sin̊(θ), cos̊(θ)]

Measurement vector (6):
    [u, v, w, h, sin(θ), cos(θ)]

Motion model: constant-velocity (position += velocity * dt)
Observation model: direct read-off of positions (H = [I₆ | 0₆])

Process noise model (CHANGED):
    Piecewise-constant white-noise acceleration.  For each
    (position, velocity) pair the continuous-time spectral
    density is σ_a², giving the per-step covariance block:

        Q_block = σ_a² · [[dt³/3,  dt²/2],
                           [dt²/2,  dt   ]]

    This ensures that N small predict steps accumulate the
    same total uncertainty as a single predict of duration N·dt,
    regardless of controller-to-camera rate ratio.

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
    sigma_accel_pos : float
        Continuous-time acceleration noise for u, v (px / s²).
        Controls how quickly the filter adapts to position changes.
        Higher → more responsive, less smooth.
    sigma_accel_size : float
        Continuous-time acceleration noise for w, h (px / s²).
    sigma_accel_angle : float
        Continuous-time acceleration noise for sin(θ), cos(θ) (1 / s²).
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
        # ---- Process noise: continuous-time acceleration density ----
        sigma_accel_pos: float = 150.0,     # px / s²
        sigma_accel_size: float = 80.0,     # px / s²
        sigma_accel_angle: float = 3.0,     # 1 / s²
        # ---- Measurement noise ----
        sigma_meas_pos: float = 5.0,
        sigma_meas_size: float = 5.0,
        sigma_meas_angle: float = 0.1,
    ):
        self.dt = dt

        # Store σ_a² for each of the 6 state channels:
        #   [u, v, w, h, sin(θ), cos(θ)]
        sa_pos2   = sigma_accel_pos ** 2
        sa_size2  = sigma_accel_size ** 2
        sa_angle2 = sigma_accel_angle ** 2
        self._sigma_a_sq = np.array([
            sa_pos2, sa_pos2,           # u, v
            sa_size2, sa_size2,         # w, h
            sa_angle2, sa_angle2,       # sin(θ), cos(θ)
        ])

        # ---- State vector x = [pos(6), vel(6)] ----
        self.x = np.zeros(self.N_STATE)
        self.x[5] = 1.0   # cos(θ) = 1

        # ---- Covariance ----
        self.P = np.eye(self.N_STATE) * 100.0

        # ---- Transition matrix F (constant-velocity) ----
        self.F = np.eye(self.N_STATE)
        self.F[:6, 6:] = np.eye(6) * dt

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

    # ------------------------------------------------------------------
    # Process noise Q — scaled correctly for arbitrary dt
    # ------------------------------------------------------------------
    def _build_Q(self, dt: float) -> np.ndarray:
        """
        Piecewise-constant white-noise acceleration model.

        For each (position_i, velocity_i) pair, the 2x2 block is:

            σ_a² · [[dt³/3,  dt²/2],
                     [dt²/2,  dt   ]]

        This makes the total accumulated uncertainty depend on
        *elapsed time*, not on *number of predict calls*.
        """
        Q = np.zeros((self.N_STATE, self.N_STATE))
        dt2 = dt * dt
        dt3 = dt2 * dt

        for i, sa2 in enumerate(self._sigma_a_sq):
            pi = i          # position index  (0..5)
            vi = i + 6      # velocity index  (6..11)
            Q[pi, pi] = sa2 * dt3 / 3.0
            Q[pi, vi] = sa2 * dt2 / 2.0
            Q[vi, pi] = sa2 * dt2 / 2.0
            Q[vi, vi] = sa2 * dt

        return Q

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
        if dt is None:
            dt = self.dt

        # Build F for this specific dt
        if dt != self.dt:
            F = np.eye(self.N_STATE)
            F[:6, 6:] = np.eye(6) * dt
        else:
            F = self.F

        # Build Q scaled to this dt
        Q = self._build_Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

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