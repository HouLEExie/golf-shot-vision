from __future__ import annotations

from typing import Tuple

import numpy as np


class KalmanTracker:
    """Small constant-velocity Kalman filter for 2D video points."""

    def __init__(self, process_noise: float = 0.08, measurement_noise: float = 5.0) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = np.zeros((4, 1), dtype=float)
        self.covariance = np.eye(4, dtype=float) * 500.0
        self.initialized = False

    def initialize(self, x: float, y: float) -> None:
        self.state = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        self.covariance = np.eye(4, dtype=float) * 10.0
        self.initialized = True

    def predict(self, dt: float = 1.0) -> Tuple[float, float]:
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        process = np.eye(4, dtype=float) * self.process_noise
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        measurement = np.array([[x], [y]], dtype=float)
        measurement_covariance = np.eye(2, dtype=float) * self.measurement_noise

        innovation = measurement - measurement_matrix @ self.state
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T + measurement_covariance
        )
        kalman_gain = self.covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(4, dtype=float)
        self.covariance = (identity - kalman_gain @ measurement_matrix) @ self.covariance
        return float(self.state[0, 0]), float(self.state[1, 0])
