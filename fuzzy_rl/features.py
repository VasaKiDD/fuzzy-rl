from dataclasses import dataclass

import numpy as np


class RBFFeaturesPendulum:
    """Normalized RBFs. Centers on a grid over angle x angular velocity.

    Observation from Pendulum is [cos(theta), sin(theta), omega], omega in [-8, 8].
    We work in the normalized coordinate z = [cos(theta), sin(theta), omega/8],
    so every dimension lives in [-1, 1].
    """

    def __init__(
        self, n_angle: int = 12, n_vel: int = 8, bandwidth: float = 0.40
    ):
        angles = np.linspace(-np.pi, np.pi, n_angle, endpoint=False)
        vels = np.linspace(-8.0, 8.0, n_vel)
        centers = []
        for a in angles:
            for v in vels:
                centers.append([np.cos(a), np.sin(a), v / 8.0])
        self.centers = np.asarray(centers)  # (n, 3)
        self.bw2 = 2.0 * bandwidth**2
        self.n = self.centers.shape[0]  # feature dimension

    @staticmethod
    def _normalize(s: np.ndarray) -> np.ndarray:
        return np.array([s[0], s[1], s[2] / 8.0])

    def __call__(self, s: np.ndarray) -> np.ndarray:
        z = self._normalize(s)
        d2 = np.sum((self.centers - z) ** 2, axis=1)
        phi = np.exp(-d2 / self.bw2)
        phi /= phi.sum() + 1e-12  # sum-to-1 normalization
        return phi
