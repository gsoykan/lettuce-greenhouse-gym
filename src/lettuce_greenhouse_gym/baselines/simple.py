"""Trivial controllers: the floor any policy must beat, and a fixed setting."""

import numpy as np
from numpy.typing import NDArray

from ..envs.control_env import LettuceGreenhouseEnv
from ..variables.controls import CONTROL
from .base import Controller


class AllOff(Controller):
    """Every actuator at its lower bound, all season.

    In delta mode the env's rate limit applies, so the heater winds down from ``u0`` over a few
    steps rather than switching off at once.
    """

    def control(self, obs: NDArray[np.float32], env: LettuceGreenhouseEnv) -> NDArray[np.float64]:
        return self.lo


class ConstantControl(Controller):
    """Hold one control all season, e.g. the benchmark's ``u0 = [0, 0, 50]``."""

    def __init__(self, u) -> None:
        self.u = np.asarray(u, dtype=np.float64)
        if self.u.shape != (CONTROL.size,):
            raise ValueError(f"expected {CONTROL.size} controls, got shape {self.u.shape}")

    def control(self, obs: NDArray[np.float32], env: LettuceGreenhouseEnv) -> NDArray[np.float64]:
        return self.u
