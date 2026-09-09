"""Gymnasium wrappers for studies that change what the policy sees, without touching the env."""

from collections.abc import Mapping, Sequence

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .envs.control_env import LettuceGreenhouseEnv


class ParameterObservation(gymnasium.ObservationWrapper):
    """Append named model coefficients to the observation, so a policy can condition on them.

    The env keeps the coefficients out of the observation on purpose, since whether a controller may
    know them is a research question, and reports them in ``info["params"]``. This wrapper is the
    "yes" answer: the current values of ``names`` are appended, in that order, after every reset and
    step, so a per-step provider is followed too.

    ``ranges`` maps a name to ``(lo, hi)`` and rescales that value to ``(value - lo) / (hi - lo)``,
    typically the training range, so a value outside it lands outside ``[0, 1]`` and is visible as
    such. Names without a range are appended raw. Which coefficients, and which range, are the
    caller's choices; nothing here is normalised by default.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        names: Sequence[str],
        ranges: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__(env)
        inner = env.unwrapped
        if not isinstance(inner, LettuceGreenhouseEnv):
            raise TypeError("ParameterObservation wraps a LettuceGreenhouseEnv")
        known = inner.model.constants.names
        unknown = [n for n in names if n not in known]
        if unknown:
            raise ValueError(f"unknown coefficient(s) {unknown}; the model has {known}")
        if not names:
            raise ValueError("names must not be empty")
        self.names = tuple(names)
        self.ranges = {k: (float(lo), float(hi)) for k, (lo, hi) in (ranges or {}).items()}
        for name, (lo, hi) in self.ranges.items():
            if name not in self.names:
                raise ValueError(f"range given for {name!r}, which is not appended")
            if not hi > lo:
                raise ValueError(f"{name}: range hi {hi} must exceed lo {lo}")
        base = env.observation_space
        if not isinstance(base, spaces.Box):
            raise TypeError("the wrapped observation space must be a Box")
        low = np.concatenate([base.low, np.full(len(self.names), -np.inf, dtype=np.float32)])
        high = np.concatenate([base.high, np.full(len(self.names), np.inf, dtype=np.float32)])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self._inner = inner

    def observation(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        params = self._inner.parameters
        values = []
        for name in self.names:
            value = params.get(name)
            if name in self.ranges:
                lo, hi = self.ranges[name]
                value = (value - lo) / (hi - lo)
            values.append(value)
        return np.concatenate([observation, np.asarray(values, dtype=np.float32)])
