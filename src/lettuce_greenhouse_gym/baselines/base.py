"""The reference-controller contract and a closed-loop runner shared by tests, README, and plots."""

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray

from ..envs.control_env import LettuceGreenhouseEnv
from ..variables.controls import CONTROL
from ..variables.exogenous import EXOGENOUS
from ..variables.observables import OBSERVABLE
from ..variables.states import STATE


class Controller(ABC):
    """A non-learning reference controller.

    Returns physical controls in actuator units (CONTROL order); the env encodes them into actions,
    so one controller works in both action modes. ``privileged`` says whether it reads ``env.state``
    and the weather forecast (an MPC) or only the observation (a grower rule); comparisons with RL
    agents must state which.
    """

    privileged: bool = False

    def reset(self, env: LettuceGreenhouseEnv) -> None:
        """Called once after ``env.reset()``. Caches the actuator limits in effect for the episode.

        Subclasses with internal state (an MPC's warm start) override this and call ``super()``.
        """
        self.lo, self.hi = env.config.effective_control_bounds()
        self.du_max = env.config.effective_du_max()

    @abstractmethod
    def control(self, obs: NDArray[np.float32], env: LettuceGreenhouseEnv) -> NDArray[np.float64]:
        """The control to apply next.

        ``obs`` is the env's observation vector; its layout follows the env config, but the first
        four entries are always the observables, so ``OBSERVABLE.unpack(obs[:4])`` reads them by
        name.
        """

    def step_info(self) -> dict[str, Any]:
        """Diagnostics about the last :meth:`control` call, such as an optimiser's iterations and
        wall time. :func:`run_episode` stores a non-empty result under ``info["controller"]``.
        The default has nothing to report."""
        return {}


@dataclass
class EpisodeLog:
    """One closed-loop season as arrays, time on axis 0."""

    x: NDArray[np.float64]  # (n_steps + 1, 4), x0 first
    u: NDArray[np.float64]  # (n_steps, 3), the control the env actually applied
    v: NDArray[np.float64]  # (n_steps, 4), weather during each step
    y: NDArray[np.float64]  # (n_steps + 1, 4)
    reward: NDArray[np.float64]  # (n_steps,)
    info: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return float(self.reward.sum())

    def to_records(self, dt: float | None = None) -> list[dict[str, Any]]:
        """One flat row per control step: the transition's end state and observables, the applied
        control, the weather during the step, the reward, every scalar in ``info`` (the reward
        breakdown), and a controller's ``step_info`` as ``controller_<key>``.

        ``dt`` adds a ``time_s`` column. Row ``k`` describes the step from state ``k`` to ``k+1``.
        """
        rows: list[dict[str, Any]] = []
        for k in range(self.u.shape[0]):
            row: dict[str, Any] = {"step": k}
            if dt is not None:
                row["time_s"] = k * dt
            row.update(
                {f"x_{n}": float(v) for n, v in zip(STATE.names, self.x[k + 1], strict=True)}
            )
            row.update(
                {f"y_{n}": float(v) for n, v in zip(OBSERVABLE.names, self.y[k + 1], strict=True)}
            )
            row.update({f"u_{n}": float(v) for n, v in zip(CONTROL.names, self.u[k], strict=True)})
            row.update(
                {f"v_{n}": float(v) for n, v in zip(EXOGENOUS.names, self.v[k], strict=True)}
            )
            row["reward"] = float(self.reward[k])
            info = self.info[k] if k < len(self.info) else {}
            for key, value in info.items():
                if isinstance(value, bool | int | float | np.number):
                    row[key] = value.item() if isinstance(value, np.generic) else value
            for key, value in (info.get("controller") or {}).items():
                if isinstance(value, bool | int | float | str | np.number):
                    row[f"controller_{key}"] = (
                        value.item() if isinstance(value, np.generic) else value
                    )
            rows.append(row)
        return rows

    def write_csv(self, path: str | Path, dt: float | None = None) -> Path:
        """:meth:`to_records` as a CSV with a header row."""
        rows = self.to_records(dt)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["step"])
            writer.writeheader()
            writer.writerows(rows)
        return path


def run_episode(
    env: LettuceGreenhouseEnv | gymnasium.Env,
    controller: Controller,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> EpisodeLog:
    """Run one full season closed-loop.

    ``env`` is the bare env or a wrapper around one: resets and steps go through the wrapper (so
    the observation a policy sees is the wrapped one), while the plant's state, control and weather
    are read from the bare env, which is also what the controller receives. ``options`` goes to
    ``env.reset`` verbatim, with the type Gymnasium gives it.
    """
    inner = env.unwrapped
    if not isinstance(inner, LettuceGreenhouseEnv):
        raise TypeError(f"run_episode needs a LettuceGreenhouseEnv, got {type(inner).__name__}")
    obs, _ = env.reset(seed=seed, options=options)
    controller.reset(inner)

    def measure() -> NDArray[np.float64]:  # float64 from g(x); the float32 obs would lose digits
        return np.asarray(inner.measurement(inner.state)).ravel()

    xs, ys = [inner.state], [measure()]
    us, vs, rs, infos = [], [], [], []
    done = False
    while not done:
        u = controller.control(obs, inner)
        vs.append(inner.weather_forecast(1)[:, 0])
        obs, r, terminated, truncated, info = env.step(inner.encode_control(u))
        done = terminated or truncated
        extra = controller.step_info()
        if extra:
            info = {**info, "controller": extra}
        us.append(inner.control)  # after clipping and rate limiting, not the request
        xs.append(inner.state)
        ys.append(measure())
        rs.append(r)
        infos.append(info)
    return EpisodeLog(np.array(xs), np.array(us), np.array(vs), np.array(ys), np.array(rs), infos)
