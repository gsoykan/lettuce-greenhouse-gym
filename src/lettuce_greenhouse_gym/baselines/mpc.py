"""A receding-horizon MPC that plans with the env's own integrator and one fixed model."""

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import casadi
import numpy as np
from numpy.typing import NDArray

from ..config import ActionMode
from ..envs.control_env import LettuceGreenhouseEnv
from ..model.base import is_symbolic
from ..model.parameters import ParameterSet
from ..rewards import PenaltyBound, economic_stage_terms
from ..variables.controls import CONTROL
from ..variables.exogenous import EXOGENOUS
from ..variables.observables import OBSERVABLE
from ..variables.states import STATE
from .base import Controller


@dataclass(frozen=True)
class MPCSettings:
    """Horizon, model choice and solver budget."""

    horizon: int = 12  # control steps looked ahead: 6 h at dt = 1800
    use_true_parameters: bool = False  # plan with the episode's drawn coefficients, not nominal
    parameters: ParameterSet | None = None
    """Coefficients to plan with, when they should be neither the model's nominal ones nor the
    episode's true ones: a deliberately mis-specified or separately identified model. Takes
    precedence over ``use_true_parameters``."""
    bounds: tuple[PenaltyBound, ...] | None = None
    """Comfort bands to plan with; ``None`` uses the env's reward bounds. A planner may tighten a
    band relative to the reward it is scored on, so the two are allowed to differ."""
    max_iter: int = 200  # IPOPT iterations per solve
    ipopt_options: Mapping[str, Any] = field(default_factory=dict)
    """Extra IPOPT options, e.g. ``{"linear_solver": "ma57"}`` or ``{"tol": 1e-6}``. Applied on top
    of the defaults (quiet output, ``max_iter``); a key given here wins."""

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {self.max_iter}")


def band_limits(bounds: tuple[PenaltyBound, ...], v: NDArray) -> tuple[NDArray, NDArray]:
    """``(lo, hi)`` arrays, one row per bound and one column per forecast step, from the forecast's
    radiation: constant rows for fixed bands, day/night rows for time-varying ones."""
    rad = v[EXOGENOUS.idx("rad")]
    lo = np.array([[b.limits(r)[0] for r in rad] for b in bounds], dtype=float)
    hi = np.array([[b.limits(r)[1] for r in rad] for b in bounds], dtype=float)
    return lo, hi


@dataclass(frozen=True)
class SolveStats:
    """What one receding-horizon solve did; :attr:`NominalMPC.solves` keeps one per step."""

    step_index: int
    success: bool
    iterations: int
    wall_time: float  # seconds, the solver call only
    status: str  # IPOPT's return status


class NominalMPC(Controller):
    """Receding-horizon optimisation of the economic reward over the env's own dynamics.

    Reads the true state, the future weather and (optionally) the true coefficients, so it is
    privileged. With ``use_true_parameters=False`` it plans with the model's nominal coefficients:
    under parameter randomisation that is a deliberate plant-model mismatch. Comfort bounds are
    soft, exactly as in the reward, so every solve is feasible. A reference implementation: horizon,
    terminal cost and solver settings are research choices, and there is no terminal term here.
    """

    privileged = True

    def __init__(self, settings: MPCSettings | None = None) -> None:
        self.settings = settings if settings is not None else MPCSettings()
        self.n_failures = 0  # solves that did not converge; the last iterate was used
        self.solves: list[SolveStats] = []

    def reset(self, env: LettuceGreenhouseEnv) -> None:
        super().reset(env)
        H = self.settings.horizon
        F, g, cfg = env.integrator, env.measurement, env.config
        if not (is_symbolic(F) and is_symbolic(g)):
            raise TypeError(
                f"{type(self).__name__} differentiates through the model, so it needs a "
                "SymbolicDynamicsModel (CasADi integrator and measurement); "
                f"{type(env.model).__name__} provides plain callables. A numeric model runs in "
                "the environment and with sampling-based planners, not with this controller."
            )
        if self.settings.parameters is not None:
            source = self.settings.parameters
            if source.names != env.model.constants.names:
                raise ValueError("MPCSettings.parameters must name the model's coefficients")
        elif self.settings.use_true_parameters:
            source = env.parameters
        else:
            source = env.model.constants
        self._c = source.to_array()
        self._F, self._g = F, g
        self.solves = []

        bounds = self.settings.bounds if self.settings.bounds is not None else cfg.reward.bounds
        self._bounds = bounds
        self._bound_idx = [OBSERVABLE.idx(b.name) for b in bounds]
        self._bound_lo = np.array([b.lo for b in bounds])
        self._bound_hi = np.array([b.hi for b in bounds])
        self._time_varying = any(b.time_varying for b in bounds)
        w_lo = np.array([b.w_lo for b in bounds])
        w_hi = np.array([b.w_hi for b in bounds])
        x_lo, x_hi = STATE.bounds()

        # Multiple shooting: the states along the horizon are decision variables, tied to the
        # controls by equality constraints. Better conditioned than unrolling F into one expression.
        # Variables are solved for each step; parameters are filled in with that step's values.
        opti = casadi.Opti()
        X = opti.variable(STATE.size, H + 1)
        U = opti.variable(CONTROL.size, H)
        # The reward's hinge penalties as slack variables: s >= 0, s >= lo - y, s >= y - hi, cost
        # w * s. Same optimum as fmax(0, .) but smooth, which the optimiser needs at the bounds.
        S_lo = opti.variable(len(bounds), H)
        S_hi = opti.variable(len(bounds), H)
        x0 = opti.parameter(STATE.size)
        V = opti.parameter(EXOGENOUS.size, H)
        u_prev = opti.parameter(CONTROL.size)
        # day/night bands change with the forecast, so they enter as parameters set every solve
        if self._time_varying:
            B_lo = opti.parameter(len(bounds), H)
            B_hi = opti.parameter(len(bounds), H)
        else:
            B_lo = casadi.DM(np.tile(self._bound_lo[:, None], (1, H)))
            B_hi = casadi.DM(np.tile(self._bound_hi[:, None], (1, H)))

        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(casadi.vec(S_lo) >= 0)
        opti.subject_to(casadi.vec(S_hi) >= 0)
        objective = 0
        for k in range(H):
            opti.subject_to(X[:, k + 1] == F(X[:, k], U[:, k], V[:, k], self._c))
            # keep the planned states inside the physical box the integrator clamps to
            opti.subject_to(opti.bounded(x_lo, X[:, k + 1], x_hi))
            y = g(X[:, k + 1])[self._bound_idx]
            opti.subject_to(S_lo[:, k] >= B_lo[:, k] - y)
            opti.subject_to(S_hi[:, k] >= y - B_hi[:, k])
            revenue, energy_cost, co2_cost = economic_stage_terms(
                X[:, k], X[:, k + 1], U[:, k], cfg.dt, cfg.reward
            )
            penalty = casadi.dot(w_lo, S_lo[:, k]) + casadi.dot(w_hi, S_hi[:, k])
            objective += revenue - energy_cost - co2_cost - penalty
            opti.subject_to(opti.bounded(self.lo, U[:, k], self.hi))
            if cfg.action_mode is ActionMode.DELTA:  # the env rate-limits only in delta mode
                prev = u_prev if k == 0 else U[:, k - 1]
                opti.subject_to(opti.bounded(-self.du_max, U[:, k] - prev, self.du_max))
        opti.minimize(-objective)
        ipopt = {"print_level": 0, "sb": "yes", "max_iter": self.settings.max_iter}
        ipopt.update(self.settings.ipopt_options)
        opti.solver("ipopt", {"print_time": False, "expand": True}, ipopt)

        self._opti, self._X, self._U, self._S_lo, self._S_hi = opti, X, U, S_lo, S_hi
        self._x0, self._V, self._u_prev = x0, V, u_prev
        self._B_lo, self._B_hi = B_lo, B_hi
        # first guess for the solver: hold the current control over the whole horizon
        self._u_plan = np.tile(env.control[:, None], (1, H))

    def _state_guess(self, x: NDArray, u_plan: NDArray, v: NDArray) -> NDArray:
        """Roll the control guess through F so the state variables start consistent with it."""
        xs = [x]
        for k in range(u_plan.shape[1]):
            xs.append(np.asarray(self._F(xs[-1], u_plan[:, k], v[:, k], self._c)).ravel())
        return np.column_stack(xs)

    def _slack_guess(self, x_guess: NDArray, lo: NDArray, hi: NDArray) -> tuple[NDArray, NDArray]:
        """Slacks consistent with the state guess: the actual band violations along it."""
        ys = np.column_stack(
            [np.asarray(self._g(x_guess[:, k])).ravel() for k in range(1, x_guess.shape[1])]
        )[self._bound_idx]
        return np.maximum(0.0, lo - ys), np.maximum(0.0, ys - hi)

    def control(self, obs: NDArray[np.float32], env: LettuceGreenhouseEnv) -> NDArray[np.float64]:
        H, o = self.settings.horizon, self._opti
        x, v = env.state, env.weather_forecast(H)
        o.set_value(self._x0, x)
        o.set_value(self._V, v)
        o.set_value(self._u_prev, env.control)
        lo, hi = band_limits(self._bounds, v)
        if self._time_varying:
            o.set_value(self._B_lo, lo)
            o.set_value(self._B_hi, hi)
        x_guess = self._state_guess(x, self._u_plan, v)
        s_lo, s_hi = self._slack_guess(x_guess, lo, hi)
        o.set_initial(self._U, self._u_plan)
        o.set_initial(self._X, x_guess)
        o.set_initial(self._S_lo, s_lo)
        o.set_initial(self._S_hi, s_hi)
        started = time.perf_counter()
        try:
            u_plan = o.solve().value(self._U)
            success = True
        except RuntimeError:  # IPOPT did not converge: keep the last iterate rather than stop
            self.n_failures += 1
            u_plan = o.debug.value(self._U)
            success = False
        stats = o.stats()
        self.solves.append(
            SolveStats(
                step_index=len(self.solves),
                success=success,
                iterations=int(stats.get("iter_count", 0)),
                wall_time=time.perf_counter() - started,
                status=str(stats.get("return_status", "")),
            )
        )
        u_plan = np.asarray(u_plan).reshape(CONTROL.size, H)
        # warm start for the next step: shift the plan by one and repeat its last control
        self._u_plan = np.concatenate([u_plan[:, 1:], u_plan[:, -1:]], axis=1)
        return np.clip(u_plan[:, 0], self.lo, self.hi)

    def step_info(self) -> dict[str, Any]:
        """The last solve's statistics; ``run_episode`` records them under ``info["controller"]``."""
        return asdict(self.solves[-1]) if self.solves else {}
