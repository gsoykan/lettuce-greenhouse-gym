"""A scenario MPC: one control sequence, several coefficient futures, the expected reward."""

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
from ..model.parameters import ParameterProvider
from ..rewards import PenaltyBound, economic_stage_terms
from ..variables.controls import CONTROL
from ..variables.exogenous import EXOGENOUS
from ..variables.observables import OBSERVABLE
from ..variables.states import STATE
from .base import Controller
from .mpc import SolveStats, band_limits


@dataclass(frozen=True)
class ScenarioMPCSettings:
    """Horizon, how many futures to hedge against, and the solver budget."""

    horizon: int = 12
    n_scenarios: int = 10
    seed: int = 0
    """Seed of the controller's own RNG for scenario draws. Kept apart from the plant's RNG on
    purpose: what the controller imagines must not be correlated with what happens."""
    bounds: tuple[PenaltyBound, ...] | None = None
    """Comfort bands to plan with; ``None`` uses the env's reward bounds."""
    max_iter: int = 500
    ipopt_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.n_scenarios < 1 or self.max_iter < 1:
            raise ValueError("horizon, n_scenarios and max_iter must be positive")


class ScenarioMPC(Controller):
    """Receding-horizon optimisation of the *expected* economic reward over sampled coefficients.

    Where :class:`NominalMPC` commits to one coefficient set, this controller draws
    ``n_scenarios`` coefficient sequences from a :class:`ParameterProvider` before every solve, runs
    one copy of the dynamics per scenario, and optimises a single control sequence against their
    average reward, with each scenario's own comfort slacks. The provider is the controller's
    *belief* about the plant: the same provider the env uses gives a well-specified controller, a
    different one a mis-specified one, a fixed provider with one scenario reproduces
    :class:`NominalMPC` exactly. Privileged: reads the true state and the weather forecast.

    Per-step schemes are honoured: the provider's ``step`` hook is called along the horizon with
    the horizon index, so a coefficient that is redrawn every plant step is redrawn every planned
    step too. A reference implementation: no scenario tree, no recourse, no terminal term.
    """

    privileged = True

    def __init__(
        self, provider: ParameterProvider, settings: ScenarioMPCSettings | None = None
    ) -> None:
        self.provider = provider
        self.settings = settings if settings is not None else ScenarioMPCSettings()
        self.n_failures = 0
        self.solves: list[SolveStats] = []
        self._rng = np.random.default_rng(self.settings.seed)

    def reset(self, env: LettuceGreenhouseEnv) -> None:
        super().reset(env)
        s = self.settings
        H, N = s.horizon, s.n_scenarios
        F, g, cfg = env.integrator, env.measurement, env.config
        if not (is_symbolic(F) and is_symbolic(g)):
            raise TypeError(
                f"{type(self).__name__} differentiates through the model, so it needs a "
                "SymbolicDynamicsModel (CasADi integrator and measurement)"
            )
        self._F, self._g = F, g
        self._n_c = env.model.constants.size
        self._rng = np.random.default_rng(s.seed)
        self.solves = []

        bounds = s.bounds if s.bounds is not None else cfg.reward.bounds
        self._bounds = bounds
        self._bound_idx = [OBSERVABLE.idx(b.name) for b in bounds]
        self._time_varying = any(b.time_varying for b in bounds)
        w_lo = np.array([b.w_lo for b in bounds])
        w_hi = np.array([b.w_hi for b in bounds])
        x_lo, x_hi = STATE.bounds()

        opti = casadi.Opti()
        U = opti.variable(CONTROL.size, H)  # one plan, shared by every scenario
        X = [opti.variable(STATE.size, H + 1) for _ in range(N)]
        S_lo = [opti.variable(len(bounds), H) for _ in range(N)]
        S_hi = [opti.variable(len(bounds), H) for _ in range(N)]
        C = [opti.parameter(self._n_c, H) for _ in range(N)]  # a coefficient future per scenario
        x0 = opti.parameter(STATE.size)
        V = opti.parameter(EXOGENOUS.size, H)
        u_prev = opti.parameter(CONTROL.size)
        if self._time_varying:
            B_lo = opti.parameter(len(bounds), H)
            B_hi = opti.parameter(len(bounds), H)
        else:
            B_lo = casadi.DM(np.tile([[b.lo] for b in bounds], (1, H)))
            B_hi = casadi.DM(np.tile([[b.hi] for b in bounds], (1, H)))

        objective = 0
        for i in range(N):
            opti.subject_to(X[i][:, 0] == x0)
            opti.subject_to(casadi.vec(S_lo[i]) >= 0)
            opti.subject_to(casadi.vec(S_hi[i]) >= 0)
            for k in range(H):
                opti.subject_to(X[i][:, k + 1] == F(X[i][:, k], U[:, k], V[:, k], C[i][:, k]))
                opti.subject_to(opti.bounded(x_lo, X[i][:, k + 1], x_hi))
                y = g(X[i][:, k + 1])[self._bound_idx]
                opti.subject_to(S_lo[i][:, k] >= B_lo[:, k] - y)
                opti.subject_to(S_hi[i][:, k] >= y - B_hi[:, k])
                revenue, energy_cost, co2_cost = economic_stage_terms(
                    X[i][:, k], X[i][:, k + 1], U[:, k], cfg.dt, cfg.reward
                )
                penalty = casadi.dot(w_lo, S_lo[i][:, k]) + casadi.dot(w_hi, S_hi[i][:, k])
                objective += (revenue - energy_cost - co2_cost - penalty) / N
        for k in range(H):
            opti.subject_to(opti.bounded(self.lo, U[:, k], self.hi))
            if cfg.action_mode is ActionMode.DELTA:
                prev = u_prev if k == 0 else U[:, k - 1]
                opti.subject_to(opti.bounded(-self.du_max, U[:, k] - prev, self.du_max))
        opti.minimize(-objective)
        ipopt = {"print_level": 0, "sb": "yes", "max_iter": s.max_iter}
        ipopt.update(s.ipopt_options)
        opti.solver("ipopt", {"print_time": False, "expand": True}, ipopt)

        self._opti, self._U, self._X, self._S_lo, self._S_hi, self._C = opti, U, X, S_lo, S_hi, C
        self._x0, self._V, self._u_prev, self._B_lo, self._B_hi = x0, V, u_prev, B_lo, B_hi
        self._u_plan = np.tile(env.control[:, None], (1, H))

    def draw_scenarios(self) -> list[NDArray[np.float64]]:
        """``n_scenarios`` coefficient sequences, each ``(n_coefficients, horizon)``, from the
        provider: ``sample`` once, then ``step`` along the horizon."""
        H = self.settings.horizon
        out = []
        for _ in range(self.settings.n_scenarios):
            current = self.provider.sample(self._rng)
            seq = np.empty((self._n_c, H))
            for k in range(H):
                updated = self.provider.step(self._rng, k, current)
                if updated is not None:
                    current = updated
                seq[:, k] = current.to_array()
            out.append(seq)
        return out

    def _guesses(
        self, x: NDArray, v: NDArray, scenarios: list[NDArray], lo: NDArray, hi: NDArray
    ) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
        """Roll the control plan through each scenario for consistent state and slack guesses."""
        xs, s_los, s_his = [], [], []
        for c in scenarios:
            traj = [x]
            for k in range(self.settings.horizon):
                traj.append(
                    np.asarray(self._F(traj[-1], self._u_plan[:, k], v[:, k], c[:, k])).ravel()
                )
            x_guess = np.column_stack(traj)
            ys = np.column_stack(
                [np.asarray(self._g(x_guess[:, k])).ravel() for k in range(1, x_guess.shape[1])]
            )[self._bound_idx]
            xs.append(x_guess)
            s_los.append(np.maximum(0.0, lo - ys))
            s_his.append(np.maximum(0.0, ys - hi))
        return xs, s_los, s_his

    def control(self, obs: NDArray[np.float32], env: LettuceGreenhouseEnv) -> NDArray[np.float64]:
        H, o = self.settings.horizon, self._opti
        x, v = env.state, env.weather_forecast(H)
        scenarios = self.draw_scenarios()
        lo, hi = band_limits(self._bounds, v)
        o.set_value(self._x0, x)
        o.set_value(self._V, v)
        o.set_value(self._u_prev, env.control)
        if self._time_varying:
            o.set_value(self._B_lo, lo)
            o.set_value(self._B_hi, hi)
        xs, s_los, s_his = self._guesses(x, v, scenarios, lo, hi)
        o.set_initial(self._U, self._u_plan)
        for i, c in enumerate(scenarios):
            o.set_value(self._C[i], c)
            o.set_initial(self._X[i], xs[i])
            o.set_initial(self._S_lo[i], s_los[i])
            o.set_initial(self._S_hi[i], s_his[i])
        started = time.perf_counter()
        try:
            u_plan = o.solve().value(self._U)
            success = True
        except RuntimeError:
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
        self._u_plan = np.concatenate([u_plan[:, 1:], u_plan[:, -1:]], axis=1)
        return np.clip(u_plan[:, 0], self.lo, self.hi)

    def step_info(self) -> dict[str, Any]:
        return asdict(self.solves[-1]) if self.solves else {}
