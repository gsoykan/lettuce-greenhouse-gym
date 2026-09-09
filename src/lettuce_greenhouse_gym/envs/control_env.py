import warnings
from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from ..config import ActionMode, EnvConfig, TimestepEncoding
from ..model.base import DynamicsModel, Integrator, Measurement
from ..model.parameters import FixedParameterProvider, ParameterProvider, ParameterSet
from ..model.vanhenten import VanHentenLettuce
from ..rewards import EconomicReward, Reward, RewardContext
from ..variables.exogenous import EXOGENOUS
from ..weather import (
    BENCHMARK_SCENARIO,
    FixedWeatherSampler,
    WeatherPerturbation,
    WeatherRepository,
    WeatherSampler,
    WeatherScenario,
    WeatherSeries,
    default_repository,
)


@dataclass(frozen=True)
class EnvState:
    """Everything that distinguishes one moment of an episode from another, as plain data.

    :meth:`LettuceGreenhouseEnv.snapshot` returns one and :meth:`LettuceGreenhouseEnv.restore`
    accepts one, so a controller can branch an episode (roll a policy out from here under several
    parameter draws, then come back) or place the plant at a chosen state and time to generate
    data. ``dataclasses.replace`` edits a field. The weather realisation belongs to the episode,
    not the state: a snapshot restores only into the episode it was taken from.
    """

    scenario: WeatherScenario
    x: NDArray[np.float64]
    u_prev: NDArray[np.float64]
    step_index: int
    parameters: ParameterSet


class LettuceGreenhouseEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.floating]]):
    """The van Henten lettuce greenhouse as a Gymnasium environment.

    Every default reproduces the published benchmark: the van Henten model on nominal parameters,
    the economic reward, and 40 days of measured Bleiswijk weather from 9 February 2014 at a
    30-minute control step. Pass any collaborator to change one piece without touching the rest.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        render_mode: str | None = None,
        model: DynamicsModel | None = None,
        reward: Reward | None = None,
        parameter_provider: ParameterProvider | None = None,
        weather_sampler: WeatherSampler | None = None,
        weather_repository: WeatherRepository | None = None,
        weather_perturbation: WeatherPerturbation | None = None,
    ) -> None:
        if render_mode is not None:
            raise ValueError(
                f"render_mode {render_mode!r} is not supported; this environment has no renderer"
            )
        self.render_mode = render_mode

        # `x if x is not None else default`, not `x or default`: truthiness would silently discard
        # a collaborator that defines __bool__ or __len__.
        self.config = config if config is not None else EnvConfig()
        self.model = model if model is not None else VanHentenLettuce()
        self.reward = reward if reward is not None else EconomicReward(self.config.reward)
        self.parameter_provider = (
            parameter_provider
            if parameter_provider is not None
            else FixedParameterProvider(self.model.constants)
        )
        self.weather_sampler = (
            weather_sampler
            if weather_sampler is not None
            else FixedWeatherSampler(BENCHMARK_SCENARIO)
        )
        self.weather_repository = (
            weather_repository if weather_repository is not None else default_repository()
        )
        self.weather_perturbation = weather_perturbation

        if not self.config.include_timestep:
            warnings.warn(
                "include_timestep=False: the episode ends with terminated=True, which is only "
                "sound if the agent can see how far through the season it is. Without it the task "
                "is not Markov and values near harvest will be biased.",
                UserWarning,
                stacklevel=2,
            )

        # compiled once: F and g depend only on dt and the model, and c is a runtime argument, so a
        # randomised ParameterSet never triggers a rebuild
        self._F = self.model.build_integrator(self.config.dt)
        self._g = self.model.build_measurement()
        self.model.check(self._F, self._g)

        # resolved once: overrides are config-time, not step-time
        self._u_lo, self._u_hi = self.config.effective_control_bounds()
        self._du_max = self.config.effective_du_max()
        self._x0 = self.config.effective_x0()
        self._u0 = self.config.effective_u0()

        self.action_space = spaces.Box(-1.0, 1.0, (self.model.controls.size,), dtype=np.float32)
        self._layout = self._build_observation_layout()
        self.observation_space = spaces.Box(
            np.concatenate([lo for _, lo, _ in self._layout]).astype(np.float32),
            np.concatenate([hi for _, _, hi in self._layout]).astype(np.float32),
            dtype=np.float32,
        )

        # episode state, filled by reset(); step() refuses to run before that
        self._started = False
        self._parameters: ParameterSet = self.model.constants
        self._c = self._parameters.to_array()
        self._scenario: WeatherScenario | None = None
        self._series: WeatherSeries | None = None
        self._weather = np.zeros((self.model.exogenous.size, 0))
        self._x = self._x0.copy()
        self._u_prev = self._u0.copy()
        self._step_index = 0

    def _build_observation_layout(self) -> list[tuple[str, NDArray, NDArray]]:
        """The observation's blocks in order, as ``(name, low, high)``::

            [ observables (4) | previous_control (3) | timestep (1) | weather (4 * window) ]

        Both ``observation_space`` and :meth:`observe` read this one description, so the two cannot
        drift apart. Adding a block means editing this function only.
        """
        parts = [("observables", *self.model.observables.bounds())]
        if self.config.include_previous_control:
            parts.append(("previous_control", self._u_lo, self._u_hi))
        if self.config.include_timestep:
            index = self.config.timestep_encoding is TimestepEncoding.INDEX
            hi = float(self.config.n_steps) if index else 1.0
            parts.append(("timestep", np.array([0.0]), np.array([hi])))
        if self.config.weather_window:
            v_lo, v_hi = self.model.exogenous.bounds()
            n = self.config.weather_window
            parts.append(("weather", np.tile(v_lo, n), np.tile(v_hi, n)))
        return parts

    def decode_action(self, action: NDArray[np.floating]) -> NDArray[np.float64]:
        """Map an action in ``[-1, 1]^n`` to a physical control, per the configured mode.

        Absolute: ``lo + (a + 1) / 2 * (hi - lo)``, the action is the level; with
        ``absolute_rate_limit`` the level is also clipped to ``u_prev +/- du_max``. Delta:
        ``clip(u_prev + a * du_max, lo, hi)``, the action is a change, zero holds. Actions are
        clipped to ``[-1, 1]`` first, since Gaussian policies emit values just outside. The inverse
        is :meth:`encode_control`; both are meaningful only after ``reset()``.
        """
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.config.action_mode is ActionMode.ABSOLUTE:
            level = self._u_lo + 0.5 * (a + 1.0) * (self._u_hi - self._u_lo)
            if self.config.absolute_rate_limit:
                level = np.clip(level, self._u_prev - self._du_max, self._u_prev + self._du_max)
            return level
        return np.clip(self._u_prev + a * self._du_max, self._u_lo, self._u_hi)

    def observe(
        self, x: NDArray[np.floating], u_prev: NDArray[np.floating], step_index: int
    ) -> NDArray[np.float32]:
        """The observation the env would emit at state ``x`` after control ``u_prev`` at step
        ``step_index`` of the current episode.

        The same function :meth:`step` uses, exposed so a planner can evaluate a policy at states it
        is only considering, along a candidate trajectory or at a sampled state, and see exactly
        what the policy would see there. Weather comes from the current episode, so the env must
        have been reset; the state itself is not changed.
        """
        if not self._started:
            raise RuntimeError("call reset() before observe()")
        if not 0 <= step_index <= self.config.n_steps:
            raise ValueError(f"step_index must be in [0, {self.config.n_steps}], got {step_index}")
        x_arr = np.asarray(x, dtype=np.float64)
        u_arr = np.asarray(u_prev, dtype=np.float64)
        if x_arr.shape != self._x0.shape or u_arr.shape != self._u0.shape:
            raise ValueError(
                f"expected x of shape {self._x0.shape} and u_prev of shape {self._u0.shape}, "
                f"got {x_arr.shape} and {u_arr.shape}"
            )
        pieces: list[NDArray[np.float64]] = [np.asarray(self._g(x_arr)).ravel()]
        if self.config.include_previous_control:
            pieces.append(u_arr)
        if self.config.include_timestep:
            index = self.config.timestep_encoding is TimestepEncoding.INDEX
            pieces.append(
                np.array([float(step_index) if index else step_index / self.config.n_steps])
            )
        if self.config.weather_window:
            pieces.append(self._forecast(step_index, self.config.weather_window).T.ravel())
        # float32 to match the declared space; astype also copies, so a caller cannot reach env state
        obs = np.concatenate(pieces).astype(np.float32)
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                f"observation {obs.shape} does not match the declared space "
                f"{self.observation_space.shape}; layout and assembly disagree"
            )
        return obs

    def _observation(self) -> NDArray[np.float32]:
        return self.observe(self._x, self._u_prev, self._step_index)

    def _forecast(self, step_index: int, n_steps: int) -> NDArray[np.float64]:
        """Realised weather for ``n_steps`` from ``step_index``, through the forecast hook."""
        end = step_index + n_steps
        if end > self._weather.shape[1]:
            raise ValueError(
                f"{self._series.name if self._series else 'weather'}: a forecast of {n_steps} "
                f"steps from step {step_index} runs past the end of the trace "
                f"({self._weather.shape[1]} steps available from the season start)"
            )
        true = self._weather[:, step_index:end]
        if self.weather_perturbation is None:
            return true
        told = np.asarray(
            self.weather_perturbation.forecast(
                self.np_random, step_index, true.copy(), self.config.dt
            ),
            dtype=np.float64,
        )
        self._check_weather(told, true.shape, "forecast")
        return told

    @staticmethod
    def _check_weather(weather: NDArray[np.float64], shape: tuple[int, ...], hook: str) -> None:
        if weather.shape != shape:
            raise ValueError(
                f"WeatherPerturbation.{hook} returned shape {weather.shape}, expected {shape}"
            )
        lo, hi = EXOGENOUS.bounds()
        if not (np.all(weather >= lo[:, None]) and np.all(weather <= hi[:, None])):
            raise ValueError(
                f"WeatherPerturbation.{hook} returned values outside the physical bounds of "
                f"{EXOGENOUS.names}"
            )

    # ---- Gymnasium API -------------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Start a new episode: sample the season and the greenhouse, plant a fresh crop.

        ``options`` may pin the weather with ``{"start_day": 40.0}`` or
        ``{"weather": WeatherScenario(...)}``, overriding whatever the sampler would have chosen.
        """
        # seeds self.np_random; the providers take that RNG rather than owning one, so a single
        # seed determines the whole episode
        super().reset(seed=seed)

        self._scenario = self.weather_sampler.sample(self.np_random, options)
        self._series = self.weather_repository.load(self._scenario.source)
        self._weather = self._episode_weather()

        self._parameters = self.parameter_provider.sample(self.np_random)
        self._c = self._parameters.to_array()

        # the crop always starts as a transplant; start_day moves the calendar, not the crop
        self._x = self._x0.copy()
        self._u_prev = self._u0.copy()
        self._step_index = 0
        self._started = True
        return self._observation(), self._info()

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one control step: integrate the dynamics, measure, and score the transition."""
        if not self._started:
            raise RuntimeError("call reset() before step()")
        if self._step_index >= self.config.n_steps:
            raise RuntimeError("the episode has ended; call reset() to start another")

        t = self._step_index * self.config.dt
        u = self.decode_action(action)
        v = self._weather[:, self._step_index]
        updated = self.parameter_provider.step(self.np_random, self._step_index, self._parameters)
        if updated is not None:  # the provider's per-step hook: the new set drives this transition
            self._parameters = updated
            self._c = self._parameters.to_array()

        x_prev = self._x
        x_next = np.asarray(self._F(x_prev, u, v, self._c)).ravel()
        if not np.all(np.isfinite(x_next)):
            # raise rather than silently truncating: F clamps to the state bounds, so a non-finite
            # state means something is genuinely wrong: a pathological parameter set, or the 0/0
            # branch of the photosynthesis term (radiation exactly zero while indoor CO2 sits on
            # the compensation point). Hiding that behind a truncation would waste a training run.
            raise RuntimeError(
                f"the integrator produced a non-finite state at step {self._step_index}: {x_next}"
            )

        self._x = x_next
        self._u_prev = u
        self._step_index += 1

        y = np.asarray(self._g(x_next)).ravel()
        reward, breakdown = self.reward(
            RewardContext(
                x_prev=x_prev, x_next=x_next, u=u, v=v, y=y, c=self._c, t=t, dt=self.config.dt
            )
        )

        # The season ends in harvest, so the horizon belongs to the task: this is a terminal state
        # of the MDP, not an externally imposed time limit. Reporting it as truncation would make
        # value estimates bootstrap past a point where no future reward exists. `truncated` stays
        # reserved for genuinely abnormal endings.
        terminated = self._step_index >= self.config.n_steps
        return self._observation(), float(reward), terminated, False, self._info(breakdown)

    def _info(self, breakdown: dict[str, float] | None = None) -> dict[str, Any]:
        """Diagnostics the policy never sees; wrappers, loggers and evaluation code do.

        ``params`` is the exo seam: the episode's physical coefficients, raw and **unnormalised**,
        with their names. Normalising against a training range is a research choice, so it belongs
        to a wrapper rather than to the environment.
        """
        info: dict = {
            "params": self._c.copy(),
            "param_names": self._parameters.names,
            "control": self._u_prev.copy(),
            "weather_source": self._scenario.source if self._scenario else None,
            "start_day": self._scenario.start_day if self._scenario else None,
        }
        if breakdown is not None:
            info.update(breakdown)
        return info

    # ---- public state, for control and model-based research ------------------------------------
    @property
    def state(self) -> NDArray[np.float64]:
        """The current physical state ``x`` (a copy). MPC plans from this, not the observation."""
        return self._x.copy()

    @property
    def control(self) -> NDArray[np.float64]:
        """The last applied physical control ``u``, in actuator units (a copy)."""
        return self._u_prev.copy()

    @property
    def parameters(self) -> ParameterSet:
        """The coefficient set in effect for this episode (immutable, so returned directly)."""
        return self._parameters

    @property
    def integrator(self) -> Integrator:
        """The compiled step ``F(x, u, v, c) -> x_next``, the exact dynamics the env runs.

        A planner using this plans with the true model. Pass ``parameters.to_array()`` as ``c`` to
        plan with this episode's coefficients. With a ``SymbolicDynamicsModel`` (the default) this
        is a ``casadi.Function`` and accepts symbolic arguments; a numeric model gives a plain
        callable.
        """
        return self._F

    @property
    def measurement(self) -> Measurement:
        """The compiled measurement ``g(x) -> y``; symbolic under the same condition as above."""
        return self._g

    def weather_forecast(self, n_steps: int) -> NDArray[np.float64]:
        """Weather for the next ``n_steps`` control steps from now, shape ``(4, n_steps)``.

        The episode's weather extends to the end of the trace, so a planner's horizon may reach past
        the end of the season; it raises only if it would run off the trace itself. Passes through
        the forecast hook of a ``WeatherPerturbation``, if one is set.
        """
        if not self._started:
            raise RuntimeError("call reset() before weather_forecast()")
        return self._forecast(self._step_index, n_steps)

    def _episode_weather(self) -> NDArray[np.float64]:
        """The plant's weather for this episode: the trace from the season start to its end, at the
        control step, through the realise hook of a ``WeatherPerturbation``."""
        series, scenario, cfg = self._series, self._scenario, self.config
        if series is None or scenario is None:
            raise RuntimeError("call reset() before the episode weather exists")
        # validates that the season itself, plus one observation window, fits the trace
        series.episode_slice(scenario.start_day, cfg.n_steps, cfg.dt, lookahead=cfg.weather_window)
        n_total = series.steps_available(scenario.start_day, cfg.dt)
        true = series.episode_slice(scenario.start_day, n_total, cfg.dt)
        if self.weather_perturbation is None:
            return true
        realised = np.asarray(
            self.weather_perturbation.realise(self.np_random, true.copy(), cfg.dt),
            dtype=np.float64,
        )
        self._check_weather(realised, true.shape, "realise")
        return realised

    # ---- branching an episode ------------------------------------------------------------------
    def snapshot(self) -> EnvState:
        """The current moment of the episode as an :class:`EnvState`; see :meth:`restore`."""
        if not self._started or self._scenario is None:
            raise RuntimeError("call reset() before snapshot()")
        return EnvState(
            scenario=self._scenario,
            x=self._x.copy(),
            u_prev=self._u_prev.copy(),
            step_index=self._step_index,
            parameters=self._parameters,
        )

    def restore(self, state: EnvState) -> NDArray[np.float32]:
        """Continue the current episode from ``state``; returns the observation there.

        The weather realisation and the RNG are the episode's and are left alone, so restoring a
        snapshot, stepping on, and restoring again replays the same weather from the same moment.
        Only a state from this episode is accepted: a different scenario needs a ``reset``.
        """
        if not self._started or self._scenario is None:
            raise RuntimeError("call reset() before restore()")
        if state.scenario != self._scenario:
            raise ValueError(
                f"the snapshot belongs to {state.scenario}, but the current episode runs on "
                f"{self._scenario}; reset with options={{'weather': ...}} first"
            )
        x = np.asarray(state.x, dtype=np.float64)
        u_prev = np.asarray(state.u_prev, dtype=np.float64)
        if x.shape != self._x0.shape or u_prev.shape != self._u0.shape:
            raise ValueError("state has the wrong shape for this model")
        lo, hi = self.model.states.bounds()
        if not (np.all(x >= lo) and np.all(x <= hi)):
            raise ValueError(f"state {x} lies outside the model's state bounds")
        if not (np.all(u_prev >= self._u_lo) and np.all(u_prev <= self._u_hi)):
            raise ValueError(f"control {u_prev} lies outside the actuator bounds")
        if not 0 <= state.step_index <= self.config.n_steps:
            raise ValueError(f"step_index must be in [0, {self.config.n_steps}]")
        if state.parameters.names != self._parameters.names:
            raise ValueError(
                "the snapshot's parameter set does not match this model's coefficients"
            )
        self._x = x.copy()
        self._u_prev = u_prev.copy()
        self._step_index = int(state.step_index)
        self._parameters = state.parameters
        self._c = state.parameters.to_array()
        return self._observation()

    def encode_control(self, u: NDArray[np.floating]) -> NDArray[np.float64]:
        """Inverse of :meth:`decode_action`: the action whose decoded control is ``u``.

        Absolute mode inverts the rescaling exactly. Delta mode returns the fraction of ``du_max``
        needed, clipped to ``[-1, 1]``: a target further than one rate limit away is reached over
        several steps, because the rate limit is the env's and a controller cannot bypass it.
        """
        if not self._started:
            raise RuntimeError("call reset() before encode_control()")
        u = np.asarray(u, dtype=np.float64)
        if u.shape != self._u_lo.shape:
            raise ValueError(f"expected a control of shape {self._u_lo.shape}, got {u.shape}")
        if self.config.action_mode is ActionMode.ABSOLUTE:
            return 2.0 * (u - self._u_lo) / (self._u_hi - self._u_lo) - 1.0
        return np.clip((u - self._u_prev) / self._du_max, -1.0, 1.0)
