"""Environment configuration: everything the env needs that is not physics.

Plain data only, no live Reward or ParameterProvider instances, so a config can be dumped to
YAML/JSON as an experiment record, diffed between runs, and compared. Collaborators are injected at
the env constructor instead.
"""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy._typing import NDArray

from .rewards import RewardConfig, benchmark_reward_config
from .variables.controls import CONTROL
from .variables.states import STATE


class ActionMode(StrEnum):
    """How an agent's action is interpreted."""

    ABSOLUTE = "absolute"
    """The action *is* the control, rescaled from [-1, 1] to the control's bounds."""

    DELTA = "delta"
    """The action is a rate-limited increment: u_next = clip(u_prev + a * du_max, lo, hi)."""


class TimestepEncoding(StrEnum):
    """How the season clock appears in the observation when ``include_timestep`` is on."""

    PROGRESS = "progress"
    """Fraction of the season elapsed, ``step / n_steps`` in [0, 1]. Independent of ``dt``."""

    INDEX = "index"
    """The raw step counter ``0 .. n_steps - 1``, as the reference RL setup observes it. Equivalent
    to ``progress`` under observation normalisation; offered for bit-for-bit reproduction."""


@dataclass(frozen=True, slots=True)
class ControlOverride:
    """Per-control replacement for a bound or rate limit. ``None`` keeps the declared value.

    A different greenhouse has a different boiler; a different experiment tunes how fast actuators
    may move. Overriding here produces new arrays at config time; the ``ControlVar`` classes are
    never mutated, so two envs in one process cannot interfere.
    """

    name: str
    lo: float | None = None
    hi: float | None = None
    du_max: float | None = None

    def __post_init__(self) -> None:
        if self.name not in CONTROL.names:
            raise ValueError(f"unknown control {self.name!r}; expected one of {CONTROL.names}")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError(f"{self.name}: lo {self.lo} exceeds hi {self.hi}")
        if self.du_max is not None and self.du_max <= 0:
            raise ValueError(f"{self.name}: du_max must be positive, got {self.du_max}")
        if self.lo is None and self.hi is None and self.du_max is None:
            raise ValueError(f"{self.name}: override sets nothing")


@dataclass(frozen=True, slots=True)
class InitialState:
    """Replacement for one component of the initial state ``x0``.

    Used for scenario differences (a larger transplant, a warmer start) without touching the
    ``StateVar`` defaults.
    """

    name: str
    value: float

    def __post_init__(self) -> None:
        if self.name not in STATE.names:
            raise ValueError(f"unknown state {self.name!r}; expected one of {STATE.names}")
        lo, hi = STATE.bounds()
        i = STATE.idx(self.name)
        if not (lo[i] <= self.value <= hi[i]):
            raise ValueError(
                f"{self.name}: initial value {self.value} outside physical bounds "
                f"[{lo[i]}, {hi[i]}]"
            )


@dataclass(frozen=True, slots=True)
class InitialControl:
    """Replacement for one component of the initial control ``u0``.

    An episode has to start from a defined control because in :attr:`ActionMode.DELTA` the first
    action is an *increment* on the previous one. Unlike :class:`InitialState`, the value cannot be
    range-checked here: control bounds are themselves overridable, so whether a value is legal is
    only knowable once :meth:`EnvConfig.effective_control_bounds` has been resolved.
    """

    name: str
    value: float

    def __post_init__(self) -> None:
        if self.name not in CONTROL.names:
            raise ValueError(f"unknown control {self.name!r}; expected one of {CONTROL.names}")


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Everything the env needs that is not physics.

    ``EnvConfig()`` is the benchmark: a 40-day season on 30-minute steps, delta actions, and an
    observation of the measured outputs, the previous control and the season clock. Pass only what
    differs, for example::

        EnvConfig(dt=900.0, action_mode=ActionMode.ABSOLUTE, weather_window=13,
                  control_overrides=(ControlOverride("heating", hi=200.0, du_max=20.0),))

    Overrides never mutate the variable classes: each ``effective_*`` method builds a fresh array
    from the declarations, so two configs in one process cannot interfere.
    """

    dt: float = 1800.0
    """Control step length, in **seconds**. The dynamics are integrated over this interval."""

    episode_days: float = 40.0
    """Season length, in **days**. Steps are derived (:attr:`n_steps`), so the season stays the same
    length when ``dt`` changes."""

    action_mode: ActionMode = ActionMode.DELTA
    """How an action becomes a control; see :class:`ActionMode`."""

    absolute_rate_limit: bool = False
    """In absolute mode, also clip the decoded control to within ``du_max`` of the previous one.

    Delta mode is rate-limited by construction. Absolute mode is not, unless this is set: then an
    action still names a level, but the actuator moves towards it at most one ``du_max`` per step,
    the way some formulations of the benchmark apply a rate constraint to level-valued inputs.
    Meaningless in delta mode, where it raises."""

    include_previous_control: bool = True
    """Append the last applied control (3 values, actuator units) to the observation. Needed in
    delta mode, where the next control depends on the current one."""

    include_timestep: bool = True
    """Append the season clock to the observation, so a policy can tell where it is in the season --
    the optimal action near harvest differs from the same climate on day one."""

    timestep_encoding: TimestepEncoding = TimestepEncoding.PROGRESS
    """Representation of the season clock; see :class:`TimestepEncoding`."""

    weather_window: int = 1
    """How many weather steps to append to the observation, **starting at the current one**.

    ``0`` shows the policy no weather at all; ``1`` shows only the present values (the benchmark's
    standard setting); ``13`` shows the present plus the next twelve steps, which at ``dt=1800`` is
    6.5 hours of forecast. Each step contributes four numbers (radiation, outdoor CO2, outdoor
    temperature, outdoor vapour), so the observation grows by ``4 * weather_window``.

    A forecast is not an unfair advantage: real growers have one, and MPC consumes one over its
    prediction horizon, so offering it to a policy keeps the comparison honest. Wired once weather
    loading exists."""

    terminal_at_harvest: bool = True
    """Report the end of the season as ``terminated`` (default) or as ``truncated``.

    Harvest ends the task, so by default it is a terminal state and a learner does not bootstrap a
    value beyond it. Some formulations treat the season as a time limit on an ongoing process and
    bootstrap through it; ``False`` gives that convention. The two train different policies, so a
    study that compares with either lineage should state which it used."""

    reward: RewardConfig = field(default_factory=benchmark_reward_config)
    """Prices and comfort bands. Plain data, so the whole config stays serialisable."""

    control_overrides: tuple[ControlOverride, ...] = ()
    """Per-control replacements for a bound or rate limit, for a different greenhouse's actuators."""

    initial_state: tuple[InitialState, ...] = ()
    """Per-component replacements for ``x0``, for a different starting scenario."""

    initial_control: tuple[InitialControl, ...] = ()
    """Per-component replacements for ``u0``, the control an episode starts from."""

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.episode_days <= 0:
            raise ValueError(f"episode_days must be positive, got {self.episode_days}")
        if self.weather_window < 0:
            raise ValueError(f"weather_window must be non-negative, got {self.weather_window}")
        if self.absolute_rate_limit and self.action_mode is not ActionMode.ABSOLUTE:
            raise ValueError("absolute_rate_limit applies to absolute action mode only")
        for label, items in (
            ("control override", [o.name for o in self.control_overrides]),
            ("initial state", [s.name for s in self.initial_state]),
            ("initial control", [u.name for u in self.initial_control]),
        ):
            if len(set(items)) != len(items):
                raise ValueError(f"duplicate {label} entries: {items}")
        # force the resolvers to run so a partial override that inverts a bound fails here,
        # at construction, rather than at the first reset()
        self.effective_control_bounds()
        # u0 can only be range-checked once the (possibly overridden) bounds are known
        self.effective_u0()

    @property
    def n_steps(self) -> int:
        """Control steps per episode, derived so the season stays fixed when ``dt`` changes."""
        return round(self.episode_days * 86400.0 / self.dt)

    def effective_control_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Declared control bounds with any overrides applied."""
        lo, hi = CONTROL.bounds()  # a fresh pair each call, safe to modify
        for ov in self.control_overrides:
            i = CONTROL.idx(ov.name)
            if ov.lo is not None:
                lo[i] = ov.lo
            if ov.hi is not None:
                hi[i] = ov.hi
        bad = np.flatnonzero(lo > hi)
        if bad.size:
            names = [CONTROL.names[i] for i in bad]
            raise ValueError(f"overrides invert the bounds for {names}")
        return lo, hi

    def effective_du_max(self) -> NDArray[np.float64]:
        """Declared per-step rate limits with any overrides applied."""
        du = CONTROL.attribute_array("du_max")
        for ov in self.control_overrides:
            if ov.du_max is not None:
                du[CONTROL.idx(ov.name)] = ov.du_max
        return du

    def effective_x0(self) -> NDArray[np.float64]:
        """The declared initial state with any per-component replacements applied."""
        x0 = STATE.attribute_array("default")
        for init in self.initial_state:
            x0[STATE.idx(init.name)] = init.value
        return x0

    def effective_u0(self) -> NDArray[np.float64]:
        """The declared initial control with any per-component replacements applied.

        Checked against the *effective* bounds, not the declared ones, so raising a control's limit
        also permits a higher starting value.
        """
        u0 = CONTROL.attribute_array("default")
        for init in self.initial_control:
            u0[CONTROL.idx(init.name)] = init.value
        lo, hi = self.effective_control_bounds()
        bad = np.flatnonzero((u0 < lo) | (u0 > hi))
        if bad.size:
            names = [CONTROL.names[i] for i in bad]
            raise ValueError(f"initial control outside the effective bounds for {names}")
        return u0
