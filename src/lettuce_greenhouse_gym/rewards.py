"""The reward: profit over one control step, minus a soft penalty for leaving comfort bounds.

The economic term's *functional form* is van Henten's own objective: van Henten (2003),
*Sensitivity Analysis of an Optimal Control Problem in Greenhouse Climate Management*, Biosystems
Engineering, defines the optimal control problem as
``J = c_pri,1 + c_pri,2*X_d(t_f) - integral(c_q*U_q + c_co2*U_c) dt``, so the objective belongs to
the model rather than to a particular experiment. Its *coefficients* are another matter: only two
of the four match that paper (see ``model/parameters/economic.py``). The soft bounds penalty is not
in the paper at all; it is how RL formulations soften its hard constraints, and it is configurable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import casadi
import numpy as np
from numpy.typing import NDArray

from lettuce_greenhouse_gym.model.parameters import ECONOMIC_COEFFS, ParameterSet
from lettuce_greenhouse_gym.variables.observables import OBSERVABLE

from .variables.controls import CONTROL
from .variables.exogenous import EXOGENOUS
from .variables.states import STATE


def _readonly(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """A defensive, immutable copy: the reward must not be able to corrupt the env's arrays."""
    out = np.array(a, dtype=np.float64)
    out.flags.writeable = False
    return out


@dataclass(frozen=True, slots=True)
class RewardContext:
    """Everything that happened during one control step, which is all a reward is allowed to see.

    Carrying the complete transition (not just what today's reward needs) means a third-party
    reward can be written without changing this library: a weather-dependent comfort band needs
    ``v``, a parameter-aware reward needs ``c``. Only the env constructs this; rewards read it.
    """

    x_prev: NDArray[np.float64]  # state before the step
    x_next: NDArray[np.float64]  # state after the step
    u: NDArray[np.float64]  # physical control applied, in actuator units
    v: NDArray[np.float64]  # weather during the step
    y: NDArray[np.float64]  # observables of x_next (OBSERVABLE order)
    c: NDArray[np.float64]  # model coefficients in effect
    t: float  # simulation time at the start of the step, seconds
    dt: float  # step length, seconds

    def __post_init__(self) -> None:
        for name in ("x_prev", "x_next", "u", "v", "y", "c"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PenaltyBound:
    """A soft comfort band on one observable, with a per-side penalty weight.

    These are *comfort* limits a controller is penalised for crossing, distinct from the physical
    bounds in the variable system, which the integrator clamps to. Weights differ per side because
    overheating and overcooling are not equally costly.
    """

    name: str
    lo: float
    hi: float
    w_lo: float
    w_hi: float
    day_lo: float | None = None
    day_hi: float | None = None
    day_radiation: float | None = None
    """A second band for daytime: when outdoor radiation exceeds ``day_radiation`` (W/m2) the band
    is ``[day_lo, day_hi]`` instead of ``[lo, hi]``. Some formulations of the benchmark keep the
    crop warmer by day than by night; set all three or none."""

    def __post_init__(self) -> None:
        if self.name not in OBSERVABLE.names:
            raise ValueError(
                f"unknown observable {self.name!r}; expected one of {OBSERVABLE.names}"
            )
        if self.lo > self.hi:
            raise ValueError(f"{self.name}: lo {self.lo} exceeds hi {self.hi}")
        if self.w_lo < 0 or self.w_hi < 0:
            raise ValueError(f"{self.name}: penalty weights must be non-negative")
        day = (self.day_lo, self.day_hi, self.day_radiation)
        if any(v is not None for v in day) and any(v is None for v in day):
            raise ValueError(f"{self.name}: set day_lo, day_hi and day_radiation together")
        if self.day_lo is not None and self.day_hi is not None and self.day_lo > self.day_hi:
            raise ValueError(f"{self.name}: day_lo {self.day_lo} exceeds day_hi {self.day_hi}")

    @property
    def time_varying(self) -> bool:
        return self.day_radiation is not None

    def limits(self, rad):
        """``(lo, hi)`` in effect at outdoor radiation ``rad``; a CasADi ``if_else`` for symbols."""
        if not self.time_varying:
            return self.lo, self.hi
        is_day = rad > self.day_radiation
        if isinstance(is_day, bool | np.bool_):
            return (self.day_lo, self.day_hi) if is_day else (self.lo, self.hi)
        return (
            casadi.if_else(is_day, self.day_lo, self.lo),
            casadi.if_else(is_day, self.day_hi, self.hi),
        )


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Prices and comfort bands for the economic reward."""

    economics: ParameterSet
    bounds: tuple[PenaltyBound, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for b in self.bounds:
            if b.name in seen:
                raise ValueError(f"duplicate penalty bound for {b.name!r}")
            seen.add(b.name)


def benchmark_reward_config(**price_overrides: float) -> RewardConfig:
    """The comfort bands and prices of the standard RL/MPC benchmark built on the van Henten model.

    The *model* is van Henten's, but these comfort bands and penalty weights are the standard
    benchmark's, not the paper's: van Henten's own constraints are 6.5-20 degC and RH <= 70%.
    They are the defaults because reproducing the benchmark is what makes results comparable.
    Pass overrides for an economic scenario, e.g.
    ``benchmark_reward_config(energy_cost=0.25)``; they go through
    :meth:`ParameterSet.override`, so an unknown or out-of-range price raises rather than being
    silently ignored.
    """
    economics = ParameterSet.from_defaults(ECONOMIC_COEFFS)
    if price_overrides:
        economics = economics.override(**price_overrides)
    return RewardConfig(
        economics=economics,
        bounds=(
            PenaltyBound("co2_ppm", 500.0, 1600.0, 5e-5, 5e-5),
            PenaltyBound("indoor_temp", 10.0, 20.0, 3e-3, 5e-3),
            PenaltyBound("rh", 0.0, 80.0, 7e-4, 7e-4),
        ),
    )


class Reward(ABC):
    """Scores one control step. The swap point: supply your own to change the objective.

    Implementations are callables, so anything with the same signature works where a ``Reward`` is
    expected; subclass this only to inherit the config handling.
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    @abstractmethod
    def __call__(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """Return ``(reward, breakdown)`` for one step.

        The breakdown names each component (revenue, costs, per-bound penalties) so a run can be
        debugged after the fact ("was the policy punished, or merely unprofitable?") and is
        surfaced through the env's ``info``.
        """


class EconomicReward(Reward):
    """Profit over the step, minus the soft comfort penalty.

    ``revenue`` uses ``c_pri,2`` from van Henten (2003) (price per kg dry weight). That paper's
    ``c_pri,1`` is a constant offset over the season; it is identical for every policy and so
    contributes nothing to a per-step reward, which is why it stays in the registry but unused.
    """

    def __call__(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        Xi = STATE.index_enum()
        Ui = CONTROL.index_enum()
        econ = self.config.economics.get

        # --- revenue: growth over the step, priced per kg dry weight -------------------------
        # negative when respiration outweighs photosynthesis (night); that is physically correct
        # and should be felt by the policy, so it is not clipped.
        delta_dry_weight = float(ctx.x_next[Xi.DRY_WEIGHT] - ctx.x_prev[Xi.DRY_WEIGHT])
        revenue = econ("product_price_2") * delta_dry_weight

        # These factors live here, not in `units`: they exist only because the prices are quoted
        # per kWh and per kg, so they belong beside the price they serve.
        # --- costs: actuator units -> the units the prices are quoted in ----------------------
        energy_kwh = float(ctx.u[Ui.HEATING]) * ctx.dt / 3600.0 / 1000.0  # W/m2 -> kWh/m2
        energy_cost = econ("energy_cost") * energy_kwh
        co2_kg = float(ctx.u[Ui.CO2_SUPPLY]) * 1e-6 * ctx.dt  # mg/m2/s -> kg/m2
        co2_cost = econ("co2_cost") * co2_kg

        # --- soft penalty: hinge outside each comfort band ------------------------------------
        breakdown: dict[str, float] = {}
        penalty = 0.0
        rad = float(ctx.v[EXOGENOUS.idx("rad")])
        for b in self.config.bounds:
            value = float(ctx.y[OBSERVABLE.idx(b.name)])
            lo, hi = b.limits(rad)
            contribution = b.w_lo * max(0.0, lo - value) + b.w_hi * max(0.0, value - hi)
            breakdown[f"penalty_{b.name}"] = contribution
            penalty += contribution

        reward = revenue - energy_cost - co2_cost - penalty
        breakdown |= {
            "revenue": revenue,
            "energy_cost": energy_cost,
            "co2_cost": co2_cost,
            "penalty": penalty,
            "delta_dry_weight": delta_dry_weight,
            "energy_kwh": energy_kwh,
            "co2_kg": co2_kg,
        }
        return reward, breakdown


def economic_stage_terms(x, x_next, u, dt: float, config: RewardConfig):
    """``(revenue, energy_cost, co2_cost)`` of one step as CasADi-compatible expressions.

    Same arithmetic as :meth:`EconomicReward.__call__`, term for term. Split from the penalty so an
    optimiser can pair these smooth terms with its own formulation of the comfort penalty.
    """
    Xi, Ui = STATE.index_enum(), CONTROL.index_enum()
    econ = config.economics.get
    revenue = econ("product_price_2") * (x_next[Xi.DRY_WEIGHT] - x[Xi.DRY_WEIGHT])
    energy_cost = econ("energy_cost") * u[Ui.HEATING] * dt / 3600.0 / 1000.0
    co2_cost = econ("co2_cost") * u[Ui.CO2_SUPPLY] * 1e-6 * dt
    return revenue, energy_cost, co2_cost


def comfort_penalty(y, config: RewardConfig, v=None):
    """The hinge penalty of :class:`EconomicReward` on an observable vector, via ``casadi.fmax``.

    ``v`` is the weather during the step; it is needed only when a bound is time-varying.
    """
    if v is None and any(b.time_varying for b in config.bounds):
        raise ValueError("a time-varying comfort bound needs the weather v to decide day or night")
    penalty = 0
    for b in config.bounds:
        value = y[OBSERVABLE.idx(b.name)]
        lo, hi = b.limits(v[EXOGENOUS.idx("rad")]) if b.time_varying else (b.lo, b.hi)
        penalty += b.w_lo * casadi.fmax(0, lo - value) + b.w_hi * casadi.fmax(0, value - hi)
    return penalty


def economic_stage_reward(x, x_next, u, y_next, dt: float, config: RewardConfig, v=None):
    """:class:`EconomicReward` for one step as a CasADi expression, for planners and MPC.

    Accepts CasADi symbols or ``DM`` values. Kept beside the numeric reward so the two are edited
    together; a test pins their equality. ``v`` is needed only for time-varying bounds.
    """
    revenue, energy_cost, co2_cost = economic_stage_terms(x, x_next, u, dt, config)
    return revenue - energy_cost - co2_cost - comfort_penalty(y_next, config, v)
