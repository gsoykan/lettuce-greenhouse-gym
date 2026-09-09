"""A grower's setpoint rules: heat when cold, vent when hot or humid, dose CO2 in daylight."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..envs.control_env import LettuceGreenhouseEnv
from ..variables.controls import CONTROL
from ..variables.exogenous import EXOGENOUS
from ..variables.observables import OBSERVABLE
from .base import Controller


@dataclass(frozen=True)
class GrowerRules:
    """Setpoints and proportional gains of the grower's climate computer.

    The setpoints are the regime under which the model's validation crops were grown, as reported
    in van Henten (1994), *Validation of a dynamic lettuce growth model for greenhouse climate
    control*, Agricultural Systems 45: a 14 degC day setpoint (raised further on bright days, a
    refinement not modelled here), 10 degC at night, and CO2 supplied in daylight to at most
    750 ppm depending on radiation and vent opening. The humidity limit is the benchmark's 80%
    comfort bound; the paper gives no number, only that the grower vented to keep humidity down.

    The gains are engineering choices with no literature source: each is set so its actuator
    reaches full range over a plausible error (5 degC of heating deficit, 3.75 degC of excess
    temperature, 200 ppm of CO2 deficit).
    """

    temp_day: float = 14.0  # degC, heating setpoint when it is light outside
    temp_night: float = 10.0  # degC, heating setpoint when it is dark
    day_radiation: float = 50.0  # W/m2, outdoor radiation above which it counts as day
    heat_gain: float = 30.0  # W/m2 per degC below the setpoint
    vent_deadband: float = 2.0  # degC above the setpoint before venting starts
    vent_temp_gain: float = 2.0  # mm/s per degC above setpoint + deadband
    rh_max: float = 80.0  # %, venting starts above this humidity
    vent_rh_gain: float = 0.5  # mm/s per % RH above rh_max
    co2_target: float = 750.0  # ppm, daytime dosing target
    co2_gain: float = 0.006  # mg/m2/s per ppm below the target
    co2_vent_cutoff: float = 2.0  # mm/s; no dosing when vents are open wider than this

    def __post_init__(self) -> None:
        for name in ("heat_gain", "vent_temp_gain", "vent_rh_gain", "co2_gain"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 < self.rh_max <= 100:
            raise ValueError(f"rh_max must be in (0, 100], got {self.rh_max}")


class GrowerHeuristic(Controller):
    """Proportional setpoint control, as a grower's climate computer does it.

    Reads the four observables and the *current* outdoor weather, which every greenhouse measures;
    it never reads the true state or future weather, so it is not privileged. Works under any
    observation layout because it takes the weather from the env, not from ``obs``.
    """

    def __init__(self, rules: GrowerRules | None = None) -> None:
        self.rules = rules if rules is not None else GrowerRules()

    def control(self, obs: NDArray[np.float32], env: LettuceGreenhouseEnv) -> NDArray[np.float64]:
        r = self.rules
        y = OBSERVABLE.unpack(obs[:4])
        v = EXOGENOUS.unpack(env.weather_forecast(1)[:, 0])
        Ui = CONTROL.index_enum()

        day = v.rad > r.day_radiation
        setpoint = r.temp_day if day else r.temp_night

        # Each rule is a proportional band, clipped to the episode's actuator limits (which may be
        # overridden, so the declared CONTROL bounds are not used here).
        u = np.empty(CONTROL.size)
        u[Ui.HEATING] = r.heat_gain * (setpoint - y.indoor_temp)
        u[Ui.VENTILATION] = r.vent_temp_gain * max(
            0.0, y.indoor_temp - setpoint - r.vent_deadband
        ) + r.vent_rh_gain * max(0.0, y.rh - r.rh_max)
        u = np.clip(u, self.lo, self.hi)

        # CO2 dosing depends on the clipped vent opening: dosing into open windows is wasted.
        u[Ui.CO2_SUPPLY] = 0.0
        if day and u[Ui.VENTILATION] <= r.co2_vent_cutoff:
            u[Ui.CO2_SUPPLY] = r.co2_gain * (r.co2_target - y.co2_ppm)
        return np.clip(u, self.lo, self.hi)
