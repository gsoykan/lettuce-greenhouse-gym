"""The four van Henten (1994) lettuce exogenous (weather / disturbance) variables.

These are not chosen by the controller: their values are supplied by a weather provider at each time
step, so they carry no ``default`` or rate limit. The declaration order fixes the disturbance-array
layout. Bounds are the physical envelope (densities and radiation are non-negative; outdoor
temperature is unbounded), used for validation and the observation space — the provider is expected
to stay within them.
"""

from typing import NamedTuple

from .base import ExogenousVar, VarGroup


class OutdoorRadiation(ExogenousVar):
    symbol = "V_rad"
    name = "rad"
    unit = "W/m2"
    lo = 0.0
    hi = float("inf")
    description = "Outdoor global solar radiation."


class OutdoorCO2(ExogenousVar):
    symbol = "V_c"
    name = "out_co2"
    unit = "kg/m3"
    lo = 0.0
    hi = float("inf")
    description = "Outdoor air CO2 concentration (mass density)."


class OutdoorTemp(ExogenousVar):
    symbol = "V_T"
    name = "out_temp"
    unit = "degC"
    lo = -float("inf")
    hi = float("inf")
    description = "Outdoor air temperature."


class OutdoorVapor(ExogenousVar):
    symbol = "V_h"
    name = "out_vapor"
    unit = "kg/m3"
    lo = 0.0
    hi = float("inf")
    description = "Outdoor air water-vapor concentration (mass density)."


class ExogenousView(NamedTuple):
    """Statically-typed view of a disturbance array — ``EXOGENOUS.unpack(v).out_temp``."""

    rad: float
    out_co2: float
    out_temp: float
    out_vapor: float


EXOGENOUS: VarGroup[ExogenousVar, ExogenousView] = VarGroup(
    OutdoorRadiation,
    OutdoorCO2,
    OutdoorTemp,
    OutdoorVapor,
    view=ExogenousView,
)
