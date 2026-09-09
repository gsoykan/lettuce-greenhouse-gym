"""The four van Henten (1994) lettuce state variables.

The declaration order fixes the ODE state-array layout. ``default`` is the ``x0`` component;
``lo``/``hi`` are the physical bounds the integrator clamps each state to after a step.
"""

from typing import NamedTuple

from .base import StateVar, VarGroup


class DryWeight(StateVar):
    symbol = "X_d"
    name = "dry_weight"
    unit = "kg/m2"
    lo = 0.002
    hi = 0.6
    default = 0.0035
    description = "Crop dry weight per unit greenhouse floor area."


class IndoorCO2(StateVar):
    symbol = "X_c"
    name = "indoor_co2"
    unit = "kg/m3"
    lo = 0.0
    hi = 0.004
    default = 1e-3
    description = "Indoor air CO2 concentration (mass density)."


class IndoorTemp(StateVar):
    symbol = "X_T"
    name = "indoor_temp"
    unit = "degC"
    lo = 5.0
    hi = 40.0
    default = 15.0
    description = "Indoor air temperature."


class IndoorVapor(StateVar):
    symbol = "X_h"
    name = "indoor_vapor"
    unit = "kg/m3"
    lo = 0.0
    hi = 0.051
    default = 0.008
    description = "Indoor air water-vapor concentration (mass density)."


class StateView(NamedTuple):
    """Statically-typed view of a state array — ``STATE.unpack(x).dry_weight``."""

    dry_weight: float
    indoor_co2: float
    indoor_temp: float
    indoor_vapor: float


STATE: VarGroup[StateVar, StateView] = VarGroup(
    DryWeight,
    IndoorCO2,
    IndoorTemp,
    IndoorVapor,
    view=StateView,
)
