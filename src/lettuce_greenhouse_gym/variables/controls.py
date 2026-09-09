"""The three van Henten (1994) lettuce control (actuator) variables.

Controls are given in *actuator* units (mg/m2/s, mm/s, W/m2); the dynamics rescale CO2 supply by
``1e-6`` and ventilation by ``1e-3`` internally to SI.

``du_max`` is the per-step rate limit for delta-action control: an action increments the previous
control by at most ``du_max`` before clamping to ``[lo, hi]``. It defaults to ``hi / 10`` and is a
control-design choice, not a physical bound.
"""

from typing import NamedTuple

from .base import ControlVar, VarGroup


class CO2Supply(ControlVar):
    symbol = "U_c"
    name = "co2_supply"
    unit = "mg/m2/s"
    lo = 0.0
    hi = 1.2
    du_max = 0.12  # hi / 10
    default = 0.0
    description = "CO2 injection rate (rescaled to kg/m2/s in the rhs)."


class Ventilation(ControlVar):
    symbol = "U_v"
    name = "ventilation"
    unit = "mm/s"
    lo = 0.0
    hi = 7.5
    du_max = 0.75  # hi / 10
    default = 0.0
    description = "Ventilation-driven air exchange rate (rescaled to m/s in the rhs)."


class Heating(ControlVar):
    symbol = "U_q"
    name = "heating"
    unit = "W/m2"
    lo = 0.0
    hi = 150.0
    du_max = 15.0  # hi / 10
    default = 50.0
    description = "Heating power supplied to the greenhouse air."


class ControlView(NamedTuple):
    """Statically-typed view of a control array — ``CONTROL.unpack(u).heating``."""

    co2_supply: float
    ventilation: float
    heating: float


CONTROL: VarGroup[ControlVar, ControlView] = VarGroup(
    CO2Supply,
    Ventilation,
    Heating,
    view=ControlView,
)
