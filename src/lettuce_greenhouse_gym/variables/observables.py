"""The four grower-facing observables produced by the measurement map ``g``.

Declaration order fixes the layout of ``y`` and must match what ``measurement`` returns. Bounds are
the values reachable from the state box: dry weight and temperature pass through unchanged, CO2 ppm
is the density converted at the hottest/densest corner, and relative humidity is capped at 100% by
the measurement itself.
"""

from typing import NamedTuple

from .base import ObservableVar, VarGroup

# Bounds: dry_weight and indoor_temp pass through the state bounds. co2_ppm and rh are *derived* --
# evaluating g at the corners of the state box (y is monotonic in each state) gives a co2_ppm maximum
# of 2335.49, rounded outward to 2400; rh's 100 is exact, since the measurement caps it. Re-derived
# by tests/test_parity.py::test_observable_bounds_are_derived_from_the_state_box.
#
# These are *physical* bounds, what the model can produce. The tighter *comfort* limits a
# controller is penalised for crossing belong to RewardConfig, never here.
#
# `rh` comes from the Magnus curve. van Henten (2003) eqn (12) defines the RH constraint through a
# different curve (coefficient c_v,4); the two agree to within ~1% but are not the same expression.


class DryWeightObs(ObservableVar):
    symbol = "X_d"
    name = "dry_weight"
    unit = "kg/m2"
    lo = 0.002
    hi = 0.6
    description = "Crop dry weight (passed through from the state)."


class CO2ppmObs(ObservableVar):
    symbol = ""  # no separate paper symbol: this is X_c expressed in ppm
    name = "co2_ppm"
    unit = "ppm"
    lo = 0.0
    hi = 2400.0
    description = "Indoor CO2 concentration in the units a grower reads."


class IndoorTempObs(ObservableVar):
    symbol = "X_T"
    name = "indoor_temp"
    unit = "degC"
    lo = 5.0
    hi = 40.0
    description = "Indoor air temperature (passed through from the state)."


class RelativeHumidityObs(ObservableVar):
    symbol = "RX_h"
    name = "rh"
    unit = "%"
    lo = 0.0
    hi = 100.0
    description = "Indoor relative humidity, capped at 100% by the measurement."


class ObservableView(NamedTuple):
    """Statically-typed view of an observable vector — ``OBSERVABLE.unpack(y).rh``."""

    dry_weight: float
    co2_ppm: float
    indoor_temp: float
    rh: float


OBSERVABLE: VarGroup[ObservableVar, ObservableView] = VarGroup(
    DryWeightObs,
    CO2ppmObs,
    IndoorTempObs,
    RelativeHumidityObs,
    view=ObservableView,
)
