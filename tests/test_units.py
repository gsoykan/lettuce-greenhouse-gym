"""Tests for the unit conversions.

The point of this module is that two call sites, the measurement map and the weather loader,
use the *same* formulas in opposite directions. So the tests that matter are the inverse
round-trips, and the check that the symbolic and numeric paths agree.
"""

import casadi
import numpy as np
import pytest

from lettuce_greenhouse_gym.units import (
    co2_density_to_ppm,
    co2_ppm_to_density,
    rh_to_vapour_density,
    saturation_vapour_pressure,
    vapour_density_to_rh,
)

TEMPS = np.array([-5.0, 0.0, 5.0, 15.0, 25.0, 35.0])


# ---- inverse round-trips (the reason this module exists) -----------------------------------------
@pytest.mark.parametrize("temp", TEMPS, ids=lambda t: f"{t:g}C")
def test_co2_round_trips_through_ppm(temp):
    density = np.array([1e-4, 5e-4, 1e-3, 4e-3])
    np.testing.assert_allclose(
        co2_ppm_to_density(temp, co2_density_to_ppm(temp, density)), density, rtol=1e-12
    )


@pytest.mark.parametrize("temp", TEMPS, ids=lambda t: f"{t:g}C")
def test_vapour_round_trips_through_rh(temp):
    density = np.array([0.001, 0.005, 0.01, 0.02])
    np.testing.assert_allclose(
        rh_to_vapour_density(temp, vapour_density_to_rh(temp, density)), density, rtol=1e-12
    )


def test_vapour_round_trip_still_holds_above_saturation():
    """`vapour_density_to_rh` is uncapped on purpose, so it stays a true inverse past 100% RH.

    Capping inside `units` would silently break this; the cap belongs to `measurement`.
    """
    density = 0.05  # supersaturated at 15 degC
    assert vapour_density_to_rh(15.0, density) > 100.0
    np.testing.assert_allclose(
        rh_to_vapour_density(15.0, vapour_density_to_rh(15.0, density)), density, rtol=1e-12
    )


# ---- the symbolic / numeric contract -------------------------------------------------------------
def test_symbolic_and_numeric_paths_agree():
    """`measurement` calls these on CasADi symbols, the weather loader on numpy arrays."""
    t, val = casadi.SX.sym("t"), casadi.SX.sym("val")
    to_ppm = casadi.Function("to_ppm", [t, val], [co2_density_to_ppm(t, val)])
    to_rh = casadi.Function("to_rh", [t, val], [vapour_density_to_rh(t, val)])
    to_density = casadi.Function("to_density", [t, val], [rh_to_vapour_density(t, val)])
    for temp in TEMPS:
        assert float(to_ppm(temp, 1e-3)) == pytest.approx(co2_density_to_ppm(temp, 1e-3))
        assert float(to_rh(temp, 0.008)) == pytest.approx(vapour_density_to_rh(temp, 0.008))
        assert float(to_density(temp, 60.0)) == pytest.approx(rh_to_vapour_density(temp, 60.0))


# ---- known values ---------------------------------------------------------------------------------
def test_known_conversion_values():
    """Pins the absolute scale, which a round-trip alone cannot: a wrong constant applied in both
    directions would round-trip perfectly and still be wrong."""
    assert co2_density_to_ppm(15.0, 1e-3) == pytest.approx(537.26, abs=0.01)
    assert vapour_density_to_rh(15.0, 0.008) == pytest.approx(62.64, abs=0.01)
    assert saturation_vapour_pressure(0.0) == pytest.approx(610.78, abs=1e-9)


def test_saturation_pressure_increases_with_temperature():
    p = saturation_vapour_pressure(TEMPS)
    assert np.all(np.diff(p) > 0)
