"""Unit conversions between model quantities and the units growers and data files use.

One source of truth for both directions: ``measurement`` converts state densities to ppm and RH,
while the weather loader converts recorded ppm and RH back to densities. Same formulas, so they
cannot drift apart.

The functions accept numpy arrays *or* CasADi symbolics: ``measurement`` builds a symbolic graph,
the loader works on arrays. ``exp`` is dispatched accordingly: ``numpy.exp`` on a CasADi ``SX`` is
not reliable.

Symbols used below: ``T`` air temperature [degC], ``T_abs`` 0 degC in kelvin, ``R`` the ideal gas
constant, ``P`` standard pressure, ``M_CO2`` / ``M_w`` the molar masses of CO2 and water.
"""

import casadi
import numpy as np

from .model.parameters import CONVERSION_CONSTANTS, MODEL_COEFFS, ParameterSet

_CONV = ParameterSet.from_defaults(CONVERSION_CONSTANTS)
_GAS_CONST = _CONV.get("gas_const")
_MOLAR_MASS_CO2 = _CONV.get("molar_mass_co2")
_PRESSURE = _CONV.get("pressure")
_MOLAR_MASS_WATER = _CONV.get("molar_mass_water")
_MAGNUS_A = _CONV.get("magnus_a")
_MAGNUS_B = _CONV.get("magnus_b")
_MAGNUS_C = _CONV.get("magnus_c")
# 0 degC in kelvin; it lives in MODEL_COEFFS because van Henten's table carries it there
_KELVIN = ParameterSet.from_defaults(MODEL_COEFFS).get("temp_abs")


def _exp(x):
    """exp that works on CasADi symbolics and numpy arrays alike."""
    if isinstance(x, casadi.SX | casadi.MX | casadi.DM):
        return casadi.exp(x)
    return np.exp(x)


def saturation_vapour_pressure(temp):
    """Magnus saturation vapour pressure [Pa] at ``temp`` [degC].

    ``p_sat(T) = a * exp(c * T / (T + b))`` with the Magnus coefficients
    ``a = 610.78 Pa``, ``b = 238.3 degC``, ``c = 17.2694``.
    """
    return _MAGNUS_A * _exp(_MAGNUS_C * temp / (temp + _MAGNUS_B))


def co2_density_to_ppm(temp, density):
    """CO2 mass density [kg/m3] -> ppm, via the ideal gas law.

    ``ppm = 1e6 * R * (T + T_abs) * rho / (P * M_CO2)``
    """
    return 1e6 * _GAS_CONST * (temp + _KELVIN) * density / (_PRESSURE * _MOLAR_MASS_CO2)


def co2_ppm_to_density(temp, ppm):
    """CO2 ppm -> mass density [kg/m3]. Inverse of :func:`co2_density_to_ppm`.

    ``rho = P * 1e-6 * ppm * M_CO2 / (R * (T + T_abs))``
    """
    return _PRESSURE * 1e-6 * ppm * _MOLAR_MASS_CO2 / (_GAS_CONST * (temp + _KELVIN))


def vapour_density_to_rh(temp, density):
    """Water-vapour density [kg/m3] -> relative humidity [%].

    ``RH = 100 * R * (T + T_abs) * rho_v / (M_w * p_sat(T))``

    Deliberately **not** capped at 100%: saturation is a measurement limit, not a physical one, and
    capping here would stop :func:`rh_to_vapour_density` being a true inverse above saturation.
    Callers that model an instrument apply their own cap.
    """
    return (
        100.0
        * _GAS_CONST
        * (temp + _KELVIN)
        / (_MOLAR_MASS_WATER * saturation_vapour_pressure(temp))
        * density
    )


def rh_to_vapour_density(temp, rh):
    """Relative humidity [%] -> water-vapour density [kg/m3]. Inverse of the above.

    ``rho_v = (RH / 100) * p_sat(T) * M_w / (R * (T + T_abs))``
    """
    pascals = (rh / 100.0) * saturation_vapour_pressure(temp)
    return pascals * _MOLAR_MASS_WATER / (_GAS_CONST * (temp + _KELVIN))
