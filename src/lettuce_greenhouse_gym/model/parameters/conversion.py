"""Physical constants for unit conversions in the measurement map ``g``.

These are *not* van Henten model parameters — they are universal physical constants (gas constant,
molar masses, standard pressure) plus the empirical Magnus coefficients used to convert state
densities to grower-facing units (CO2 ppm, relative humidity). They are never randomized, hence a
separate registry named ``CONVERSION_CONSTANTS`` (not ``..._PARAMS``). The Magnus coefficients have
no formal symbol.
"""

from .base import Constant

CONVERSION_CONSTANTS: tuple[Constant, ...] = (
    Constant(
        "R",
        "gas_const",
        "J/(mol*K)",
        8.3144598,
        description="ideal gas constant (CO2 ppm conversion)",
    ),
    Constant("M_CO2", "molar_mass_co2", "kg/mol", 44.01e-3, description="molar mass of CO2"),
    Constant("P", "pressure", "Pa", 101325.0, description="standard atmospheric pressure"),
    Constant("M_w", "molar_mass_water", "kg/mol", 18.01528e-3, description="molar mass of water"),
    Constant(
        "", "magnus_a", "Pa", 610.78, description="Magnus saturation-vapour-pressure coefficient"
    ),
    Constant(
        "", "magnus_b", "degC", 238.3, description="Magnus saturation-vapour-pressure coefficient"
    ),
    Constant(
        "", "magnus_c", "-", 17.2694, description="Magnus saturation-vapour-pressure coefficient"
    ),
)
