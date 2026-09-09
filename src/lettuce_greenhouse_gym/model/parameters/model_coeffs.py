"""The 23 van Henten (1994) model coefficients — the ``c`` the dynamics ``f`` consume.

This is the registry a provider randomizes (the exo seam): every entry is a physical coefficient of
the ODE. The order is fixed and load-bearing — the right-hand side indexes this vector positionally,
so entries must never be reordered. Values follow the van Henten reduced model; symbols use the
notation of van Henten (2003), *Sensitivity Analysis of an Optimal Control Problem in Greenhouse
Climate Management*, Biosystems Engineering, whose equation numbers the comments below refer to.
"""

from .base import Constant

MODEL_COEFFS: tuple[Constant, ...] = (
    Constant(
        "c_v,1", "sat_vp1", "J/m3", 9348.0, description="saturation water vapour pressure coeff"
    ),
    Constant("c_v,2", "sat_vp2", "-", 17.4, description="saturation water vapour pressure coeff"),
    Constant(
        "c_v,3", "sat_vp3", "degC", 239.0, description="saturation water vapour pressure coeff"
    ),
    # Carried for index alignment but never read by the dynamics: c_v,4 parameterises eqn (12),
    # the RH *constraint* (not the transpiration term, which uses c_v,1), and RH here comes from
    # the Magnus formula instead.
    Constant(
        "c_v,4",
        "sat_vp4",
        "J/m3",
        10998.0,
        description="saturation water vapour coeff for the RH constraint (eqn 12)",
    ),
    Constant("c_R", "gas_const_kmol", "J/(K*kmol)", 8314.0, description="gas constant"),
    Constant("c_T,abs", "temp_abs", "K", 273.15, description="temperature in K at 0 degC"),
    # Known discrepancy: van Henten's 1994 thesis and the 2003 paper both give c_leak as
    # 0.75e-4 m/s, while the established benchmark value is 0.75e-5, ten times smaller. We use
    # the benchmark value for comparability. Unresolved, so treat conclusions that hinge on this
    # coefficient's magnitude as provisional.
    Constant(
        "c_leak", "leak", "m/s", 0.75e-5, description="leakage air exchange through the cover"
    ),
    Constant(
        "c_cap,c", "cap_co2", "m", 4.1, description="volumetric CO2 capacity of greenhouse air"
    ),
    Constant(
        "c_cap,h",
        "cap_vapor",
        "m",
        4.1,
        description="volumetric vapour capacity of greenhouse air",
    ),
    Constant(
        "c_cap,q", "cap_heat", "J/(m2*degC)", 3e4, description="heat capacity of greenhouse air"
    ),
    Constant(
        "c_cap,q,v",
        "cap_heat_vent",
        "J/(m3*degC)",
        1290.0,
        description="heat capacity per air volume",
    ),
    Constant(
        "c_ai,ou",
        "heat_transfer_cover",
        "W/(m2*degC)",
        6.1,
        description="heat transfer through the cover",
    ),
    Constant(
        "c_rad,q",
        "rad_heat_load",
        "-",
        0.2,
        description="heat load coefficient from solar radiation",
    ),
    Constant("c_αβ", "yield_factor", "-", 0.544, description="yield factor"),
    Constant(
        "c_resp,d",
        "resp_dry",
        "1/s",
        2.65e-7,
        description="respiration rate (respired dry matter)",
    ),
    Constant(
        "c_resp,c", "resp_co2", "1/s", 4.87e-7, description="respiration rate (produced CO2)"
    ),
    Constant("c_pl,d", "canopy_surface", "m2/kg", 53.0, description="effective canopy surface"),
    Constant("c_rad,phot", "light_use_eff", "kg/J", 3.55e-9, description="light use efficiency"),
    Constant(
        "c_co2,1",
        "photo_co2_1",
        "m/(s*degC2)",
        5.11e-6,
        description="temperature effect on CO2 diffusion",
    ),
    Constant(
        "c_co2,2",
        "photo_co2_2",
        "m/(s*degC)",
        2.3e-4,
        description="temperature effect on CO2 diffusion",
    ),
    Constant(
        "c_co2,3", "photo_co2_3", "m/s", 6.29e-4, description="temperature effect on CO2 diffusion"
    ),
    Constant("c_Γ", "co2_compensation", "kg/m3", 5.2e-5, description="CO2 compensation point"),
    Constant(
        "c_v,pl,ai",
        "transp_coeff",
        "m/s",
        3.6e-3,
        description="canopy transpiration mass-transfer coeff",
    ),
)
