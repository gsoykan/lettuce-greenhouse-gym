import casadi

from lettuce_greenhouse_gym.model.base import SymbolicDynamicsModel
from lettuce_greenhouse_gym.model.parameters import MODEL_COEFFS, ParameterSet
from lettuce_greenhouse_gym.units import co2_density_to_ppm, vapour_density_to_rh
from lettuce_greenhouse_gym.variables.controls import CONTROL
from lettuce_greenhouse_gym.variables.exogenous import EXOGENOUS
from lettuce_greenhouse_gym.variables.observables import OBSERVABLE
from lettuce_greenhouse_gym.variables.states import STATE


class VanHentenLettuce(SymbolicDynamicsModel):
    states = STATE
    controls = CONTROL
    exogenous = EXOGENOUS
    observables = OBSERVABLE

    def __init__(self, constants: ParameterSet | None = None) -> None:
        super().__init__(constants or ParameterSet.from_defaults(MODEL_COEFFS))

    def rhs(self, x, u, v, c, t):
        """The four van Henten balances as a CasADi column expression.

        Equations follow van Henten (2003), *Sensitivity Analysis of an Optimal Control Problem in
        Greenhouse Climate Management*, Biosystems Engineering, section 2: crop growth (2), indoor
        CO2 (3), energy (4) and water vapour (5), with the auxiliary fluxes (6)-(11). ``t`` is
        unused: the weather ``v`` is already the value at the current time.
        """
        Xi = self.states.index_enum()
        Ui = self.controls.index_enum()
        Vi = self.exogenous.index_enum()

        def coef(name: str):
            """Named access into the symbolic coefficient vector."""
            return c[self.constants.idx(name)]

        # --- states ---

        dry_weight = x[Xi.DRY_WEIGHT]
        indoor_co2 = x[Xi.INDOOR_CO2]
        indoor_temp = x[Xi.INDOOR_TEMP]
        indoor_vapor = x[Xi.INDOOR_VAPOR]

        # --- controls: actuator units -> SI ---
        co2_supply = u[Ui.CO2_SUPPLY] / 1e6  # mg/m2/s -> kg/m2/s
        ventilation = u[Ui.VENTILATION] / 1e3  # mm/s -> m/s
        heating = u[Ui.HEATING]  # W/m2

        # --- exogenous (weather) ---
        rad = v[Vi.RAD]
        out_co2 = v[Vi.OUT_CO2]
        out_temp = v[Vi.OUT_TEMP]
        out_vapor = v[Vi.OUT_VAPOR]

        # --- coefficients ---
        # sat_vp4 (c_v,4) is absent: the dynamics do not use it. Eqn (10)'s transpiration term
        # takes its saturated-vapour concentration from sat_vp1; c_v,4 belongs to eqn (12), the
        # RH constraint, which `measurement` computes from the Magnus formula instead.
        sat_vp1 = coef("sat_vp1")
        sat_vp2 = coef("sat_vp2")
        sat_vp3 = coef("sat_vp3")
        gas_const_kmol = coef("gas_const_kmol")
        temp_abs = coef("temp_abs")
        leak = coef("leak")
        cap_co2 = coef("cap_co2")
        cap_vapor = coef("cap_vapor")
        cap_heat = coef("cap_heat")
        cap_heat_vent = coef("cap_heat_vent")
        heat_transfer_cover = coef("heat_transfer_cover")
        rad_heat_load = coef("rad_heat_load")
        yield_factor = coef("yield_factor")
        resp_dry = coef("resp_dry")
        resp_co2 = coef("resp_co2")
        canopy_surface = coef("canopy_surface")
        light_use_eff = coef("light_use_eff")
        photo_co2_1 = coef("photo_co2_1")
        photo_co2_2 = coef("photo_co2_2")
        photo_co2_3 = coef("photo_co2_3")
        co2_compensation = coef("co2_compensation")
        transp_coeff = coef("transp_coeff")

        # --- shared terms ---
        # canopy closure: fraction of floor shaded by crop; appears in growth (6) AND transpiration (10)
        canopy_closure = 1 - casadi.exp(-canopy_surface * dry_weight)

        # temperature-dependent CO2 diffusion into the leaves (eqn 6, denominator/numerator factor)
        co2_diffusion = (
            -photo_co2_1 * casadi.power(indoor_temp, 2) + photo_co2_2 * indoor_temp - photo_co2_3
        )
        phot = co2_diffusion * (indoor_co2 - co2_compensation)

        # gross canopy photosynthesis: rectangular hyperbola of light and CO2 (eqn 6)
        gross = canopy_closure * (light_use_eff * rad * phot) / (light_use_eff * rad + phot)

        # temperature-driven respiration factor (eqns 2 and 3)
        respiration = casadi.power(2, 0.1 * indoor_temp - 2.5)

        # saturated water-vapour concentration at air temperature (eqn 10, bracketed term)
        sat_vapor = (
            sat_vp1
            / (gas_const_kmol * (indoor_temp + temp_abs))
            * casadi.exp(sat_vp2 * indoor_temp / (indoor_temp + sat_vp3))
        )

        # --- crop growth (eqn 2) ---
        d_dry_weight = yield_factor * gross - resp_dry * dry_weight * respiration

        # --- indoor CO2 mass balance (eqn 3; vent flux eqn 7) ---
        d_indoor_co2 = (1 / cap_co2) * (
            -gross
            + resp_co2 * dry_weight * respiration
            + co2_supply
            - (ventilation + leak) * (indoor_co2 - out_co2)
        )

        # --- indoor energy balance (eqn 4; Q_vent,q eqn 8, Q_rad,q eqn 9) ---
        d_indoor_temp = (1 / cap_heat) * (
            heating
            - (cap_heat_vent * ventilation + heat_transfer_cover) * (indoor_temp - out_temp)
            + rad_heat_load * rad
        )

        # --- indoor water-vapour balance (eqn 5; transpiration eqn 10, vent flux eqn 11) ---
        d_indoor_vapor = (1 / cap_vapor) * (
            canopy_closure * transp_coeff * (sat_vapor - indoor_vapor)
            - (ventilation + leak) * (indoor_vapor - out_vapor)
        )

        return casadi.vertcat(d_dry_weight, d_indoor_co2, d_indoor_temp, d_indoor_vapor)

    def measurement(self, x):
        """Map the state to grower-facing observables ``y``.

        Returns ``[dry_weight, co2_ppm, indoor_temp, relative_humidity]``. CO2 density and vapour
        density are converted to the units a grower reads (ppm, %); temperature and dry weight pass
        through. Relative humidity is capped at 100%.
        """

        Xi = self.states.index_enum()
        dry_weight = x[Xi.DRY_WEIGHT]
        indoor_co2 = x[Xi.INDOOR_CO2]
        indoor_temp = x[Xi.INDOOR_TEMP]
        indoor_vapor = x[Xi.INDOOR_VAPOR]

        # conversions live in `units`, shared with the weather loader's inverse direction so the
        # two cannot drift apart
        co2_ppm = co2_density_to_ppm(indoor_temp, indoor_co2)

        # capped at 100%: an instrument cannot read past saturation, though the state may exceed it.
        # The cap belongs here, not in `units`, which must stay a true inverse pair.
        relative_humidity = casadi.fmin(100.0, vapour_density_to_rh(indoor_temp, indoor_vapor))

        return casadi.vertcat(dry_weight, co2_ppm, indoor_temp, relative_humidity)
