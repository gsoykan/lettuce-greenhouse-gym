"""Economic coefficients used by the reward (not by the dynamics).

Prices for heating energy, CO2 and lettuce. These parameterise the profit term and belong to an
economic scenario, so they live apart from the physical model coefficients.

The values are the **standard benchmark's contemporary euro prices**, chosen so results stay
comparable with the RL/MPC literature built on them. They are not van Henten's, who states all four
in 1992 Dutch guilders: ``c_pri,1 = 1.8 Hfl/m2``, ``c_pri,2 = 1.6 Hfl/kg``, ``c_q = 6.35e-9 Hfl/J``,
``c_co2 = 0.42 Hfl/kg``. Two of ours are those values converted at 2.20371 NLG/EUR
(``co2_cost``, ``product_price_1``); the other two are deliberate modern re-pricings.

Consequence, worth knowing before quoting results: the price *levels* weight yield about 30x and
energy about 12x higher than van Henten's own objective, a net shift of roughly 2.5x toward yield
-- so the optimum under this reward is not the optimum of the problem as published in 2003.
"""

from .base import Constant

ECONOMIC_COEFFS: tuple[Constant, ...] = (
    Constant("c_q", "energy_cost", "EUR/kWh", 0.1281, description="price of heating energy"),
    Constant("c_co2", "co2_cost", "EUR/kg", 0.1906, description="price of CO2"),
    # kept as the exact expression, not a rounded decimal, to preserve numeric parity.
    Constant(
        "c_pri,1",
        "product_price_1",
        "EUR/m2",
        1.8 / 2.20371,
        description="lettuce price coefficient",
    ),
    Constant(
        "c_pri,2", "product_price_2", "EUR/kg", 22.285125, description="lettuce price coefficient"
    ),
)
