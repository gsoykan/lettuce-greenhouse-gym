"""Tests for the reward: the data structures, the arithmetic, and parity with the benchmark.

``data/reward_parity.npz`` was generated once from the reference implementation (400 transitions at
mixed 900 s / 1800 s steps; 96% violate at least one comfort band, so the penalty hinges are
exercised). It stores each reward component separately, so a failure names the term that broke.
"""

import pathlib

import casadi
import numpy as np
import pytest

from lettuce_greenhouse_gym.model.parameters import ECONOMIC_COEFFS, ParameterSet
from lettuce_greenhouse_gym.rewards import (
    EconomicReward,
    PenaltyBound,
    RewardConfig,
    RewardContext,
    benchmark_reward_config,
    comfort_penalty,
    economic_stage_reward,
)
from lettuce_greenhouse_gym.variables.states import STATE

FIXTURE = np.load(pathlib.Path(__file__).parent / "data" / "reward_parity.npz")
TOL = 1e-12  # observed agreement is ~1e-17; leave headroom for other platforms

REWARD = EconomicReward(benchmark_reward_config())


def _ctx(**overrides) -> RewardContext:
    """A valid, comfortable step; override any field. Crop grew 1e-4 kg, 50 W/m2 heating, no CO2."""
    base = {
        "x_prev": np.array([0.0035, 1e-3, 15.0, 0.008]),
        "x_next": np.array([0.0036, 1e-3, 15.0, 0.008]),
        "u": np.array([0.0, 0.0, 50.0]),
        "v": np.zeros(4),
        "y": np.array([0.0036, 800.0, 15.0, 60.0]),
        "c": np.zeros(23),
        "t": 0.0,
        "dt": 900.0,
    }
    return RewardContext(**(base | overrides))


def test_context_arrays_are_read_only():
    with pytest.raises(ValueError, match="read-only"):
        _ctx().y[0] = 1.0


def test_context_copies_its_inputs():
    """Defensive copy, not just a read-only view: mutating the source must not reach the context."""
    source = np.array([0.0036, 800.0, 15.0, 60.0])
    ctx = _ctx(y=source)
    source[0] = 999.0
    assert ctx.y[0] != 999.0


def test_penalty_bound_rejects_unknown_observable():
    with pytest.raises(ValueError, match="unknown observable"):
        PenaltyBound("not_a_var", 0.0, 1.0, 1.0, 1.0)


def test_penalty_bound_rejects_lo_above_hi():
    with pytest.raises(ValueError, match="exceeds hi"):
        PenaltyBound("rh", 5.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize(("w_lo", "w_hi"), [(-1.0, 1.0), (1.0, -1.0)], ids=["w_lo", "w_hi"])
def test_penalty_bound_rejects_negative_weight(w_lo, w_hi):
    with pytest.raises(ValueError, match="non-negative"):
        PenaltyBound("rh", 0.0, 80.0, w_lo, w_hi)


def test_benchmark_config_uses_the_published_prices():
    cfg = benchmark_reward_config()
    assert cfg.economics.get("energy_cost") == 0.1281
    assert cfg.economics.get("co2_cost") == 0.1906
    assert cfg.economics.get("product_price_2") == 22.285125
    assert [b.name for b in cfg.bounds] == ["co2_ppm", "indoor_temp", "rh"]


def test_benchmark_config_applies_price_overrides():
    cfg = benchmark_reward_config(energy_cost=0.25)
    assert cfg.economics.get("energy_cost") == 0.25
    assert cfg.economics.get("co2_cost") == 0.1906  # untouched


def test_benchmark_config_rejects_an_unknown_price():
    with pytest.raises(ValueError, match="unknown parameter"):
        benchmark_reward_config(enrgy_cost=0.25)


def test_reward_config_rejects_duplicate_bounds():
    econ = ParameterSet.from_defaults(ECONOMIC_COEFFS)
    with pytest.raises(ValueError, match="duplicate"):
        RewardConfig(
            econ,
            (PenaltyBound("rh", 0.0, 80.0, 1.0, 1.0), PenaltyBound("rh", 0.0, 70.0, 1.0, 1.0)),
        )


def test_revenue_and_costs_are_hand_computable():
    _, bd = REWARD(_ctx())
    assert bd["delta_dry_weight"] == pytest.approx(1e-4)
    assert bd["revenue"] == pytest.approx(22.285125 * 1e-4)
    assert bd["energy_kwh"] == pytest.approx(50.0 * 900.0 / 3.6e6)  # W/m2 -> kWh/m2
    assert bd["energy_cost"] == pytest.approx(0.1281 * 0.0125)
    assert bd["co2_cost"] == 0.0


def test_revenue_is_negative_when_the_crop_loses_weight():
    """At night respiration outweighs photosynthesis; that loss must reach the policy, not be clipped."""
    _, bd = REWARD(_ctx(x_next=np.array([0.0034, 1e-3, 15.0, 0.008])))
    assert bd["revenue"] < 0


def test_no_penalty_strictly_inside_the_bands():
    _, bd = REWARD(_ctx())
    assert bd["penalty"] == 0.0


@pytest.mark.parametrize(
    ("rh", "expected"),
    [(80.0, 0.0), (90.0, 7e-4 * 10.0), (100.0, 7e-4 * 20.0)],
    ids=["at_bound", "10_over", "20_over"],
)
def test_rh_hinge_is_zero_at_the_bound_then_linear(rh, expected):
    _, bd = REWARD(_ctx(y=np.array([0.0036, 800.0, 15.0, rh])))
    assert bd["penalty_rh"] == pytest.approx(expected)


def test_temperature_weights_are_asymmetric():
    """3e-3 below vs 5e-3 above: a symmetric implementation passes one of these and fails the other."""
    _, below = REWARD(_ctx(y=np.array([0.0036, 800.0, 5.0, 60.0])))  # 5 below lo=10
    _, above = REWARD(_ctx(y=np.array([0.0036, 800.0, 25.0, 60.0])))  # 5 above hi=20
    assert below["penalty_indoor_temp"] == pytest.approx(3e-3 * 5.0)
    assert above["penalty_indoor_temp"] == pytest.approx(5e-3 * 5.0)


def _evaluate_fixture() -> dict[str, np.ndarray]:
    """Run our reward over every fixture row once; the tests then just compare columns."""
    n = len(FIXTURE["reward"])
    keys = ("reward", "revenue", "energy_cost", "co2_cost", "penalty")
    out = {k: np.empty(n) for k in keys}
    for i in range(n):
        ctx = RewardContext(
            x_prev=FIXTURE["x_prev"][i],
            x_next=FIXTURE["x_next"][i],
            u=FIXTURE["u"][i],
            v=np.zeros(4),
            y=FIXTURE["y"][i],
            c=np.zeros(23),
            t=0.0,
            dt=float(FIXTURE["dt"][i]),
        )
        value, breakdown = REWARD(ctx)
        out["reward"][i] = value
        for k in keys[1:]:
            out[k][i] = breakdown[k]
    return out


OURS = _evaluate_fixture()


@pytest.mark.parametrize(
    "key", ["reward", "revenue", "energy_cost", "co2_cost", "penalty"], ids=lambda k: k
)
def test_matches_reference(key):
    np.testing.assert_allclose(OURS[key], FIXTURE[key], atol=TOL, rtol=0)


# ---- day/night comfort bands ------------------------------------------------------------------------


def _day_night_config():
    base = benchmark_reward_config()
    bounds = tuple(
        PenaltyBound(
            b.name, 10.0, 15.0, b.w_lo, b.w_hi, day_lo=15.0, day_hi=20.0, day_radiation=10.0
        )
        if b.name == "indoor_temp"
        else b
        for b in base.bounds
    )
    return RewardConfig(base.economics, bounds)


def test_day_night_band_switches_on_radiation_in_both_reward_forms():
    config = _day_night_config()
    reward = EconomicReward(config)
    x = STATE.attribute_array("default")
    u = np.array([0.5, 2.0, 60.0])
    y = np.array([3.5, 800.0, 17.0, 70.0])  # 17 degC: inside the day band, above the night band
    for rad, expect_penalty in ((0.0, True), (300.0, False)):
        v = np.array([rad, 7.5e-4, 5.0, 5.0e-3])
        ctx = RewardContext(x_prev=x, x_next=x, u=u, v=v, y=y, c=np.zeros(1), t=0.0, dt=1800.0)
        _, parts = reward(ctx)
        assert (parts["penalty_indoor_temp"] > 0) is expect_penalty
        symbolic = float(economic_stage_reward(x, x, u, y, 1800.0, config, v))
        numeric, _ = reward(ctx)
        assert symbolic == pytest.approx(numeric, abs=1e-12)
    with pytest.raises(ValueError, match="needs the weather"):
        comfort_penalty(y, config)
    with pytest.raises(ValueError, match="together"):
        PenaltyBound("indoor_temp", 10.0, 20.0, 1e-3, 1e-3, day_lo=15.0)
    with pytest.raises(ValueError, match="day_lo"):
        PenaltyBound(
            "indoor_temp", 10.0, 20.0, 1e-3, 1e-3, day_lo=21.0, day_hi=20.0, day_radiation=10.0
        )
    assert not benchmark_reward_config().bounds[1].time_varying


def test_day_night_band_is_symbolic_when_the_weather_is():
    b = PenaltyBound(
        "indoor_temp", 10.0, 15.0, 1e-3, 1e-3, day_lo=15.0, day_hi=20.0, day_radiation=10.0
    )
    rad = casadi.SX.sym("rad")
    lo, hi = b.limits(rad)
    f = casadi.Function("f", [rad], [lo, hi])
    assert [float(z) for z in f(0.0)] == [10.0, 15.0]
    assert [float(z) for z in f(50.0)] == [15.0, 20.0]
