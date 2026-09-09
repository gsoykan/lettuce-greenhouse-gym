import numpy as np
import pytest

from lettuce_greenhouse_gym.model.parameters import (
    MODEL_COEFFS,
    Constant,
    FixedParameterProvider,
    ParameterSet,
    RandomizedParameterProvider,
)

BASE = ParameterSet.from_defaults(MODEL_COEFFS)

# a set with a *bounded* constant, for the physical-bounds rejection tests (MODEL_COEFFS are unbounded)
BOUNDED = ParameterSet.from_defaults((Constant("c_x", "x", "u", 5.0, lo=0.0, hi=10.0),))


# ---- FixedParameterProvider ----------------------------------------------------------------------
def test_fixed_returns_nominal():
    fp = FixedParameterProvider(BASE)
    result = fp.sample(np.random.default_rng(0))
    np.testing.assert_array_equal(BASE.to_array(), result.to_array())


def test_fixed_returns_same_object():
    fp = FixedParameterProvider(BASE)
    result = fp.sample(np.random.default_rng(0))
    assert result is BASE  # returns the base directly (no copy) — safe because it is immutable


def test_fixed_ignores_rng():
    fp = FixedParameterProvider(BASE)
    result = fp.sample(np.random.default_rng(0))
    result_2 = fp.sample(np.random.default_rng(1))
    np.testing.assert_array_equal(result.to_array(), result_2.to_array())


# ---- RandomizedParameterProvider: behaviour ------------------------------------------------------
def test_randomized_draws_within_range():
    rp = RandomizedParameterProvider(BASE, ranges={"leak": (1e-5, 2e-5)})
    result = rp.sample(np.random.default_rng(0))
    leak_value = result.get("leak")
    assert 1e-5 <= leak_value <= 2e-5


def test_randomized_leaves_others_nominal():
    rp = RandomizedParameterProvider(BASE, ranges={"leak": (1e-5, 2e-5)})
    result = rp.sample(np.random.default_rng(0))
    # a parameter not in `ranges` keeps its nominal value
    assert result.get("cap_co2") == BASE.get("cap_co2")


def test_randomized_is_reproducible():
    rp = RandomizedParameterProvider(BASE, ranges={"leak": (1e-5, 2e-5)})
    # fresh generator with the same seed each time -> identical draws
    a = rp.sample(np.random.default_rng(7))
    b = rp.sample(np.random.default_rng(7))
    np.testing.assert_array_equal(a.to_array(), b.to_array())


# ---- RandomizedParameterProvider: construction validation ----------------------------------------
def test_randomized_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown parameter"):
        RandomizedParameterProvider(BASE, ranges={"nope": (0.0, 1.0)})


def test_randomized_rejects_lo_greater_than_hi():
    with pytest.raises(ValueError, match="exceeds hi"):
        RandomizedParameterProvider(BASE, ranges={"leak": (2.0, 1.0)})


def test_randomized_rejects_range_above_physical_hi():
    with pytest.raises(ValueError, match="above physical bound"):
        RandomizedParameterProvider(BOUNDED, ranges={"x": (5.0, 100.0)})


def test_randomized_rejects_range_below_physical_lo():
    with pytest.raises(ValueError, match="below physical bound"):
        RandomizedParameterProvider(BOUNDED, ranges={"x": (-1.0, 5.0)})


# ---- relative scaling ------------------------------------------------------------------------
def test_relative_provider_scales_every_coefficient_within_the_half_width():
    from lettuce_greenhouse_gym.model.parameters import MODEL_COEFFS, ParameterSet
    from lettuce_greenhouse_gym.model.parameters.providers import RandomizedParameterProvider

    base = ParameterSet.from_defaults(MODEL_COEFFS)
    provider = RandomizedParameterProvider.relative(base, 0.05)
    rng = np.random.default_rng(0)
    for _ in range(50):
        drawn = provider.sample(rng).to_array()
        ratio = drawn / base.to_array()
        assert np.all(ratio >= 0.95 - 1e-12)
        assert np.all(ratio <= 1.05 + 1e-12)
    assert provider.per_step is False


def test_relative_provider_restricts_to_named_coefficients_and_flags_per_step():
    from lettuce_greenhouse_gym.model.parameters import MODEL_COEFFS, ParameterSet
    from lettuce_greenhouse_gym.model.parameters.providers import RandomizedParameterProvider

    base = ParameterSet.from_defaults(MODEL_COEFFS)
    provider = RandomizedParameterProvider.relative(base, 0.1, ("leak",), per_step=True)
    drawn = provider.sample(np.random.default_rng(1))
    changed = [n for n in base.names if drawn.get(n) != base.get(n)]
    assert changed == ["leak"]
    assert provider.per_step is True
    with pytest.raises(ValueError, match="half_width"):
        RandomizedParameterProvider.relative(base, 1.5)


def test_default_step_hook_keeps_the_episode_parameters():
    from lettuce_greenhouse_gym.model.parameters import MODEL_COEFFS, ParameterSet
    from lettuce_greenhouse_gym.model.parameters.providers import FixedParameterProvider

    base = ParameterSet.from_defaults(MODEL_COEFFS)
    assert FixedParameterProvider(base).step(np.random.default_rng(0), 3, base) is None
