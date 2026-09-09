from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from lettuce_greenhouse_gym.model.parameters import (
    CONVERSION_CONSTANTS,
    ECONOMIC_COEFFS,
    MODEL_COEFFS,
    Constant,
    ParameterSet,
)


# ---- Section A: the machinery (throwaway constants) ----------------------------------------------
def test_constant_is_frozen():
    c = Constant("c_x", "x", "u", 5.0)
    with pytest.raises(FrozenInstanceError):
        c.default = 9.0


def test_constant_is_slotted():
    c = Constant("c_x", "x", "u", 5.0)
    with pytest.raises(AttributeError):
        object.__setattr__(c, "typo", 1)


def test_constant_rejects_swapped_bounds():
    with pytest.raises(ValueError, match="exceeds hi"):
        Constant("c_x", "x", "u", 5.0, lo=10.0, hi=0.0)


def test_constant_rejects_default_below_lo():
    with pytest.raises(ValueError, match="below lo"):
        Constant("c_x", "x", "u", -1.0, lo=0.0)


def test_constant_rejects_default_above_hi():
    with pytest.raises(ValueError, match="above hi"):
        Constant("c_x", "x", "u", 99.0, hi=10.0)


THROWAWAY = (
    Constant("c_a", "a", "u", 1.0, lo=0.0, hi=10.0),  # bounded -> lets us test OOB paths
    Constant("c_b", "b", "u", 2.0),  # unbounded
    Constant("c_leak", "leak", "m/s", 0.75e-5, lo=0.0),
)


def test_from_defaults_to_array():
    ps = ParameterSet.from_defaults(THROWAWAY)
    arr = ps.to_array()
    assert arr.dtype == np.float64
    assert arr.shape == (3,)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 0.75e-5])


def test_get_and_idx():
    ps = ParameterSet.from_defaults(THROWAWAY)
    assert ps.idx("leak") == 2
    assert ps.get("b") == 2.0
    with pytest.raises(ValueError, match="unknown parameter"):
        ps.idx("nope")


def test_override_returns_new_set():
    ps = ParameterSet.from_defaults(THROWAWAY)
    ps2 = ps.override(a=5.0)
    assert ps2.get("a") == 5.0
    assert ps.get("a") == 1.0  # original untouched (immutability)


def test_override_rejects_unknown_name():
    ps = ParameterSet.from_defaults(THROWAWAY)
    with pytest.raises(ValueError, match="unknown parameter"):
        ps.override(nope=1.0)


def test_override_rejects_out_of_bounds():
    ps = ParameterSet.from_defaults(THROWAWAY)
    with pytest.raises(ValueError, match="above hi"):
        ps.override(a=999.0)


def test_construction_rejects_wrong_length():
    with pytest.raises(ValueError, match="entries"):
        ParameterSet(THROWAWAY, (1.0, 2.0))


def test_construction_rejects_out_of_bounds():
    with pytest.raises(ValueError, match="below lo"):
        ParameterSet(THROWAWAY, (-1.0, 2.0, 0.75e-5))


def test_construction_rejects_duplicate_name():
    dup = (Constant("c_a", "a", "u", 1.0), Constant("c_a2", "a", "u", 2.0))
    with pytest.raises(ValueError, match="duplicate"):
        ParameterSet.from_defaults(dup)


def test_values_are_immutable():
    ps = ParameterSet.from_defaults(THROWAWAY)
    with pytest.raises(TypeError):
        ps.values[0] = 9.0


# ---- Section B: the coefficient registries ------------------------------------------------------

ORACLE_PARAMS = [
    9348.0,
    17.4,
    239.0,
    10998.0,
    8314.0,
    273.15,
    0.75e-5,
    4.1,
    4.1,
    3e4,
    1290.0,
    6.1,
    0.2,
    0.544,
    2.65e-7,
    4.87e-7,
    53.0,
    3.55e-9,
    5.11e-6,
    2.3e-4,
    6.29e-4,
    5.2e-5,
    3.6e-3,  # 0-22  model
    8.3144598,
    44.01e-3,
    101325.0,
    18.01528e-3,
    610.78,
    238.3,
    17.2694,  # 23-29 conversion
    0.1281,
    0.1906,
    1.8 / 2.20371,
    22.285125,  # 30-33 economic
]

ALL_CONSTANTS = MODEL_COEFFS + CONVERSION_CONSTANTS + ECONOMIC_COEFFS


@pytest.mark.parametrize(
    ("registry", "expected_size"),
    [
        (MODEL_COEFFS, 23),
        (CONVERSION_CONSTANTS, 7),
        (ECONOMIC_COEFFS, 4),
    ],
    ids=["model", "conversion", "economic"],
)
def test_registry_size(registry, expected_size):
    assert len(registry) == expected_size


def test_concatenation_matches_oracle():
    values = [c.default for c in ALL_CONSTANTS]
    np.testing.assert_array_equal(values, ORACLE_PARAMS)


def test_all_names_unique():
    assert len(ALL_CONSTANTS) == len({c.name for c in ALL_CONSTANTS})


def test_leak_parity_trap():
    assert MODEL_COEFFS[6].name == "leak"
    assert MODEL_COEFFS[6].default == 0.75e-5


def test_dynamics_c_vector_is_23():
    c = ParameterSet.from_defaults(MODEL_COEFFS)
    assert c.size == 23
    assert c.to_array().dtype == np.float64
