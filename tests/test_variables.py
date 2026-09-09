"""Contract tests for the typed variable system (states / controls / exogenous).

These pin the *behaviour* of ``VarGroup`` and the concrete group definitions — array layout,
bounds, Gym space, (un)packing, and the input-validation guard rails. Parity with any reference
implementation is checked elsewhere (``test_parity.py``); here we only assert the system's own
contract, so the file is self-contained.
"""

from typing import NamedTuple

import numpy as np
import pytest
from gymnasium import spaces

from lettuce_greenhouse_gym.variables.base import StateVar, VarGroup
from lettuce_greenhouse_gym.variables.controls import CONTROL
from lettuce_greenhouse_gym.variables.exogenous import EXOGENOUS
from lettuce_greenhouse_gym.variables.observables import OBSERVABLE
from lettuce_greenhouse_gym.variables.states import STATE

# --------------------------------------------------------------------------------------------------
# One table drives every shared test. Each row fully describes a group's expected contract; a test
# picks the columns it needs. ``sample`` is a valid in-bounds packing; ``oob`` is the same packing
# with exactly one value pushed out of bounds (rest valid, so the test isolates bound-checking).
# --------------------------------------------------------------------------------------------------
CASES = [
    {
        "id": "state",
        "group": STATE,
        "size": 4,
        "names": ("dry_weight", "indoor_co2", "indoor_temp", "indoor_vapor"),
        "symbols": ("X_d", "X_c", "X_T", "X_h"),
        "enum_name": "StateIdx",
        "lo": np.array([0.002, 0.0, 5.0, 0.0]),
        "hi": np.array([0.6, 0.004, 40.0, 0.051]),
        "sample": {
            "dry_weight": 0.0035,
            "indoor_co2": 1e-3,
            "indoor_temp": 15.0,
            "indoor_vapor": 0.008,
        },
        "oob": {
            "dry_weight": 999.0,
            "indoor_co2": 1e-3,
            "indoor_temp": 15.0,
            "indoor_vapor": 0.008,
        },
    },
    {
        "id": "control",
        "group": CONTROL,
        "size": 3,
        "names": ("co2_supply", "ventilation", "heating"),
        "symbols": ("U_c", "U_v", "U_q"),
        "enum_name": "ControlIdx",
        "lo": np.array([0.0, 0.0, 0.0]),
        "hi": np.array([1.2, 7.5, 150.0]),
        "sample": {"co2_supply": 0.5, "ventilation": 3.0, "heating": 50.0},
        "oob": {"co2_supply": 99.0, "ventilation": 3.0, "heating": 50.0},
    },
    {
        "id": "exog",
        "group": EXOGENOUS,
        "size": 4,
        "names": ("rad", "out_co2", "out_temp", "out_vapor"),
        "symbols": ("V_rad", "V_c", "V_T", "V_h"),
        "enum_name": "ExogenousIdx",
        "lo": np.array([0.0, 0.0, -np.inf, 0.0]),
        "hi": np.array([np.inf, np.inf, np.inf, np.inf]),
        # out_temp is deliberately negative: the open lower bound must accept it.
        "sample": {"rad": 800.0, "out_co2": 7e-4, "out_temp": -10.0, "out_vapor": 0.005},
        # only finite bound to violate here is a lower one (all his are +inf).
        "oob": {"rad": -1.0, "out_co2": 7e-4, "out_temp": -10.0, "out_vapor": 0.005},
    },
    {
        "id": "observable",
        "group": OBSERVABLE,
        "size": 4,
        "names": ("dry_weight", "co2_ppm", "indoor_temp", "rh"),
        "symbols": ("X_d", "", "X_T", "RX_h"),
        "enum_name": "ObservableIdx",
        "lo": np.array([0.002, 0.0, 5.0, 0.0]),
        "hi": np.array([0.6, 2400.0, 40.0, 100.0]),
        "sample": {"dry_weight": 0.0035, "co2_ppm": 537.0, "indoor_temp": 15.0, "rh": 62.0},
        "oob": {"dry_weight": 0.0035, "co2_ppm": 537.0, "indoor_temp": 15.0, "rh": 150.0},
    },
]
IDS = [c["id"] for c in CASES]


# ---- layout / metadata ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_size(case):
    assert case["group"].size == case["size"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_names(case):
    assert case["group"].names == case["names"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_symbols(case):
    assert case["group"].symbols == case["symbols"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_index_enum(case):
    enum = case["group"].index_enum()
    assert enum.__name__ == case["enum_name"]
    # every name maps to its declaration position, upper-cased.
    assert [m.name for m in enum] == [n.upper() for n in case["names"]]
    assert [int(m) for m in enum] == list(range(case["size"]))


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_idx_by_name(case):
    for i, name in enumerate(case["names"]):
        assert case["group"].idx(name) == i


# ---- bounds / gym space --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_bounds(case):
    lo, hi = case["group"].bounds()
    assert lo.dtype == np.float64
    assert hi.dtype == np.float64
    assert lo.shape == (case["size"],)
    assert hi.shape == (case["size"],)
    # assert_array_equal handles inf/-inf entries correctly (unlike plain ==).
    np.testing.assert_array_equal(lo, case["lo"])
    np.testing.assert_array_equal(hi, case["hi"])


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_gym_space(case):
    space = case["group"].gym_space()
    assert isinstance(space, spaces.Box)
    assert space.shape == (case["size"],)
    assert space.dtype == np.float32
    np.testing.assert_array_equal(space.low, case["lo"].astype(np.float32))
    np.testing.assert_array_equal(space.high, case["hi"].astype(np.float32))


# ---- pack / unpack round-trip --------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_pack_returns_ordered_float64(case):
    arr = case["group"].pack(**case["sample"])
    assert arr.dtype == np.float64
    assert arr.shape == (case["size"],)
    # values land in declaration order, not kwargs order.
    expected = [case["sample"][n] for n in case["names"]]
    np.testing.assert_array_equal(arr, expected)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_roundtrip(case):
    group = case["group"]
    view = group.unpack(group.pack(**case["sample"]))
    assert view._asdict() == case["sample"]


# ---- validation guard rails ----------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_pack_rejects_missing_key(case):
    incomplete = dict(case["sample"])
    incomplete.pop(case["names"][0])  # drop one required key
    with pytest.raises(ValueError, match="missing"):
        case["group"].pack(**incomplete)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_pack_rejects_extra_key(case):
    with pytest.raises(ValueError, match="extra"):
        case["group"].pack(**case["sample"], not_a_real_var=1.0)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_pack_rejects_out_of_bounds(case):
    with pytest.raises(ValueError, match="out-of-bounds"):
        case["group"].pack(**case["oob"])


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_unpack_rejects_wrong_shape(case):
    wrong = np.zeros(case["size"] + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="expected"):
        case["group"].unpack(wrong)


# ---- exogenous open bounds (bespoke: infinities are their own failure class) ----------------------
def test_exogenous_accepts_below_zero_temperature():
    # -inf lower bound on out_temp must accept a freezing night without raising.
    EXOGENOUS.pack(rad=0.0, out_co2=7e-4, out_temp=-15.0, out_vapor=0.005)


def test_exogenous_rejects_negative_radiation():
    # rad keeps a finite lower bound of 0 even though its upper bound is +inf.
    with pytest.raises(ValueError, match="out-of-bounds"):
        EXOGENOUS.pack(rad=-1.0, out_co2=7e-4, out_temp=20.0, out_vapor=0.005)


# ---- VarGroup construction contracts (test the mechanism with throwaway vars) ---------------------
class _A(StateVar):
    symbol = "a"
    name = "a"
    unit = "u"
    lo = 0.0
    hi = 1.0
    default = 0.0


class _B(StateVar):
    symbol = "b"
    name = "b"
    unit = "u"
    lo = 0.0
    hi = 1.0
    default = 0.0


class _DupA(StateVar):
    symbol = "a2"
    name = "a"  # same name as _A on purpose
    unit = "u"
    lo = 0.0
    hi = 1.0
    default = 0.0


class _TwoFieldView(NamedTuple):
    a: float
    b: float


def test_vargroup_rejects_view_count_mismatch():
    # view has 2 fields but only 1 var is given.
    with pytest.raises(ValueError, match="fields"):
        VarGroup(_A, view=_TwoFieldView)


def test_vargroup_rejects_duplicate_name():
    with pytest.raises(ValueError, match="duplicate"):
        VarGroup(_A, _DupA, view=_TwoFieldView)


def test_attribute_array_gathers_in_declaration_order():
    du = CONTROL.attribute_array("du_max")
    assert du.dtype == np.float64
    np.testing.assert_array_equal(du, [0.12, 0.75, 15.0])


def test_attribute_array_raises_when_the_attribute_is_missing():
    """Exogenous variables have no `default`; asking must fail clearly, not obscurely."""
    with pytest.raises(AttributeError, match="not defined on every variable"):
        EXOGENOUS.attribute_array("default")


def test_units_are_strings_in_declaration_order():
    from lettuce_greenhouse_gym.variables.controls import CONTROL
    from lettuce_greenhouse_gym.variables.observables import OBSERVABLE

    assert CONTROL.units == ("mg/m2/s", "mm/s", "W/m2")
    assert OBSERVABLE.units == ("kg/m2", "ppm", "degC", "%")
