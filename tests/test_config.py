"""Tests for the environment configuration: validation, derivation, and the override plumbing."""

import numpy as np
import pytest

from lettuce_greenhouse_gym.config import (
    ActionMode,
    ControlOverride,
    EnvConfig,
    InitialControl,
    InitialState,
)
from lettuce_greenhouse_gym.variables.controls import CONTROL
from lettuce_greenhouse_gym.variables.states import STATE


def test_action_mode_members_are_plain_strings():
    """StrEnum: a dumped config says "delta", not "ActionMode.DELTA", so it round-trips."""
    assert ActionMode.DELTA == "delta"
    assert ActionMode.ABSOLUTE == "absolute"


def test_control_override_rejects_unknown_control():
    with pytest.raises(ValueError, match="unknown control"):
        ControlOverride("boiler", hi=1.0)


def test_control_override_rejects_lo_above_hi():
    with pytest.raises(ValueError, match="exceeds hi"):
        ControlOverride("heating", lo=9.0, hi=1.0)


@pytest.mark.parametrize("du_max", [0.0, -1.0], ids=["zero", "negative"])
def test_control_override_rejects_non_positive_du_max(du_max):
    with pytest.raises(ValueError, match="du_max must be positive"):
        ControlOverride("heating", du_max=du_max)


def test_control_override_rejects_setting_nothing():
    """An all-None override is a forgotten keyword, not an intent; silently ignoring it hides a bug."""
    with pytest.raises(ValueError, match="sets nothing"):
        ControlOverride("heating")


def test_initial_state_rejects_unknown_state():
    with pytest.raises(ValueError, match="unknown state"):
        InitialState("temperature", 15.0)


@pytest.mark.parametrize("value", [-1.0, 99.0], ids=["below", "above"])
def test_initial_state_rejects_value_outside_physical_bounds(value):
    with pytest.raises(ValueError, match="outside physical bounds"):
        InitialState("indoor_temp", value)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("dt", 0.0, "dt must be positive"),
        ("dt", -1.0, "dt must be positive"),
        ("episode_days", 0.0, "episode_days must be positive"),
        ("weather_window", -1, "weather_window must be non-negative"),
    ],
    ids=["dt_zero", "dt_negative", "days_zero", "window_negative"],
)
def test_env_config_rejects_invalid_scalars(field_name, value, message):
    with pytest.raises(ValueError, match=message):
        EnvConfig(**{field_name: value})


def test_env_config_rejects_duplicate_control_overrides():
    with pytest.raises(ValueError, match="duplicate control override"):
        EnvConfig(
            control_overrides=(
                ControlOverride("heating", hi=200.0),
                ControlOverride("heating", hi=250.0),
            )
        )


def test_env_config_rejects_duplicate_initial_state():
    with pytest.raises(ValueError, match="duplicate initial state"):
        EnvConfig(
            initial_state=(InitialState("dry_weight", 0.004), InitialState("dry_weight", 0.005))
        )


def test_env_config_rejects_a_partial_override_that_inverts_a_band():
    """lo=200 is fine in isolation; heating's declared hi is 150, so only resolution can see it."""
    with pytest.raises(ValueError, match="invert the bounds"):
        EnvConfig(control_overrides=(ControlOverride("heating", lo=200.0),))


@pytest.mark.parametrize(
    ("dt", "days", "expected"),
    [(1800.0, 40.0, 1920), (900.0, 40.0, 3840), (1800.0, 1.0, 48)],
    ids=["benchmark", "15min_steps", "one_day"],
)
def test_n_steps_is_derived_from_dt_and_days(dt, days, expected):
    assert EnvConfig(dt=dt, episode_days=days).n_steps == expected


def test_defaults_resolve_to_the_declared_values():
    c = EnvConfig()
    lo, hi = c.effective_control_bounds()
    np.testing.assert_array_equal(lo, CONTROL.bounds()[0])
    np.testing.assert_array_equal(hi, CONTROL.bounds()[1])
    np.testing.assert_array_equal(c.effective_du_max(), CONTROL.attribute_array("du_max"))
    np.testing.assert_array_equal(c.effective_x0(), STATE.attribute_array("default"))


def test_overrides_are_applied():
    c = EnvConfig(
        control_overrides=(ControlOverride("heating", hi=200.0, du_max=20.0),),
        initial_state=(InitialState("dry_weight", 0.005),),
    )
    i = CONTROL.idx("heating")
    assert c.effective_control_bounds()[1][i] == 200.0
    assert c.effective_du_max()[i] == 20.0
    assert c.effective_x0()[STATE.idx("dry_weight")] == 0.005


def test_an_override_leaves_the_other_entries_alone():
    c = EnvConfig(control_overrides=(ControlOverride("heating", hi=200.0),))
    j = CONTROL.idx("ventilation")
    assert c.effective_control_bounds()[1][j] == CONTROL.bounds()[1][j]


def test_resolvers_return_fresh_arrays():
    """A caller may keep and modify a resolved array without corrupting later calls."""
    c = EnvConfig()
    x0 = c.effective_x0()
    x0[0] = 999.0
    assert c.effective_x0()[0] != 999.0


def test_overrides_never_mutate_the_variable_classes():
    """The invariant the entire override design rests on.

    Overriding builds a fresh array and edits the copy; the ControlVar/StateVar classes keep their
    declared values. If this fails, two envs in one process silently share, and corrupt, bounds.
    """
    declared_hi = CONTROL.bounds()[1].copy()
    declared_x0 = STATE.attribute_array("default").copy()

    overridden = EnvConfig(
        control_overrides=(ControlOverride("heating", hi=200.0, du_max=20.0),),
        initial_state=(InitialState("dry_weight", 0.005),),
    )
    overridden.effective_control_bounds()
    overridden.effective_du_max()
    overridden.effective_x0()

    np.testing.assert_array_equal(CONTROL.bounds()[1], declared_hi)
    np.testing.assert_array_equal(STATE.attribute_array("default"), declared_x0)
    # and a fresh config still sees the declared values
    np.testing.assert_array_equal(EnvConfig().effective_control_bounds()[1], declared_hi)
    np.testing.assert_array_equal(EnvConfig().effective_x0(), declared_x0)


# ---- initial control (u0) ------------------------------------------------------------------------
def test_initial_control_rejects_unknown_control():
    with pytest.raises(ValueError, match="unknown control"):
        InitialControl("boiler", 1.0)


def test_effective_u0_defaults_to_the_declared_controls():
    np.testing.assert_array_equal(EnvConfig().effective_u0(), CONTROL.attribute_array("default"))


def test_initial_control_override_is_applied():
    c = EnvConfig(initial_control=(InitialControl("heating", 80.0),))
    assert c.effective_u0()[CONTROL.idx("heating")] == 80.0


def test_env_config_rejects_duplicate_initial_control():
    with pytest.raises(ValueError, match="duplicate initial control"):
        EnvConfig(
            initial_control=(InitialControl("heating", 10.0), InitialControl("heating", 20.0))
        )


def test_env_config_rejects_u0_outside_the_control_bounds():
    with pytest.raises(ValueError, match="initial control outside"):
        EnvConfig(initial_control=(InitialControl("heating", 999.0),))


def test_u0_is_judged_against_the_overridden_bounds_not_the_declared_ones():
    """heating=200 is illegal by default (hi=150) but legal once hi is raised, so the bounds must
    resolve before u0 is checked. Both directions are asserted: checking u0 first fails the second
    half, skipping the check entirely fails the first."""
    with pytest.raises(ValueError, match="initial control outside"):
        EnvConfig(initial_control=(InitialControl("heating", 200.0),))

    ok = EnvConfig(
        control_overrides=(ControlOverride("heating", hi=200.0),),
        initial_control=(InitialControl("heating", 200.0),),
    )
    assert ok.effective_u0()[CONTROL.idx("heating")] == 200.0


def test_initial_control_overrides_never_mutate_the_control_classes():
    declared_u0 = CONTROL.attribute_array("default").copy()
    EnvConfig(initial_control=(InitialControl("heating", 80.0),)).effective_u0()
    np.testing.assert_array_equal(CONTROL.attribute_array("default"), declared_u0)
    np.testing.assert_array_equal(EnvConfig().effective_u0(), declared_u0)
