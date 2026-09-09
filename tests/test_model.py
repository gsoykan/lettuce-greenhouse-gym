"""The dynamics-model seam: what the env needs from a model, and what the MPC additionally needs.

Two toy models stand in for the ones a research project would bring. ``NumericTwin`` is the van
Henten integrator hidden behind plain Python callables, the shape of a learned, non-symbolic
model, and must reproduce the default env exactly. ``VanHentenWithGain`` is a symbolic subclass
with one extra coefficient, the shape of a hybrid or re-parameterised model.
"""

import casadi
import numpy as np
import pytest

from lettuce_greenhouse_gym import (
    MODEL_COEFFS,
    Constant,
    EnvConfig,
    LettuceGreenhouseEnv,
    ParameterSet,
    SymbolicDynamicsModel,
    VanHentenLettuce,
)
from lettuce_greenhouse_gym.baselines import AllOff, NominalMPC, run_episode
from lettuce_greenhouse_gym.model.base import DynamicsModel, is_symbolic
from lettuce_greenhouse_gym.variables.controls import CONTROL
from lettuce_greenhouse_gym.variables.exogenous import EXOGENOUS
from lettuce_greenhouse_gym.variables.observables import OBSERVABLE
from lettuce_greenhouse_gym.variables.states import STATE

SHORT = EnvConfig(episode_days=1.0)


class NumericTwin(DynamicsModel):
    """The reference dynamics behind plain callables: no CasADi visible to the env."""

    states, controls, exogenous, observables = STATE, CONTROL, EXOGENOUS, OBSERVABLE

    def __init__(self) -> None:
        self._reference = VanHentenLettuce()
        super().__init__(self._reference.constants)

    def build_integrator(self, dt):
        F = self._reference.build_integrator(dt)

        def step(x, u, v, c):
            return np.asarray(F(x, u, v, c)).ravel()

        return step

    def build_measurement(self):
        g = self._reference.build_measurement()
        return lambda x: np.asarray(g(x)).ravel()


class WrongWidth(NumericTwin):
    def build_integrator(self, dt):
        return lambda x, u, v, c: np.zeros(3)


class VanHentenWithGain(VanHentenLettuce):
    """Symbolic subclass with one extra, named coefficient scaling the growth balance."""

    def __init__(self, gain: float = 1.0) -> None:
        constants = ParameterSet.from_defaults(
            (
                *MODEL_COEFFS,
                Constant("k_g", "growth_gain", "-", gain, 0.0, 10.0, "test coefficient"),
            )
        )
        SymbolicDynamicsModel.__init__(self, constants)

    def rhs(self, x, u, v, c, t):
        dxdt = super().rhs(x, u, v, c, t)
        scale = casadi.SX.ones(self.states.size)
        scale[self.states.idx("dry_weight")] = c[self.constants.idx("growth_gain")]
        return dxdt * scale


def _episode(model, seed=0, controller=None):
    env = LettuceGreenhouseEnv(SHORT, model=model)
    return run_episode(env, controller or AllOff(), seed=seed)


def test_numeric_model_reproduces_the_default_env_exactly():
    reference = _episode(VanHentenLettuce())
    numeric = _episode(NumericTwin())
    np.testing.assert_array_equal(numeric.x, reference.x)
    np.testing.assert_array_equal(numeric.y, reference.y)
    assert numeric.total_return == reference.total_return


def test_numeric_model_exposes_plain_callables_and_the_mpc_says_so():
    env = LettuceGreenhouseEnv(SHORT, model=NumericTwin())
    assert not is_symbolic(env.integrator)
    assert is_symbolic(LettuceGreenhouseEnv(SHORT).integrator)
    env.reset(seed=0)
    with pytest.raises(TypeError, match="SymbolicDynamicsModel"):
        NominalMPC().reset(env)


def test_wrong_output_width_fails_at_construction_naming_the_model():
    with pytest.raises(ValueError, match=r"WrongWidth\.build_integrator returned 3 values"):
        LettuceGreenhouseEnv(SHORT, model=WrongWidth())


def test_symbolic_subclass_with_an_extra_coefficient_flows_by_name():
    unity = _episode(VanHentenWithGain(1.0))
    reference = _episode(VanHentenLettuce())
    np.testing.assert_allclose(unity.x, reference.x, rtol=1e-12, atol=0)

    env = LettuceGreenhouseEnv(SHORT, model=VanHentenWithGain(2.0))
    _, info = env.reset(seed=0)
    assert info["params"].size == len(MODEL_COEFFS) + 1
    doubled = run_episode(env, AllOff(), seed=0)
    growth_ref = np.diff(reference.x[:, STATE.idx("dry_weight")])
    growth_2x = np.diff(doubled.x[:, STATE.idx("dry_weight")])
    assert np.all(np.abs(growth_2x) >= np.abs(growth_ref) - 1e-15)
    assert not np.allclose(growth_2x, growth_ref)


def test_symbolic_subclass_plans_with_the_mpc():
    env = LettuceGreenhouseEnv(SHORT, model=VanHentenWithGain(1.5))
    log = run_episode(env, NominalMPC(), seed=0)
    assert np.isfinite(log.total_return)
    assert log.u.shape[0] == SHORT.n_steps


def test_abstract_layers_require_the_right_methods():
    class Bare(DynamicsModel):
        states, controls, exogenous, observables = STATE, CONTROL, EXOGENOUS, OBSERVABLE

    with pytest.raises(TypeError, match="build_integrator"):
        Bare(VanHentenLettuce().constants)

    class OnlyRhs(SymbolicDynamicsModel):
        states, controls, exogenous, observables = STATE, CONTROL, EXOGENOUS, OBSERVABLE

        def rhs(self, x, u, v, c, t):
            return x

    with pytest.raises(TypeError, match="measurement"):
        OnlyRhs(VanHentenLettuce().constants)
