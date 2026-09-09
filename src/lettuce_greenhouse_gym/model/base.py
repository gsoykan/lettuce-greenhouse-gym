"""The dynamics-model contract, in two layers.

:class:`DynamicsModel` is what the environment needs from any model of the greenhouse: the variable
groups, the coefficient set, and two callables, a one-step integrator and a measurement map. It
says nothing about how the step is computed, so a learned or purely numeric model implements it
directly.

:class:`SymbolicDynamicsModel` is the layer the reference model lives on: continuous-time dynamics
and a measurement written as CasADi expressions, compiled here into ``casadi.Function`` objects.
Those are also what a gradient-based planner such as ``NominalMPC`` needs, because it differentiates
through the step; a numeric model runs in the environment and with sampling-based planners, not
with that controller.

Both layers predict the same thing: the environment's own states, controls, weather and observables.
The reward and the comfort bands index those states by name, so a model behind this seam predicts
physical quantities, not a latent code.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import casadi
import numpy as np
from numpy.typing import ArrayLike

from lettuce_greenhouse_gym.model.parameters import ParameterSet
from lettuce_greenhouse_gym.variables.base import VarGroup

Integrator = Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Any]
"""``F(x, u, v, c) -> x_next``: one control step, positional arguments, an array-like of
``states.size`` values (a ``casadi.DM`` or a NumPy array; the environment calls ``np.asarray``).
The result is expected to lie within ``states.bounds()``."""

Measurement = Callable[[ArrayLike], Any]
"""``g(x) -> y``: the state as the grower sees it, an array-like of ``observables.size`` values."""


class DynamicsModel(ABC):
    """What the environment requires of a model. Subclass this for a numeric or learned model."""

    states: VarGroup
    controls: VarGroup
    exogenous: VarGroup
    constants: ParameterSet
    observables: VarGroup

    def __init__(self, constants: ParameterSet) -> None:
        self.constants = constants

    @abstractmethod
    def build_integrator(self, dt: float) -> Integrator:
        """The one-step map for a control step of ``dt`` seconds; see :data:`Integrator`.

        Built once per environment. ``c`` is passed on every call so a randomised coefficient set
        never forces a rebuild; a model without coefficients simply ignores it.
        """

    @abstractmethod
    def build_measurement(self) -> Measurement:
        """The measurement map; see :data:`Measurement`."""

    def check(self, integrator: Integrator, measurement: Measurement) -> None:
        """Run the built callables once on nominal inputs and verify their output widths.

        The environment calls this at construction so a wrong-shaped model fails there, with the
        model named, rather than at the first ``step``.
        """
        x = self.states.attribute_array("default")
        u = self.controls.attribute_array("default")
        lo, hi = self.exogenous.bounds()
        v = np.maximum(lo, 0.0)  # any admissible weather will do for a shape check
        finite = np.isfinite(lo) & np.isfinite(hi)
        v[finite] = (lo[finite] + hi[finite]) / 2
        c = self.constants.to_array()
        name = type(self).__name__
        x_next = np.asarray(integrator(x, u, v, c), dtype=np.float64).ravel()
        if x_next.size != self.states.size or not np.all(np.isfinite(x_next)):
            raise ValueError(
                f"{name}.build_integrator returned {x_next.size} values ({x_next}); expected "
                f"{self.states.size} finite states"
            )
        y = np.asarray(measurement(x), dtype=np.float64).ravel()
        if y.size != self.observables.size or not np.all(np.isfinite(y)):
            raise ValueError(
                f"{name}.build_measurement returned {y.size} values ({y}); expected "
                f"{self.observables.size} finite observables"
            )


class SymbolicDynamicsModel(DynamicsModel):
    """Continuous-time dynamics ``dx/dt = f(x, u, v, c, t)`` as CasADi expressions.

    Subclasses write :meth:`rhs` and :meth:`measurement`; this class compiles them (RK4 with four
    finite elements, then a clamp to the state bounds) into ``casadi.Function`` objects, which is
    what makes the model usable inside an optimiser as well as in the environment.
    """

    @abstractmethod
    def rhs(self, x, u, v, c, t):
        """Continuous dynamics as a CasADi column expression (vertcat of state derivatives).
        Array-based: index x/u/v/c through the group index enums, never magic integers."""

    @abstractmethod
    def measurement(self, x):
        """Map a state vector to grower-facing observables y (CO2 ppm, RH%, ...). CasADi expr."""

    def build_measurement(self) -> casadi.Function:
        """Compile the measurement map into ``g(x) -> y``, checking its width."""
        x = casadi.SX.sym("x", self.states.size)
        y = self.measurement(x)
        if y.numel() != self.observables.size:
            raise ValueError(
                f"measurement returned {y.numel()} outputs but "
                f"{self.observables.size} observables are declared"
            )
        return casadi.Function("g", [x], [y], ["x"], ["y"])

    def build_integrator(self, dt: float) -> casadi.Function:
        x = casadi.SX.sym("x", self.states.size)
        u = casadi.SX.sym("u", self.controls.size)
        v = casadi.SX.sym("v", self.exogenous.size)
        c = casadi.SX.sym("c", self.constants.size)

        dxdt = self.rhs(x, u, v, c, t=0.0)

        par = casadi.vertcat(v, c)

        opts = {"simplify": True, "number_of_finite_elements": 4}
        integ = casadi.integrator(
            "integ", "rk", {"x": x, "u": u, "p": par, "ode": dxdt}, 0.0, dt, opts
        )

        x_next = integ(x0=x, u=u, p=par)["xf"]
        lo, hi = self.states.bounds()
        x_next = casadi.fmin(casadi.fmax(x_next, casadi.DM(lo)), casadi.DM(hi))

        F = casadi.Function("F", [x, u, v, c], [x_next], ["x", "u", "v", "c"], ["x_next"])

        return F


def is_symbolic(fn: Integrator | Measurement) -> bool:
    """Whether a compiled callable is a CasADi ``Function``, i.e. usable on symbolic arguments."""
    return isinstance(fn, casadi.Function)


__all__ = [
    "DynamicsModel",
    "Integrator",
    "Measurement",
    "SymbolicDynamicsModel",
    "is_symbolic",
]
