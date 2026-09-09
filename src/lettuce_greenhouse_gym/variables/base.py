"""Typed variable system: quantities are *descriptor classes*, groups own the array layout.

A concrete variable (e.g. ``DryWeight``) is a subclass of :class:`StateVar` that sets its metadata
as class attributes — it is never instantiated. A :class:`VarGroup` orders a set of such classes and
is the single source of truth for that group's index layout, bounds, Gym space, index enum, and the
(un)packing between named values and plain ``float64`` arrays. The arrays are what the ODE ``f``
consumes; the typed views are for the boundary only.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from enum import IntEnum
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


def _index_enum(name: str, members: Mapping[str, int]) -> Any:
    """An ``IntEnum`` built at runtime from a group's variable names.

    Built through a plainly typed factory: the functional ``IntEnum(...)`` form needs literal
    arguments for a type checker to model it, and these are not literals.
    """
    factory: Callable[..., Any] = IntEnum
    return factory(name, dict(members))


class BaseVar(ABC):
    """A physical quantity. Metadata lives on the class; instances are never created."""

    symbol: ClassVar[str]  # paper symbol, e.g. "X_d"
    name: ClassVar[str]  # snake identifier, e.g. "dry_weight"
    unit: ClassVar[str]
    lo: ClassVar[float]  # physical lower bound (use -np.inf if unbounded)
    hi: ClassVar[float]  # physical upper bound (use  np.inf if unbounded)
    description: ClassVar[str] = ""


class StateVar(BaseVar):
    default: ClassVar[float]  # x0 component


class ControlVar(BaseVar):
    du_max: ClassVar[float]  # per-step rate limit
    default: ClassVar[float]  # u0 component


class ExogenousVar(BaseVar):
    """Weather / disturbance; its value is supplied by a provider at time t."""


class ObservableVar(BaseVar):
    """A grower-facing measured quantity, derived from the state by the measurement map."""


V = TypeVar("V", bound=BaseVar)
VView = TypeVar("VView", bound=tuple)  # a NamedTuple type for this group's fields


class VarGroup(Generic[V, VView]):
    """Ordered collection of variable *classes* — the single source of truth for a group's array
    layout, bounds, Gym space, index enum, and (un)packing."""

    def __init__(self, *var_types: type[V], view: type[VView]) -> None:
        fields = getattr(view, "_fields", None)
        if fields is not None and len(fields) != len(var_types):
            raise ValueError(
                f"view {view.__name__} has {len(fields)} fields but {len(var_types)} vars given"
            )
        self._vars: tuple[type[V], ...] = var_types
        self._view = view
        self._index: dict[str, int] = {}
        for i, vt in enumerate(var_types):
            if vt.name in self._index:
                raise ValueError(f"duplicate var name {vt.name!r} in group")
            self._index[vt.name] = i
        self._enum = _index_enum(  # e.g. StateView -> "StateIdx"
            f"{view.__name__.removesuffix('View')}Idx",
            {vt.name.upper(): i for i, vt in enumerate(var_types)},
        )

    @property
    def size(self) -> int:
        return len(self._vars)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(vt.name for vt in self._vars)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(vt.symbol for vt in self._vars)

    def idx(self, key: str | type[V]) -> int:
        """Index of a variable by its name or its class."""
        return self._index[key if isinstance(key, str) else key.name]

    def index_enum(self) -> Any:
        """IntEnum of positions, so the ODE can write ``x[StateIdx.DRY_WEIGHT]`` (never ``x[0]``).

        Typed ``Any`` because the members are the group's variable names, known only at runtime;
        a checker cannot verify ``StateIdx.DRY_WEIGHT`` and should not pretend to.
        """
        return self._enum

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lo = np.array([vt.lo for vt in self._vars], dtype=np.float64)
        hi = np.array([vt.hi for vt in self._vars], dtype=np.float64)
        return lo, hi

    @property
    def units(self) -> tuple[str, ...]:
        """Unit strings in declaration order, for labels."""
        return tuple(vt.unit for vt in self._vars)

    def attribute_array(self, attr: str) -> NDArray[np.float64]:
        """Gather a numeric class attribute across the group, in declaration order.

        Complements :meth:`bounds` for the per-kind attributes it cannot know about: ``default``
        on states and controls, ``du_max`` on controls. Raises if the attribute is not defined on
        every variable, so asking an exogenous group for ``default`` fails clearly.
        """
        try:
            return np.array([getattr(vt, attr) for vt in self._vars], dtype=np.float64)
        except AttributeError as exc:
            raise AttributeError(
                f"{attr!r} is not defined on every variable in this group"
            ) from exc

    def gym_space(self) -> spaces.Box:
        """This group as a Gymnasium ``Box``, for an env's action or observation space.

        Derived here so the group stays the single source of truth for bounds. ``float32`` is the
        Gymnasium convention, and ``Box`` accepts the +/-inf entries an unbounded group produces.

        Note this is one group's box; an env whose observation concatenates several (say observables
        plus the previous control) composes them itself.
        """
        lo, hi = self.bounds()
        return spaces.Box(low=lo.astype(np.float32), high=hi.astype(np.float32), dtype=np.float32)

    def pack(self, **by_name: float) -> NDArray[np.float64]:
        """Validate a complete, in-bounds set of named values -> ordered float64 array."""
        missing = set(self.names) - by_name.keys()
        extra = by_name.keys() - set(self.names)
        if missing or extra:
            raise ValueError(
                f"pack {self._view.__name__}: missing={sorted(missing)} extra={sorted(extra)}"
            )
        arr = np.array([by_name[n] for n in self.names], dtype=np.float64)
        lo, hi = self.bounds()
        oob = (arr < lo) | (arr > hi)
        if oob.any():
            bad = {self.names[i]: float(arr[i]) for i in np.flatnonzero(oob)}
            raise ValueError(f"pack {self._view.__name__}: out-of-bounds {bad}")
        return arr

    def unpack(self, arr: NDArray[np.floating]) -> VView:
        """Array -> statically-typed view (e.g. ``STATE.unpack(x).dry_weight``)."""
        if arr.shape != (self.size,):
            raise ValueError(
                f"unpack {self._view.__name__}: expected ({self.size},), got {arr.shape}"
            )
        # the view is a NamedTuple class; call it as a plain constructor, not as ``tuple(iterable)``
        view: Callable[..., VView] = self._view
        return view(*(float(v) for v in arr))
