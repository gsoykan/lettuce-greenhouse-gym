"""Parameter machinery: the :class:`Constant` definition and the :class:`ParameterSet` container.

A :class:`Constant` is the immutable *definition* of one coefficient (identity, unit, nominal value,
optional bounds) — never a bare float. Registries of these definitions live in sibling modules
(``model_coeffs``, ``conversion``, ``economic``); the *current* values for a given episode live in a
:class:`ParameterSet` built on top of one such registry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Constant:
    """Immutable definition of a single model coefficient.

    ``default`` is the nominal value the coefficient ships with; ``lo``/``hi`` are optional bounds
    (``None`` = unbounded) used when a provider randomizes the parameter. The object carries the
    coefficient's identity and unit so it is never reduced to an anonymous float.
    """

    symbol: str
    name: str
    unit: str
    default: float
    lo: float | None = None
    hi: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        # raise (not assert): a definition-time integrity check must survive `python -O`.
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError(f"{self.name}: lo {self.lo} exceeds hi {self.hi}")
        if self.lo is not None and self.default < self.lo:
            raise ValueError(f"{self.name}: default {self.default} below lo {self.lo}")
        if self.hi is not None and self.default > self.hi:
            raise ValueError(f"{self.name}: default {self.default} above hi {self.hi}")


@dataclass(frozen=True, slots=True)
class ParameterSet:
    """A registry of :class:`Constant` definitions plus the current value of each coefficient.

    ``constants`` are the shared, fixed definitions; ``values`` holds this set's current numbers in
    the same order (defaulting to each constant's ``default`` via :meth:`from_defaults`). The set is
    immutable — :meth:`override` returns a *new* set rather than mutating in place — and every set is
    validated at construction, so an out-of-bounds or malformed set can never exist.
    """

    constants: tuple[Constant, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.constants):
            raise ValueError(
                f"values has {len(self.values)} entries but {len(self.constants)} constants given"
            )
        seen: set[str] = set()
        for c, v in zip(self.constants, self.values, strict=True):
            if c.name in seen:
                raise ValueError(f"duplicate constant name {c.name!r}")
            seen.add(c.name)
            if c.lo is not None and v < c.lo:
                raise ValueError(f"{c.name}: value {v} below lo {c.lo}")
            if c.hi is not None and v > c.hi:
                raise ValueError(f"{c.name}: value {v} above hi {c.hi}")

    @classmethod
    def from_defaults(cls, constants: tuple[Constant, ...]) -> ParameterSet:
        """Build a set whose values are each constant's nominal ``default``."""
        return cls(constants, tuple(c.default for c in constants))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.constants)

    @property
    def size(self) -> int:
        return len(self.constants)

    def idx(self, name: str) -> int:
        """Position of a coefficient by name."""
        for i, c in enumerate(self.constants):
            if c.name == name:
                return i
        raise ValueError(f"unknown parameter {name!r}")

    def get(self, name: str) -> float:
        """Current value of a coefficient by name."""
        return self.values[self.idx(name)]

    def to_array(self) -> NDArray[np.float64]:
        """The ordered ``c`` vector the dynamics consume."""
        return np.array(self.values, dtype=np.float64)

    def override(self, **by_name: float) -> ParameterSet:
        """Return a new set with some values replaced; unknown names or out-of-bounds values raise."""
        unknown = by_name.keys() - set(self.names)
        if unknown:
            raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
        new_values = list(self.values)
        for name, value in by_name.items():
            new_values[self.idx(name)] = value
        # a fresh set re-runs __post_init__, so bound violations raise here.
        return ParameterSet(self.constants, tuple(new_values))
