"""Parameter providers: supply the parameter set for an episode (the context / randomisation seam).

A provider decides *which greenhouse the model is* for one episode. ``sample`` receives the RNG from
the caller (so the env owns seeding and every episode is reproducible from one seed) and returns a
:class:`ParameterSet`.
"""

from abc import ABC, abstractmethod

import numpy as np

from .base import ParameterSet


class ParameterProvider(ABC):
    """Decides which greenhouse the model is, at the two points where that can change.

    The env calls :meth:`sample` once at ``reset`` and :meth:`step` once per transition, *before* the
    observation that precedes that transition is emitted. What happens there is the study's business
    (domain randomisation, i.i.d. parameter noise, a drift, a fault on day 20); the env guarantees
    that a returned set drives the transition it was drawn for and is what ``info["params"]``,
    ``env.parameters`` and a context-observing wrapper report ahead of it. Two reference schemes
    ship below.
    """

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> ParameterSet:
        """The parameter set an episode starts with."""

    def step(
        self, rng: np.random.Generator, step_index: int, current: ParameterSet
    ) -> ParameterSet | None:
        """The set for transition ``step_index``, or ``None`` to keep ``current``.

        Called after :meth:`sample` at reset (``step_index`` 0) and after every transition for the
        next one, so the value is known before the action that meets it. The default keeps the
        episode's parameters fixed. Override for time-varying behaviour.
        """
        return None


class FixedParameterProvider(ParameterProvider):
    """Always returns the same nominal set (no randomization)."""

    def __init__(self, base: ParameterSet) -> None:
        self._base = base

    def sample(self, rng: np.random.Generator) -> ParameterSet:
        return self._base  # ParameterSet is immutable, so returning it directly is safe


class RandomizedParameterProvider(ParameterProvider):
    """Draws a chosen subset of parameters uniformly from per-parameter ranges each episode.

    ``ranges`` maps a parameter name to an absolute ``(lo, hi)`` sampled uniformly; parameters absent
    from it stay at their nominal value. Each range must lie within the constant's physical bounds —
    a range that pokes outside them is a configuration error and is rejected at construction, so a
    drawn value can never be invalid.
    """

    def __init__(
        self, base: ParameterSet, ranges: dict[str, tuple[float, float]], *, per_step: bool = False
    ) -> None:
        for name, (lo, hi) in ranges.items():
            if name not in base.names:
                raise ValueError(f"unknown parameter {name!r}")
            if lo > hi:
                raise ValueError(f"{name}: range lo {lo} exceeds hi {hi}")
            constant = base.constants[base.idx(name)]
            if constant.lo is not None and lo < constant.lo:
                raise ValueError(f"{name}: range lo {lo} below physical bound {constant.lo}")
            if constant.hi is not None and hi > constant.hi:
                raise ValueError(f"{name}: range hi {hi} above physical bound {constant.hi}")
        self._base = base
        self._ranges = ranges
        self.per_step = per_step  # True: redraw before every transition (i.i.d. parameter noise)

    @classmethod
    def relative(
        cls,
        base: ParameterSet,
        half_width: float,
        names: tuple[str, ...] | None = None,
        *,
        per_step: bool = False,
    ) -> "RandomizedParameterProvider":
        """Every named coefficient (default: all) drawn as ``p * (1 + U(-half_width, half_width))``.

        The benchmark literature's perturbation level δ draws ``U(-δ/2, δ/2)``, so pass
        ``half_width = δ / 2`` to match it.
        """
        if not 0 < half_width < 1:
            raise ValueError(f"half_width must be in (0, 1), got {half_width}")
        chosen = base.names if names is None else tuple(names)
        ranges = {}
        for name in chosen:
            p = base.get(name)
            lo, hi = sorted((p * (1 - half_width), p * (1 + half_width)))  # p may be negative
            ranges[name] = (lo, hi)
        return cls(base, ranges, per_step=per_step)

    def sample(self, rng: np.random.Generator) -> ParameterSet:
        # ranges are validated in __init__ to lie within physical bounds, so no clamp is needed;
        # override() remains the final backstop if an invalid value ever reached here.
        drawn = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in self._ranges.items()}
        return self._base.override(**drawn)

    def step(
        self, rng: np.random.Generator, step_index: int, current: ParameterSet
    ) -> ParameterSet | None:
        return self.sample(rng) if self.per_step else None
