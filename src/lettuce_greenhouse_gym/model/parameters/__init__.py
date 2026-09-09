"""Model parameters: the constant coefficients of the dynamics, measurement, and reward.

The registries are kept apart because they have different consumers and lifecycles:
``MODEL_COEFFS`` (dynamics ``f``, randomized by a provider), ``CONVERSION_CONSTANTS`` (measurement
``g``, never randomized), and ``ECONOMIC_COEFFS`` (reward). Each is an ordered tuple of
:class:`Constant` definitions; wrap one in a :class:`ParameterSet` to attach current values.
"""

from .base import Constant, ParameterSet
from .conversion import CONVERSION_CONSTANTS
from .economic import ECONOMIC_COEFFS
from .model_coeffs import MODEL_COEFFS
from .providers import (
    FixedParameterProvider,
    ParameterProvider,
    RandomizedParameterProvider,
)

__all__ = [
    "CONVERSION_CONSTANTS",
    "ECONOMIC_COEFFS",
    "MODEL_COEFFS",
    "Constant",
    "FixedParameterProvider",
    "ParameterProvider",
    "ParameterSet",
    "RandomizedParameterProvider",
]
