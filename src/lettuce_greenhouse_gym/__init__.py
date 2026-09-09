"""lettuce-greenhouse-gym: the van Henten lettuce greenhouse as a Gymnasium environment.

Everything a study needs is importable from the top level, grouped by the question it answers:

* *what to run*: :class:`LettuceGreenhouseEnv`, :class:`EnvConfig`, :class:`ActionMode`
* *what to optimise*: :class:`Reward`, :class:`EconomicReward`, :class:`RewardConfig`
* *what the world is*: :class:`DynamicsModel`, :class:`VanHentenLettuce`, the parameter
  registries and providers, :class:`WeatherSeries` and the samplers
* *how things are laid out*: the ``STATE``/``CONTROL``/``EXOGENOUS``/``OBSERVABLE`` groups,
  which fix vector order and bounds everywhere, and the unit conversions in :mod:`units`

``import lettuce_greenhouse_gym`` also registers ``LettuceGreenhouse-v0`` with Gymnasium.
"""

from importlib.metadata import PackageNotFoundError, version

from gymnasium.envs.registration import register

from . import baselines, units
from .config import (
    ActionMode,
    ControlOverride,
    EnvConfig,
    InitialControl,
    InitialState,
    TimestepEncoding,
)
from .envs.control_env import EnvState, LettuceGreenhouseEnv
from .model.base import DynamicsModel, SymbolicDynamicsModel
from .model.parameters import (
    CONVERSION_CONSTANTS,
    ECONOMIC_COEFFS,
    MODEL_COEFFS,
    Constant,
    FixedParameterProvider,
    ParameterProvider,
    ParameterSet,
    RandomizedParameterProvider,
)
from .model.vanhenten import VanHentenLettuce
from .rewards import (
    EconomicReward,
    PenaltyBound,
    Reward,
    RewardConfig,
    RewardContext,
    benchmark_reward_config,
    economic_stage_reward,
)
from .variables.base import VarGroup
from .variables.controls import CONTROL
from .variables.exogenous import EXOGENOUS
from .variables.observables import OBSERVABLE
from .variables.states import STATE
from .weather import (
    BENCHMARK_SCENARIO,
    BLEISWIJK_2014,
    SYNTHETIC,
    CyclingWeatherSampler,
    FixedWeatherSampler,
    RandomWeatherSampler,
    WeatherLoader,
    WeatherPerturbation,
    WeatherRepository,
    WeatherSampler,
    WeatherScenario,
    WeatherSeries,
    bleiswijk_2014,
    default_repository,
    split_start_days,
)
from .wrappers import ParameterObservation

try:
    __version__ = version("lettuce-greenhouse-gym")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

# No max_episode_steps: the season ends in harvest, which the env reports as `terminated`, so a
# TimeLimit wrapper would mislabel a terminal state as a truncation.
register(
    id="LettuceGreenhouse-v0",
    entry_point="lettuce_greenhouse_gym.envs.control_env:LettuceGreenhouseEnv",
)

__all__ = [
    "BENCHMARK_SCENARIO",
    "BLEISWIJK_2014",
    "CONTROL",
    "CONVERSION_CONSTANTS",
    "ECONOMIC_COEFFS",
    "EXOGENOUS",
    "MODEL_COEFFS",
    "OBSERVABLE",
    "STATE",
    "SYNTHETIC",
    "ActionMode",
    "Constant",
    "ControlOverride",
    "CyclingWeatherSampler",
    "DynamicsModel",
    "EconomicReward",
    "EnvConfig",
    "EnvState",
    "FixedParameterProvider",
    "FixedWeatherSampler",
    "InitialControl",
    "InitialState",
    "LettuceGreenhouseEnv",
    "ParameterObservation",
    "ParameterProvider",
    "ParameterSet",
    "PenaltyBound",
    "RandomWeatherSampler",
    "RandomizedParameterProvider",
    "Reward",
    "RewardConfig",
    "RewardContext",
    "SymbolicDynamicsModel",
    "TimestepEncoding",
    "VanHentenLettuce",
    "VarGroup",
    "WeatherLoader",
    "WeatherPerturbation",
    "WeatherRepository",
    "WeatherSampler",
    "WeatherScenario",
    "WeatherSeries",
    "__version__",
    "baselines",
    "benchmark_reward_config",
    "bleiswijk_2014",
    "default_repository",
    "economic_stage_reward",
    "split_start_days",
    "units",
]
