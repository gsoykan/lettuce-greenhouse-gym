"""The experiment dataclasses. Plain, frozen data; validation in ``__post_init__``."""

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import EnvConfig
from ..weather import BENCHMARK_SCENARIO, BLEISWIJK_2014, SYNTHETIC

PACKAGED_SOURCES = frozenset({SYNTHETIC, BLEISWIJK_2014})
ALGORITHMS = ("ppo", "sac", "ddpg", "td3")
OFF_POLICY = ("sac", "ddpg", "td3")
ACTION_NOISES = ("normal", "ornstein_uhlenbeck")

# "package.module:function", "package.module:Class.method", or "path/to/script.py:function"
_TARGET = re.compile(r"^(?P<module>[^:]+):(?P<attr>[A-Za-z_][\w.]*)$")


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Held-out start days: shuffle the viable days once with ``seed``, cut at ``train_fraction``."""

    train_fraction: float = 0.8
    seed: int = 0


@dataclass(frozen=True, slots=True)
class WeatherFileSpec:
    """A user-supplied trace in the package's CSV format (``time, Io, To, RH, Vo, CO2ppm``).

    ``epoch_day`` is the day of year of the first row (1.0 for a file starting 1 January 00:00).
    """

    path: str
    epoch_day: float

    def __post_init__(self) -> None:
        if not 1.0 <= self.epoch_day <= 367.0:
            raise ValueError(f"epoch_day must be a day of year in [1, 367], got {self.epoch_day}")


@dataclass(frozen=True, slots=True)
class CallableSpec:
    """A user-supplied callable, named so a YAML spec can reach code outside the package.

    ``target`` is ``package.module:function`` (importable) or ``path/to/script.py:function`` (a
    script; a relative path is taken from the YAML's directory), with attribute chains allowed.
    It is called with ``kwargs``, so ``functools.partial`` semantics apply. A ``path`` entry in
    ``kwargs`` is anchored like a file path; other entries pass through verbatim. What the callable
    must return depends on the field that holds the spec.
    """

    target: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _TARGET.match(self.target):
            raise ValueError(
                f"target must be 'module:callable' or 'script.py:callable', got {self.target!r}"
            )

    def _parts(self) -> tuple[str, str]:
        m = _TARGET.match(self.target)
        if m is None:  # unreachable after __post_init__, kept for the checker and for safety
            raise ValueError(f"malformed target {self.target!r}")
        return m["module"], m["attr"]

    @property
    def module(self) -> str:
        return self._parts()[0]

    @property
    def attribute(self) -> str:
        return self._parts()[1]

    @property
    def is_script(self) -> bool:
        return self.module.endswith(".py")


@dataclass(frozen=True, slots=True)
class WeatherLoaderSpec(CallableSpec):
    """A weather source in any format: the callable returns a ``WeatherSeries``, e.g.
    ``{target: "my_weather:load_epw", kwargs: {path: amsterdam.epw}}``."""


@dataclass(frozen=True, slots=True)
class WeatherSpec:
    """Which trace(s), and how start days are chosen. Exactly one of the three day modes is set.

    ``source`` names one or several traces: the packaged ones (``bleiswijk_2014``, ``synthetic``),
    keys of ``files`` (CSV traces) or keys of ``loaders`` (any source). With several sources,
    training draws a source and a start day per episode. ``eval_source`` / ``eval_start_days`` let
    evaluation run on other traces or days; unset, evaluation reuses the training ones.
    """

    source: str | tuple[str, ...] = BLEISWIJK_2014
    start_day: float | None = BENCHMARK_SCENARIO.start_day
    start_days: tuple[float, ...] | None = None
    split: SplitSpec | None = None
    files: dict[str, WeatherFileSpec] = field(default_factory=dict)
    eval_source: str | tuple[str, ...] | None = None
    eval_start_days: tuple[float, ...] | None = None
    loaders: dict[str, WeatherLoaderSpec] = field(default_factory=dict)
    perturbation: CallableSpec | None = None
    """A ``WeatherPerturbation`` for the env: the callable, given ``kwargs``, returns one."""
    sampler: CallableSpec | None = None
    """A ``WeatherSampler`` for training, built by the callable from ``kwargs``, replacing the
    start-day modes above, for scenarios that pair particular traces with particular days. The
    traces it names must still be packaged, in ``files`` or in ``loaders``."""
    eval_sampler: CallableSpec | None = None
    """A ``WeatherSampler`` for evaluation; unset, evaluation uses the start-day modes if one is
    set, otherwise a fresh instance of ``sampler``."""

    def __post_init__(self) -> None:
        modes = [m is not None for m in (self.start_day, self.start_days, self.split)]
        if self.sampler is None and sum(modes) != 1:
            raise ValueError("set exactly one of start_day, start_days, split")
        if self.sampler is not None and sum(modes) > 1:
            raise ValueError("with a sampler, at most one of start_day, start_days, split")
        if self.eval_sampler is not None and self.sampler is None:
            raise ValueError("eval_sampler needs a sampler")
        if not self.sources:
            raise ValueError("source must name at least one weather trace")
        clashes = PACKAGED_SOURCES & (self.files.keys() | self.loaders.keys())
        if clashes:
            raise ValueError(
                f"files/loaders may not reuse packaged source names: {sorted(clashes)}"
            )
        twice = self.files.keys() & self.loaders.keys()
        if twice:
            raise ValueError(f"source names defined under both files and loaders: {sorted(twice)}")
        known = PACKAGED_SOURCES | self.files.keys() | self.loaders.keys()
        unknown = [s for s in self.sources + self.eval_sources if s not in known]
        if unknown:
            raise ValueError(
                f"unknown weather source(s) {unknown}; list them under files or loaders"
            )
        if self.split is not None and (
            len(self.sources) > 1
            or self.eval_source is not None
            or self.eval_start_days is not None
        ):
            raise ValueError("split works on a single source and already defines the eval days")
        if self.eval_start_days is not None and not self.eval_start_days:
            raise ValueError("eval_start_days must not be empty")

    @property
    def sources(self) -> tuple[str, ...]:
        return (self.source,) if isinstance(self.source, str) else tuple(self.source)

    @property
    def eval_sources(self) -> tuple[str, ...]:
        if self.eval_source is None:
            return self.sources
        return (
            (self.eval_source,) if isinstance(self.eval_source, str) else tuple(self.eval_source)
        )


@dataclass(frozen=True, slots=True)
class WandbSpec:
    project: str
    entity: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainSpec:
    """Algorithm and budget. ``algo_kwargs`` go to the SB3 constructor unchanged."""

    algo: str = "ppo"
    # any SB3 policy name; a custom network goes through algo_kwargs["policy_kwargs"]
    policy: str = "MlpPolicy"
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    seed: int = 0
    normalize_obs: bool = True
    net_arch: tuple[int, ...] | None = None
    algo_kwargs: dict[str, Any] = field(default_factory=dict)
    device: str = "auto"
    log_dir: str = "runs"
    run_name: str | None = None
    wandb: WandbSpec | None = None
    # evaluate on the *eval* weather (held-out days under a split) every this many env steps;
    # the best model by evaluation return is kept in <run>/best/
    eval_every: int | None = None
    n_eval_episodes: int = 1
    # save <run>/checkpoints/ every this many env steps
    checkpoint_every: int | None = None
    # exploration noise on the action (off-policy algorithms), as a fraction of the [-1, 1] range
    action_noise_sigma: float | None = None
    action_noise: str = "normal"  # or "ornstein_uhlenbeck"

    def __post_init__(self) -> None:
        if self.algo not in ALGORITHMS:
            raise ValueError(f"algo must be one of {ALGORITHMS}, got {self.algo!r}")
        if self.total_timesteps < 1 or self.n_envs < 1:
            raise ValueError("total_timesteps and n_envs must be positive")
        for name in ("eval_every", "checkpoint_every"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive or None, got {value}")
        if self.n_eval_episodes < 1:
            raise ValueError("n_eval_episodes must be positive")
        if self.action_noise not in ACTION_NOISES:
            raise ValueError(
                f"action_noise must be one of {ACTION_NOISES}, got {self.action_noise!r}"
            )
        if self.action_noise_sigma is not None:
            if self.algo not in OFF_POLICY:
                raise ValueError(f"action_noise_sigma applies to {OFF_POLICY} only")
            if self.action_noise_sigma <= 0:
                raise ValueError("action_noise_sigma must be positive")


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """How the dynamics coefficients vary. Everything empty means the nominal model.

    Exactly one scheme may be set: ``ranges`` (absolute per-name ``(lo, hi)``, drawn uniformly) or
    ``relative`` (every coefficient, or those in ``names``, times ``1 + U(-relative, relative)``).
    ``per_step`` redraws before every transition (parameter noise) instead of once per episode
    (domain randomisation).
    """

    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    relative: float | None = None
    names: tuple[str, ...] | None = None
    per_step: bool = False
    provider: CallableSpec | None = None
    """Any other scheme: the callable is called as ``target(constants, **kwargs)`` with the model's
    nominal ``ParameterSet`` and returns a ``ParameterProvider``. Exclusive with the two above."""

    def __post_init__(self) -> None:
        if self.provider is not None and (self.ranges or self.relative is not None):
            raise ValueError("`provider` replaces `ranges` / `relative`; set one scheme only")
        if self.ranges and self.relative is not None:
            raise ValueError("set either `ranges` or `relative`, not both")
        if self.relative is not None and not 0 < self.relative < 1:
            raise ValueError(f"relative must be in (0, 1), got {self.relative}")
        if self.names is not None and self.relative is None:
            raise ValueError("`names` restricts `relative`; set `relative` too")

    @property
    def is_nominal(self) -> bool:
        return not self.ranges and self.relative is None and self.provider is None


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """The complete description of a run. The defaults are the published benchmark."""

    env: EnvConfig = field(default_factory=EnvConfig)
    weather: WeatherSpec = field(default_factory=WeatherSpec)
    parameters: ParameterSpec = field(default_factory=ParameterSpec)
    train: TrainSpec = field(default_factory=TrainSpec)

    def __post_init__(self) -> None:
        for name, cls in (
            ("env", EnvConfig),
            ("weather", WeatherSpec),
            ("parameters", ParameterSpec),
            ("train", TrainSpec),
        ):
            value = getattr(self, name)
            if not isinstance(value, cls):
                raise TypeError(f"{name} must be a {cls.__name__}, got {type(value).__name__}")
