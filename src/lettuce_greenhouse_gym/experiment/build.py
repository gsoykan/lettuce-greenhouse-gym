"""Spec -> live objects, through the package's existing constructors only."""

import importlib
import importlib.util
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..envs.control_env import LettuceGreenhouseEnv
from ..model.parameters import ParameterProvider, RandomizedParameterProvider
from ..model.vanhenten import VanHentenLettuce
from ..weather import (
    CyclingWeatherSampler,
    FixedWeatherSampler,
    RandomWeatherSampler,
    WeatherLoader,
    WeatherPerturbation,
    WeatherRepository,
    WeatherSampler,
    WeatherScenario,
    WeatherSeries,
    default_repository,
    split_start_days,
)
from .spec import CallableSpec, ExperimentSpec, WeatherLoaderSpec

Role = Literal["train", "eval"]


def resolve_target(target: str) -> Callable[..., Any]:
    """The callable a ``WeatherLoaderSpec.target`` names, imported by module path or from a script.

    Resolving happens in whichever process builds the env, so a spec with loaders works unchanged
    in subprocess vector-env workers and when a saved run is reloaded.
    """
    ls = CallableSpec(target)
    if ls.is_script:
        path = Path(ls.module).expanduser()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load weather loader script {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(ls.module)
    obj: object = module
    for part in ls.attribute.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{target} is not callable")
    return obj


def build_loader(spec: WeatherLoaderSpec) -> WeatherLoader:
    """``partial(target, **kwargs)``: the zero-argument callable a repository registers."""
    return partial(resolve_target(spec.target), **spec.kwargs)


def build_repository(
    spec: ExperimentSpec, repository: WeatherRepository | None = None
) -> WeatherRepository:
    """The packaged sources plus every file and loader the spec lists, registered lazily by name."""
    repo = repository if repository is not None else default_repository()
    for name, f in spec.weather.files.items():
        if name not in repo.names:
            repo.register(
                name, partial(WeatherSeries.from_csv, Path(f.path).expanduser(), f.epoch_day, name)
            )
    for name, ls in spec.weather.loaders.items():
        if name not in repo.names:
            repo.register(name, build_loader(ls))
    return repo


def _split_days(spec: ExperimentSpec, repo: WeatherRepository) -> tuple[tuple, tuple]:
    w = spec.weather
    if w.split is None:
        raise ValueError("the weather spec is not in split mode")
    train_days, eval_days = split_start_days(
        repo.load(w.sources[0]),
        spec.env.n_steps,
        spec.env.dt,
        np.random.default_rng(w.split.seed),
        w.split.train_fraction,
        lookahead=spec.env.weather_window,
    )
    return tuple(train_days), tuple(eval_days)


def _custom_sampler(cs: CallableSpec) -> WeatherSampler:
    sampler = resolve_target(cs.target)(**cs.kwargs)
    if not isinstance(sampler, WeatherSampler):
        raise TypeError(
            f"weather sampler {cs.target} returned {type(sampler).__name__}, not a WeatherSampler"
        )
    return sampler


def build_sampler(
    spec: ExperimentSpec, role: Role = "train", repository: WeatherRepository | None = None
) -> WeatherSampler:
    """Training draws a source and a start day at random; evaluation cycles through them in order.

    A ``sampler`` / ``eval_sampler`` callable spec replaces that logic for its role.
    """
    w = spec.weather
    repo = build_repository(spec, repository)
    if role == "train" and w.sampler is not None:
        return _custom_sampler(w.sampler)
    if role == "eval" and w.eval_sampler is not None:
        return _custom_sampler(w.eval_sampler)
    no_day_mode = w.start_day is None and w.start_days is None and w.split is None
    if role == "eval" and no_day_mode and w.sampler is not None:
        return _custom_sampler(w.sampler)
    if role == "train":
        sources = w.sources
        if w.start_day is not None:
            days: tuple[float, ...] = (w.start_day,)
        elif w.start_days is not None:
            days = w.start_days
        else:
            days, _ = _split_days(spec, repo)
        if len(sources) == 1 and len(days) == 1:
            return FixedWeatherSampler(WeatherScenario(sources[0], days[0]))
        return RandomWeatherSampler(sources, days)
    sources = w.eval_sources
    if w.eval_start_days is not None:
        days = w.eval_start_days
    elif w.start_day is not None:
        days = (w.start_day,)
    elif w.start_days is not None:
        days = w.start_days
    else:
        _, days = _split_days(spec, repo)
    if len(sources) == 1 and len(days) == 1:
        return FixedWeatherSampler(WeatherScenario(sources[0], days[0]))
    return CyclingWeatherSampler(sources, days)


def build_provider(spec: ExperimentSpec, model: VanHentenLettuce) -> ParameterProvider | None:
    """``None`` means the env's default: fixed nominal coefficients."""
    p = spec.parameters
    if p.is_nominal:
        return None
    if p.provider is not None:
        provider = resolve_target(p.provider.target)(model.constants, **p.provider.kwargs)
        if not isinstance(provider, ParameterProvider):
            raise TypeError(
                f"parameters.provider {p.provider.target} returned {type(provider).__name__}, "
                "not a ParameterProvider"
            )
        return provider
    if p.relative is not None:
        return RandomizedParameterProvider.relative(
            model.constants, p.relative, p.names, per_step=p.per_step
        )
    return RandomizedParameterProvider(model.constants, dict(p.ranges), per_step=p.per_step)


def build_env(
    spec: ExperimentSpec, role: Role = "train", repository: WeatherRepository | None = None
) -> LettuceGreenhouseEnv:
    """The environment a spec describes. ``build_env(ExperimentSpec())`` is the benchmark env."""
    model = VanHentenLettuce()
    repo = build_repository(spec, repository)
    return LettuceGreenhouseEnv(
        spec.env,
        model=model,
        parameter_provider=build_provider(spec, model),
        weather_sampler=build_sampler(spec, role, repo),
        weather_repository=repo,
        weather_perturbation=build_perturbation(spec),
    )


def build_perturbation(spec: ExperimentSpec) -> WeatherPerturbation | None:
    """The ``WeatherPerturbation`` a spec names, or ``None`` for the trace as measured."""
    ps = spec.weather.perturbation
    if ps is None:
        return None
    perturbation = resolve_target(ps.target)(**ps.kwargs)
    if not isinstance(perturbation, WeatherPerturbation):
        raise TypeError(
            f"weather.perturbation {ps.target} returned {type(perturbation).__name__}, "
            "not a WeatherPerturbation"
        )
    return perturbation
