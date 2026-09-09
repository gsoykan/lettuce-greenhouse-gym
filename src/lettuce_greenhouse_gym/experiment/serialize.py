"""Spec <-> plain dict <-> YAML. The schema is documented in the package docstring."""

import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import (
    ActionMode,
    ControlOverride,
    EnvConfig,
    InitialControl,
    InitialState,
    TimestepEncoding,
)
from ..rewards import PenaltyBound, RewardConfig, benchmark_reward_config
from .spec import (
    CallableSpec,
    ExperimentSpec,
    ParameterSpec,
    SplitSpec,
    TrainSpec,
    WandbSpec,
    WeatherFileSpec,
    WeatherLoaderSpec,
    WeatherSpec,
)


def _callable(d: dict[str, Any] | None) -> CallableSpec | None:
    return None if d is None else CallableSpec(d["target"], dict(d.get("kwargs") or {}))


def _plain(value: Any) -> Any:
    """Recursively turn a spec value into YAML-safe plain data."""
    if isinstance(value, RewardConfig):
        return {
            "prices": {n: float(value.economics.get(n)) for n in value.economics.names},
            "bounds": [asdict(b) for b in value.bounds],
        }
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_dict(spec: ExperimentSpec) -> dict[str, Any]:
    """Canonical plain-data form: enums as strings, tuples as lists, reward as prices + bounds."""
    raw = {
        "env": {f: getattr(spec.env, f) for f in EnvConfig.__dataclass_fields__},
        "weather": asdict(spec.weather),
        "parameters": asdict(spec.parameters),
        "train": asdict(spec.train),
    }
    return _plain(raw)


def _reward_from_dict(d: dict[str, Any] | None) -> RewardConfig:
    prices = (d or {}).get("prices") or {}
    config = benchmark_reward_config(**prices)
    bounds = (d or {}).get("bounds")
    if bounds is None:
        return config
    return RewardConfig(config.economics, tuple(PenaltyBound(**b) for b in bounds))


def _env_from_dict(d: dict[str, Any]) -> EnvConfig:
    d = dict(d)
    if "action_mode" in d:
        d["action_mode"] = ActionMode(d["action_mode"])
    if "timestep_encoding" in d:
        d["timestep_encoding"] = TimestepEncoding(d["timestep_encoding"])
    if "reward" in d:
        d["reward"] = _reward_from_dict(d["reward"])
    for key, cls in (
        ("control_overrides", ControlOverride),
        ("initial_state", InitialState),
        ("initial_control", InitialControl),
    ):
        if key in d:
            d[key] = tuple(cls(**item) for item in d[key])
    return EnvConfig(**d)


def from_dict(d: dict[str, Any]) -> ExperimentSpec:
    """Inverse of :func:`to_dict`; missing keys take the benchmark defaults."""
    weather = dict(d.get("weather") or {})
    for key in ("source", "eval_source"):
        if isinstance(weather.get(key), list):
            weather[key] = tuple(weather[key])
    for key in ("start_days", "eval_start_days"):
        if weather.get(key) is not None:
            weather[key] = tuple(weather[key])
    if weather.get("split") is not None:
        weather["split"] = SplitSpec(**weather["split"])
    weather["files"] = {k: WeatherFileSpec(**v) for k, v in (weather.get("files") or {}).items()}
    weather["loaders"] = {
        k: WeatherLoaderSpec(v["target"], dict(v.get("kwargs") or {}))
        for k, v in (weather.get("loaders") or {}).items()
    }
    for key in ("perturbation", "sampler", "eval_sampler"):
        weather[key] = _callable(weather.get(key))
    parameters = dict(d.get("parameters") or {})
    parameters["provider"] = _callable(parameters.get("provider"))
    parameters["ranges"] = {
        k: (float(lo), float(hi)) for k, (lo, hi) in (parameters.get("ranges") or {}).items()
    }
    if parameters.get("names") is not None:
        parameters["names"] = tuple(parameters["names"])
    train = dict(d.get("train") or {})
    if train.get("net_arch") is not None:
        train["net_arch"] = tuple(train["net_arch"])
    if train.get("wandb") is not None:
        wandb = dict(train["wandb"])
        wandb["tags"] = tuple(wandb.get("tags", ()))
        train["wandb"] = WandbSpec(**wandb)
    return ExperimentSpec(
        env=_env_from_dict(d.get("env") or {}),
        weather=WeatherSpec(**weather),
        parameters=ParameterSpec(**parameters),
        train=TrainSpec(**train),
    )


def save_yaml(spec: ExperimentSpec, path: str | Path) -> Path:
    """Write the canonical form of ``spec``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_dict(spec), sort_keys=False))
    return path


# YAML 1.1 (PyYAML) reads "2e-5" as a string because its float grammar needs a dot; "2.0e-5" is
# a float. Hand-written specs and --set values use the short form, so coerce it after parsing.
_SCIENTIFIC = re.compile(r"^[-+]?\d+(?:\.\d*)?[eE][-+]?\d+$")


def _coerce_scientific(value: Any) -> Any:
    if isinstance(value, str) and _SCIENTIFIC.match(value):
        return float(value)
    if isinstance(value, dict):
        return {k: _coerce_scientific(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_scientific(v) for v in value]
    return value


def parse_yaml(text: str) -> Any:
    """``yaml.safe_load`` plus numbers in scientific notation, the one place YAML 1.1 surprises."""
    return _coerce_scientific(yaml.safe_load(text))


def _anchored(value: Any, base: str | Path) -> Any:
    if isinstance(value, str) and value and not Path(value).is_absolute():
        return str((Path(base) / value).resolve())
    return value


def _anchor_callable(cs: dict[str, Any] | None, base: str | Path) -> None:
    """A ``CallableSpec`` as plain data: anchor a script target and a ``path`` keyword argument."""
    if not cs:
        return
    module, sep, attr = str(cs.get("target", "")).rpartition(":")
    if sep and module.endswith(".py"):
        cs["target"] = f"{_anchored(module, base)}:{attr}"
    kwargs = cs.get("kwargs") or {}
    if "path" in kwargs:
        kwargs["path"] = _anchored(kwargs["path"], base)


def anchor_files(d: dict[str, Any], base: str | Path) -> dict[str, Any]:
    """Resolve the relative paths a spec may hold against ``base`` (a directory).

    These are ``weather.files.*.path`` and, for every callable spec (``weather.loaders.*``,
    ``weather.perturbation``, ``weather.sampler``, ``weather.eval_sampler``,
    ``parameters.provider``), a script target written as
    ``script.py:callable`` and a ``kwargs.path`` entry.
    """
    weather = d.get("weather") or {}
    for f in (weather.get("files") or {}).values():
        if "path" in f:
            f["path"] = _anchored(f["path"], base)
    for ls in (weather.get("loaders") or {}).values():
        _anchor_callable(ls, base)
    for key in ("perturbation", "sampler", "eval_sampler"):
        _anchor_callable(weather.get(key), base)
    _anchor_callable((d.get("parameters") or {}).get("provider"), base)
    return d


def load_yaml(path: str | Path) -> ExperimentSpec:
    """Read a spec written by :func:`save_yaml`, or any hand-written subset of it.

    Relative paths in the weather block (see :func:`anchor_files`) are taken relative to the YAML
    file's directory.
    """
    path = Path(path)
    return from_dict(anchor_files(parse_yaml(path.read_text()) or {}, path.parent))
