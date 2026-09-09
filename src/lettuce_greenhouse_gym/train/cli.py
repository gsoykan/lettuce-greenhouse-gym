"""``lettuce-gym``: train, evaluate, and run the reference controllers from a spec plus flags.

Precedence, lowest to highest: ``ExperimentSpec()`` defaults, ``--config`` YAML, repeatable
``--set key.path=value``, named flags.
"""

import argparse
import copy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ..baselines import AllOff, ConstantControl, GrowerHeuristic, NominalMPC, run_episode
from ..experiment import (
    ExperimentSpec,
    anchor_files,
    build_env,
    build_repository,
    from_dict,
    load_yaml,
    parse_yaml,
    to_dict,
)


def _assign(tree: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``tree[a][b][c] = value`` for ``dotted == "a.b.c"``, creating dicts along the way."""
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):  # None (e.g. wandb: null) becomes a dict
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def merge(base: dict[str, Any], sets: Sequence[str], flags: dict[str, Any]) -> dict[str, Any]:
    """Apply ``--set`` items (YAML-parsed values) then named flags (``None`` = not given)."""
    out = copy.deepcopy(base)
    for item in sets:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        _assign(out, key.strip(), parse_yaml(raw))
    for dotted, value in flags.items():
        if value is not None:
            _assign(out, dotted, value)
    return out


# named flag -> dotted spec key
_FLAGS = {
    "algo": "train.algo",
    "timesteps": "train.total_timesteps",
    "n_envs": "train.n_envs",
    "seed": "train.seed",
    "log_dir": "train.log_dir",
    "run_name": "train.run_name",
    "normalize_obs": "train.normalize_obs",
    "wandb": "train.wandb.project",
    "eval_every": "train.eval_every",
    "checkpoint_every": "train.checkpoint_every",
}


def spec_from_args(args: argparse.Namespace) -> ExperimentSpec:
    base = to_dict(load_yaml(args.config)) if args.config else to_dict(ExperimentSpec())
    flags = {dotted: getattr(args, name, None) for name, dotted in _FLAGS.items()}
    return from_dict(anchor_files(merge(base, args.set or [], flags), Path.cwd()))


def _add_spec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, help="YAML spec; see lettuce_greenhouse_gym.experiment"
    )
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="override one spec key"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="show the merged spec and exit"
    )


def _cell(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(_cell(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.rjust(widths[k]) for k in keys))
    for r in rows:
        print("  ".join(_cell(r[k]).rjust(widths[k]) for k in keys))


def cmd_train(args: argparse.Namespace) -> int:
    spec = spec_from_args(args)
    if args.print_config:
        print(yaml.safe_dump(to_dict(spec), sort_keys=False), end="")
        return 0
    from .sb3 import train, train_seeds

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
        for result in train_seeds(spec, seeds):
            print(f"saved {result.model_path} (seed {result.spec.train.seed})")
        return 0
    result = train(spec)
    print(f"saved {result.model_path} after {result.timesteps} steps")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .sb3 import evaluate, load, return_components

    spec, model, vecnormalize = load(args.run_dir)
    options = None if args.start_day is None else {"start_day": args.start_day}
    log = evaluate(
        model,
        spec,
        vecnormalize=vecnormalize,
        seed=args.seed,
        deterministic=not args.stochastic,
        options=options,
    )
    _print_table([{"controller": args.run_dir.name, **return_components(log)}])
    if args.plot is not None:
        from ..plot import plot_episode

        plot_episode(log, spec.env, title=args.run_dir.name).savefig(args.plot, dpi=120)
        print(f"figure saved to {args.plot}")
    return 0


BASELINES = {
    "all_off": lambda spec: AllOff(),
    "hold_u0": lambda spec: ConstantControl(spec.env.effective_u0()),
    "heuristic": lambda spec: GrowerHeuristic(),
    "mpc": lambda spec: NominalMPC(),
}


def cmd_baselines(args: argparse.Namespace) -> int:
    from .sb3 import return_components  # no SB3 import happens here; the module is lazy

    spec = spec_from_args(args)
    if args.print_config:
        print(yaml.safe_dump(to_dict(spec), sort_keys=False), end="")
        return 0
    names = args.controllers.split(",")
    unknown = set(names) - set(BASELINES)
    if unknown:
        raise SystemExit(f"unknown controller(s) {sorted(unknown)}; choose from {list(BASELINES)}")
    rows = []
    for name in names:
        log = run_episode(build_env(spec, "eval"), BASELINES[name](spec), seed=args.seed)
        rows.append({"controller": name, **return_components(log)})
        if args.plot is not None:
            from ..plot import plot_episode

            args.plot.mkdir(parents=True, exist_ok=True)
            plot_episode(log, spec.env, title=name).savefig(args.plot / f"{name}.png", dpi=120)
    _print_table(rows)
    return 0


def cmd_weather_fetch_knmi(args: argparse.Namespace) -> int:
    from ..knmi import fetch_year

    years = _parse_years(args.years)
    for year in years:
        path = fetch_year(args.station, year, args.out, dt=args.dt, co2_ppm=args.co2)
        print(f"wrote {path}")
    print(f"attribution written to {Path(args.out) / 'SOURCE.md'} (KNMI, CC BY 4.0)")
    return 0


def cmd_weather_list(args: argparse.Namespace) -> int:
    spec = spec_from_args(args) if (args.config or args.set) else ExperimentSpec()
    for name in build_repository(spec).names:
        print(name)
    return 0


def _parse_years(text: str) -> list[int]:
    """``2010`` or ``2001-2010`` or ``2001,2003``."""
    years: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            years.extend(range(a, b + 1))
        else:
            years.append(int(part))
    return years


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lettuce-gym",
        description="Train, evaluate, and run the reference controllers of LettuceGreenhouse-v0.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train PPO or SAC per a spec")
    _add_spec_arguments(p_train)
    p_train.add_argument("--algo", choices=["ppo", "sac"])
    p_train.add_argument("--timesteps", type=int)
    p_train.add_argument("--n-envs", type=int, dest="n_envs")
    p_train.add_argument("--seed", type=int)
    p_train.add_argument("--log-dir", dest="log_dir")
    p_train.add_argument("--run-name", dest="run_name")
    p_train.add_argument(
        "--normalize-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="normalize_obs",
    )
    p_train.add_argument("--wandb", metavar="PROJECT", help="log to this Weights & Biases project")
    p_train.add_argument(
        "--eval-every",
        type=int,
        dest="eval_every",
        metavar="STEPS",
        help="evaluate on the eval weather every STEPS env steps and keep the best model",
    )
    p_train.add_argument(
        "--checkpoint-every",
        type=int,
        dest="checkpoint_every",
        metavar="STEPS",
        help="save a checkpoint every STEPS env steps",
    )
    p_train.add_argument(
        "--seeds", metavar="S1,S2,...", help="train one run per seed, named <run-name>_seed<S>"
    )
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="run a trained policy for one season")
    p_eval.add_argument("run_dir", type=Path)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument(
        "--stochastic", action="store_true", help="sample actions instead of the mean"
    )
    p_eval.add_argument(
        "--start-day", type=float, dest="start_day", help="override the season start"
    )
    p_eval.add_argument(
        "--plot", type=Path, metavar="FILE", help="save a season figure (needs the plot extra)"
    )
    p_eval.set_defaults(func=cmd_eval)

    p_base = sub.add_parser("baselines", help="season returns of the reference controllers")
    _add_spec_arguments(p_base)
    p_base.add_argument(
        "--controllers",
        default="all_off,hold_u0,heuristic",
        help=f"comma-separated subset of {list(BASELINES)}; mpc takes ~100 s",
    )
    p_base.add_argument("--seed", type=int, default=0)
    p_base.add_argument(
        "--plot", type=Path, metavar="DIR", help="save one figure per controller into DIR"
    )
    p_base.set_defaults(func=cmd_baselines)

    p_weather = sub.add_parser("weather", help="weather traces: fetch KNMI years, list sources")
    weather_sub = p_weather.add_subparsers(dest="weather_command", required=True)
    p_fetch = weather_sub.add_parser("fetch-knmi", help="download KNMI station-years as traces")
    p_fetch.add_argument("--station", type=int, default=344, help="KNMI station (344 = Rotterdam)")
    p_fetch.add_argument("--years", required=True, help="e.g. 2010, 2001-2010, or 2001,2003")
    p_fetch.add_argument("--out", type=Path, default=Path("weather"), help="output directory")
    p_fetch.add_argument("--dt", type=float, default=300.0, help="row spacing in seconds")
    p_fetch.add_argument("--co2", type=float, default=400.0, help="constant outdoor CO2 in ppm")
    p_fetch.set_defaults(func=cmd_weather_fetch_knmi)
    p_list = weather_sub.add_parser("list", help="sources a spec can name")
    _add_spec_arguments(p_list)
    p_list.set_defaults(func=cmd_weather_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
