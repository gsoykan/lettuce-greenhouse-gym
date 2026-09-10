"""The experiment spec: defaults are the benchmark, dicts round-trip, builders make the right objects."""

from pathlib import Path

import numpy as np
import pytest

from lettuce_greenhouse_gym import EnvConfig
from lettuce_greenhouse_gym.baselines import AllOff, run_episode
from lettuce_greenhouse_gym.config import (
    ActionMode,
    ControlOverride,
    InitialControl,
    TimestepEncoding,
)
from lettuce_greenhouse_gym.experiment import (
    CallableSpec,
    ExperimentSpec,
    ParameterSpec,
    SplitSpec,
    TrainSpec,
    WeatherFileSpec,
    WeatherLoaderSpec,
    WeatherSpec,
    build_env,
    build_sampler,
    from_dict,
    load_yaml,
    resolve_target,
    save_yaml,
    to_dict,
)
from lettuce_greenhouse_gym.model.parameters import RandomizedParameterProvider
from lettuce_greenhouse_gym.rewards import PenaltyBound, RewardConfig, benchmark_reward_config
from lettuce_greenhouse_gym.weather import (
    CyclingWeatherSampler,
    FixedWeatherSampler,
    RandomWeatherSampler,
    WeatherSeries,
)

CUSTOM = ExperimentSpec(
    env=EnvConfig(
        dt=900.0,
        action_mode=ActionMode.ABSOLUTE,
        timestep_encoding=TimestepEncoding.INDEX,
        reward=RewardConfig(
            benchmark_reward_config(energy_cost=0.25).economics,
            (PenaltyBound("indoor_temp", 12.0, 22.0, 1e-3, 2e-3),),
        ),
        control_overrides=(ControlOverride("heating", hi=200.0),),
        initial_control=(InitialControl("heating", 80.0),),
    ),
    weather=WeatherSpec(start_day=None, split=SplitSpec(0.7, 3)),
    parameters=ParameterSpec(ranges={"leak": (2e-5, 3e-5)}),
    train=TrainSpec(algo="sac", n_envs=2, net_arch=(64, 64), algo_kwargs={"batch_size": 64}),
)


def test_default_spec_is_the_benchmark():
    spec = ExperimentSpec()
    assert spec.env == EnvConfig()
    assert isinstance(build_sampler(spec), FixedWeatherSampler)
    log = run_episode(build_env(spec), AllOff(), seed=0)
    assert log.total_return == pytest.approx(-64.53472461089223, rel=1e-6)


def test_dict_round_trip_is_the_identity():
    d = to_dict(CUSTOM)
    assert from_dict(d) == CUSTOM
    assert to_dict(from_dict(d)) == d
    assert d["env"]["action_mode"] == "absolute"  # plain data, not an enum
    assert d["env"]["timestep_encoding"] == "index"
    assert d["env"]["reward"]["prices"]["energy_cost"] == 0.25


def test_missing_keys_take_benchmark_defaults():
    assert from_dict({}) == ExperimentSpec()
    spec = from_dict({"env": {"reward": {"prices": {"energy_cost": 0.3}}}})
    assert spec.env.reward.economics.get("energy_cost") == 0.3
    spec = from_dict({"env": {"reward": {"prices": {"energy_cost": 0.3}}}})
    assert spec.env.reward.economics.get("energy_cost") == 0.3
    assert spec.env.reward.bounds == EnvConfig().reward.bounds


def test_weather_modes_build_the_expected_samplers():
    listed = ExperimentSpec(weather=WeatherSpec(start_day=None, start_days=(20.0, 60.0)))
    assert isinstance(build_sampler(listed, "train"), RandomWeatherSampler)
    assert isinstance(build_sampler(listed, "eval"), CyclingWeatherSampler)
    train = build_sampler(CUSTOM, "train")
    ev = build_sampler(CUSTOM, "eval")
    assert set(train.start_days).isdisjoint(ev.start_days)
    assert len(train.start_days) > len(ev.start_days)


def test_parameter_ranges_become_a_randomizing_provider():
    env = build_env(CUSTOM)
    assert isinstance(env.parameter_provider, RandomizedParameterProvider)
    assert build_env(ExperimentSpec()).parameter_provider is not None  # env default, fixed


def test_spec_validation():
    with pytest.raises(ValueError, match="exactly one"):
        WeatherSpec(start_days=(20.0,))  # start_day still set by default
    with pytest.raises(ValueError, match="algo"):
        TrainSpec(algo="dqn")
    with pytest.raises(ValueError, match="unknown"):
        from_dict({"env": {"reward": {"prices": {"no_such_price": 1.0}}}})
    with pytest.raises(ValueError, match="exceeds"):
        build_env(ExperimentSpec(parameters=ParameterSpec(ranges={"leak": (2e-5, 1e-5)})))


def test_yaml_round_trip_is_plain_data(tmp_path):
    import yaml

    path = save_yaml(CUSTOM, tmp_path / "run" / "spec.yaml")  # parent dir is created
    assert load_yaml(path) == CUSTOM
    text = path.read_text()
    assert "!!python" not in text  # no object tags: safe to load from anywhere
    assert yaml.safe_load(text) == to_dict(CUSTOM)


def test_hand_written_partial_yaml_loads(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text("train:\n  algo: sac\nparameters:\n  ranges: {leak: [2e-5, 3e-5]}\n")
    spec = load_yaml(path)
    assert spec.train.algo == "sac"
    assert spec.parameters.ranges == {"leak": (2e-5, 3e-5)}
    assert spec.env == EnvConfig()


def test_hand_written_scientific_notation_is_numeric(tmp_path):
    """PyYAML reads ``5e-5`` as a string (YAML 1.1); the loader must hand back numbers."""
    path = tmp_path / "spec.yaml"
    path.write_text(
        "env:\n  reward:\n    bounds:\n"
        "      - {name: rh, lo: 0, hi: 80, w_lo: 7e-4, w_hi: 7e-4}\n"
        "parameters:\n  ranges: {leak: [2e-5, 3e-5]}\n"
    )
    spec = load_yaml(path)
    assert spec.env.reward.bounds[0].w_lo == 7e-4
    assert spec.parameters.ranges == {"leak": (2e-5, 3e-5)}


def test_relative_parameter_spec_builds_a_per_step_provider():
    spec = ExperimentSpec(parameters=ParameterSpec(relative=0.05, per_step=True))
    env = build_env(spec)
    assert isinstance(env.parameter_provider, RandomizedParameterProvider)
    assert env.parameter_provider.per_step is True
    d = to_dict(spec)
    assert d["parameters"] == {
        "ranges": {},
        "relative": 0.05,
        "names": None,
        "per_step": True,
        "provider": None,
    }
    assert from_dict(d) == spec
    assert from_dict(to_dict(ExperimentSpec())).parameters.is_nominal


def test_parameter_spec_validation():
    with pytest.raises(ValueError, match="not both"):
        ParameterSpec(ranges={"leak": (2e-5, 3e-5)}, relative=0.05)
    with pytest.raises(ValueError, match="relative"):
        ParameterSpec(relative=1.5)
    with pytest.raises(ValueError, match="names"):
        ParameterSpec(names=("leak",))
    with pytest.raises(TypeError, match="parameters must be a ParameterSpec"):
        ExperimentSpec(parameters={"leak": (2e-5, 3e-5)})


# ---- user traces, several sources, train != eval ----------------------------------------------
def _write_synthetic_csv(path, days=3.0):
    """A 5-minute trace in the package's CSV format starting 1 January 00:00 (epoch_day 1.0)."""
    from lettuce_greenhouse_gym.units import co2_density_to_ppm, vapour_density_to_rh

    s = WeatherSeries.synthetic(days=days, dt=300.0, epoch_day=1.0)
    rad, co2, temp, vap = s.values
    table = np.column_stack(
        [
            np.arange(s.n_samples) * 300.0,
            rad,
            temp,
            vapour_density_to_rh(temp, vap),
            np.zeros(s.n_samples),
            co2_density_to_ppm(temp, co2),
        ]
    )
    np.savetxt(path, table, delimiter=",")
    return path


def _multi_source_spec(tmp_path):
    a = _write_synthetic_csv(tmp_path / "year_a.csv")
    b = _write_synthetic_csv(tmp_path / "year_b.csv")
    return ExperimentSpec(
        env=EnvConfig(episode_days=0.5),
        weather=WeatherSpec(
            source=("year_a", "year_b"),
            start_day=1.0,
            files={"year_a": WeatherFileSpec(str(a), 1.0), "year_b": WeatherFileSpec(str(b), 1.0)},
            eval_source="synthetic",
            eval_start_days=(2.0,),
        ),
    )


def test_multi_source_spec_round_trips_and_builds_the_right_samplers(tmp_path):
    spec = _multi_source_spec(tmp_path)
    d = to_dict(spec)
    assert d["weather"]["source"] == ["year_a", "year_b"]
    assert d["weather"]["files"]["year_a"]["epoch_day"] == 1.0
    assert from_dict(d) == spec
    train = build_sampler(spec, "train")
    assert isinstance(train, RandomWeatherSampler)
    assert train.sources == ("year_a", "year_b")
    ev = build_sampler(spec, "eval")
    assert isinstance(ev, FixedWeatherSampler)
    assert ev.sample(np.random.default_rng(0)).source == "synthetic"


def test_build_env_reads_user_files_and_reports_their_names(tmp_path):
    spec = _multi_source_spec(tmp_path)
    env = build_env(spec, "train")
    seen = {env.reset(seed=s)[1]["weather_source"] for s in range(8)}
    assert seen == {"year_a", "year_b"}
    env.step(np.zeros(3))
    assert build_env(spec, "eval").reset(seed=0)[1]["weather_source"] == "synthetic"


def test_weather_spec_validation():
    with pytest.raises(ValueError, match="unknown weather source"):
        WeatherSpec(source="nope")
    with pytest.raises(ValueError, match="packaged"):
        WeatherSpec(files={"bleiswijk_2014": WeatherFileSpec("x.csv", 1.0)})
    with pytest.raises(ValueError, match="single source"):
        WeatherSpec(source=("bleiswijk_2014", "synthetic"), start_day=None, split=SplitSpec())
    with pytest.raises(ValueError, match="already defines"):
        WeatherSpec(start_day=None, split=SplitSpec(), eval_source="synthetic")
    with pytest.raises(ValueError, match="epoch_day"):
        WeatherFileSpec("x.csv", 0.0)
    assert WeatherSpec("bleiswijk_2014", 40.0, None, None).sources == (
        "bleiswijk_2014",
    )  # positional


def test_load_yaml_anchors_relative_file_paths_to_the_yaml_directory(tmp_path):
    _write_synthetic_csv(tmp_path / "trace.csv")
    (tmp_path / "spec.yaml").write_text(
        "env: {episode_days: 0.5}\n"
        "weather:\n  source: mine\n  start_day: 1.0\n"
        "  files: {mine: {path: trace.csv, epoch_day: 1.0}}\n"
    )
    spec = load_yaml(tmp_path / "spec.yaml")
    assert spec.weather.files["mine"].path == str((tmp_path / "trace.csv").resolve())
    assert build_env(spec).reset(seed=0)[1]["weather_source"] == "mine"


# ---- loaders: any callable that returns a WeatherSeries -------------------------------------------

EXAMPLES = Path(__file__).parents[1] / "examples"
EXAMPLE_LOADER = str(EXAMPLES / "weather_epw.py")
TXT_LOADER = str(EXAMPLES / "weather_txt.py")


def test_loader_spec_validates_its_target():
    for bad in ("no_colon", "pkg.mod:", "pkg.mod:1abc", ":func"):
        with pytest.raises(ValueError, match="target"):
            WeatherLoaderSpec(bad)
    ls = WeatherLoaderSpec("my_pkg.weather:Reader.load")
    assert (ls.module, ls.attribute, ls.is_script) == ("my_pkg.weather", "Reader.load", False)
    assert WeatherLoaderSpec("scripts/w.py:load").is_script


def test_loader_spec_names_are_sources_and_may_not_clash():
    spec = WeatherSpec(
        source="gen",
        start_day=2.0,
        loaders={
            "gen": WeatherLoaderSpec("lettuce_greenhouse_gym.weather:WeatherSeries.synthetic")
        },
    )
    assert spec.sources == ("gen",)
    with pytest.raises(ValueError, match="packaged"):
        WeatherSpec(loaders={"synthetic": WeatherLoaderSpec("m:f")})
    with pytest.raises(ValueError, match="both files and loaders"):
        WeatherSpec(
            files={"x": WeatherFileSpec("x.csv", 1.0)}, loaders={"x": WeatherLoaderSpec("m:f")}
        )
    with pytest.raises(ValueError, match="files or loaders"):
        WeatherSpec(source="nope")


def test_module_target_with_kwargs_builds_an_env_that_reports_the_loader_name():
    spec = ExperimentSpec(
        env=EnvConfig(episode_days=0.5),
        weather=WeatherSpec(
            source="gen",
            start_day=2.0,
            loaders={
                "gen": WeatherLoaderSpec(
                    "lettuce_greenhouse_gym.weather:WeatherSeries.synthetic", {"days": 4.0}
                )
            },
        ),
    )
    assert from_dict(to_dict(spec)) == spec
    env = build_env(spec)
    assert env.reset(seed=0)[1]["weather_source"] == "gen"
    assert env.weather_repository.load("gen").span_days == 4.0


def test_script_target_loads_the_epw_example(tmp_path):
    write_demo = resolve_target(f"{EXAMPLE_LOADER}:write_demo_epw")
    epw = write_demo(tmp_path / "demo.epw", 3)
    spec = ExperimentSpec(
        env=EnvConfig(episode_days=0.5),
        weather=WeatherSpec(
            source="site",
            start_day=1.5,
            loaders={"site": WeatherLoaderSpec(f"{EXAMPLE_LOADER}:load_epw", {"path": str(epw)})},
        ),
    )
    env = build_env(spec)
    _, info = env.reset(seed=0)
    assert info["weather_source"] == "site"
    series = env.weather_repository.load("site")
    assert series.dt == 300.0
    assert series.epoch_day == pytest.approx(
        1.0 + 1 / 24
    )  # first EPW row is the hour ending 01:00
    with pytest.raises(AttributeError):
        resolve_target(f"{EXAMPLE_LOADER}:no_such_function")
    with pytest.raises(TypeError, match="not callable"):
        resolve_target(f"{EXAMPLE_LOADER}:YEAR")


def test_load_yaml_anchors_loader_script_and_path_kwarg_to_the_yaml_directory(tmp_path):
    (tmp_path / "loaders").mkdir()
    (tmp_path / "loaders" / "gen.py").write_text(
        "from lettuce_greenhouse_gym import WeatherSeries\n"
        "def make(path, days):\n"
        "    assert path.startswith('/')\n"
        "    return WeatherSeries.synthetic(days=days)\n"
    )
    (tmp_path / "spec.yaml").write_text(
        "env: {episode_days: 0.5}\n"
        "weather:\n  source: gen\n  start_day: 1.0\n"
        "  loaders: {gen: {target: loaders/gen.py:make, kwargs: {path: data/x.bin, days: 3.0}}}\n"
    )
    spec = load_yaml(tmp_path / "spec.yaml")
    ls = spec.weather.loaders["gen"]
    assert ls.module == str((tmp_path / "loaders" / "gen.py").resolve())
    assert ls.kwargs == {"path": str((tmp_path / "data" / "x.bin").resolve()), "days": 3.0}
    assert build_env(spec).reset(seed=0)[1]["weather_source"] == "gen"
    out = save_yaml(spec, tmp_path / "out.yaml")
    assert load_yaml(out) == spec


def test_script_target_loads_the_txt_example(tmp_path):
    txt = resolve_target(f"{TXT_LOADER}:write_demo_txt")(tmp_path / "station.txt", 5)
    spec = ExperimentSpec(
        env=EnvConfig(episode_days=3.0),
        weather=WeatherSpec(
            source="station",
            start_day=61.0,
            loaders={"station": WeatherLoaderSpec(f"{TXT_LOADER}:load_txt", {"path": str(txt)})},
        ),
    )
    env = build_env(spec)
    assert env.reset(seed=0)[1]["weather_source"] == "station"
    series = env.weather_repository.load("station")
    assert series.epoch_day == 60.0  # 1 March 2023 00:00
    assert series.dt == 300.0
    assert series.n_samples == 119 * 12 + 1  # 120 hourly rows span 119 h, on a 5-minute grid


# ---- callable specs for parameter providers and weather perturbations -----------------------------


def test_parameter_provider_spec_builds_any_provider_from_a_target(tmp_path):
    spec = ExperimentSpec(
        env=EnvConfig(episode_days=0.5),
        parameters=ParameterSpec(
            provider=CallableSpec("lettuce_greenhouse_gym.model.parameters:FixedParameterProvider")
        ),
    )
    assert not spec.parameters.is_nominal
    env = build_env(spec)
    assert type(env.parameter_provider).__name__ == "FixedParameterProvider"
    d = to_dict(spec)
    assert d["parameters"]["provider"] == {
        "target": "lettuce_greenhouse_gym.model.parameters:FixedParameterProvider",
        "kwargs": {},
    }
    assert from_dict(d) == spec
    with pytest.raises(ValueError, match="one scheme only"):
        ParameterSpec(relative=0.05, provider=CallableSpec("m:f"))
    (tmp_path / "bad.py").write_text("def make(constants):\n    return object()\n")
    bad = ExperimentSpec(
        parameters=ParameterSpec(provider=CallableSpec(f"{tmp_path / 'bad.py'}:make"))
    )
    with pytest.raises(TypeError, match="not a ParameterProvider"):
        build_env(bad)


def test_weather_perturbation_spec_reaches_the_env_and_anchors_its_script(tmp_path):
    (tmp_path / "hooks.py").write_text(
        "import numpy as np\n"
        "from lettuce_greenhouse_gym import WeatherPerturbation\n"
        "class Warmer(WeatherPerturbation):\n"
        "    def __init__(self, degrees):\n"
        "        self.degrees = degrees\n"
        "    def realise(self, rng, weather, dt):\n"
        "        out = weather.copy(); out[2] += self.degrees; return out\n"
    )
    (tmp_path / "spec.yaml").write_text(
        "env: {episode_days: 0.5}\n"
        "weather:\n  perturbation: {target: hooks.py:Warmer, kwargs: {degrees: 2.0}}\n"
        "parameters:\n"
        "  provider: {target: lettuce_greenhouse_gym.model.parameters:FixedParameterProvider}\n"
    )
    spec = load_yaml(tmp_path / "spec.yaml")
    assert spec.weather.perturbation.module == str((tmp_path / "hooks.py").resolve())
    warm = build_env(spec)
    plain = build_env(ExperimentSpec(env=EnvConfig(episode_days=0.5)))
    o_warm, _ = warm.reset(seed=0)
    o_plain, _ = plain.reset(seed=0)
    assert o_warm[10] == pytest.approx(o_plain[10] + 2.0)
    assert load_yaml(save_yaml(spec, tmp_path / "out.yaml")) == spec
    bad = ExperimentSpec(weather=WeatherSpec(perturbation=CallableSpec("builtins:dict")))
    with pytest.raises(TypeError, match="not a WeatherPerturbation"):
        build_env(bad)


def test_sampler_specs_replace_the_day_modes_per_role(tmp_path):
    (tmp_path / "paired.py").write_text(
        "from lettuce_greenhouse_gym import CyclingWeatherSampler, WeatherScenario\n"
        "class Paired(CyclingWeatherSampler):\n"
        "    def __init__(self, pairs):\n"
        "        self._pairs = [WeatherScenario(s, float(d)) for s, d in pairs]\n"
        "        self._i = 0\n"
        "    def sample(self, rng, options=None):\n"
        "        chosen = self._pairs[self._i % len(self._pairs)]\n"
        "        self._i += 1\n"
        "        return chosen\n"
    )
    target = f"{tmp_path / 'paired.py'}:Paired"
    pairs = [["bleiswijk_2014", 40.0], ["synthetic", 3.0]]
    spec = ExperimentSpec(
        env=EnvConfig(episode_days=0.5),
        weather=WeatherSpec(
            start_day=None,
            sampler=CallableSpec(target, {"pairs": pairs}),
            eval_sampler=CallableSpec(target, {"pairs": [["synthetic", 5.0]]}),
        ),
    )
    train = build_sampler(spec, "train")
    seen = [train.sample(np.random.default_rng(0)) for _ in range(3)]
    assert [(s.source, s.start_day) for s in seen] == [
        ("bleiswijk_2014", 40.0),
        ("synthetic", 3.0),
        ("bleiswijk_2014", 40.0),
    ]
    assert build_sampler(spec, "eval").sample(np.random.default_rng(0)).source == "synthetic"
    assert from_dict(to_dict(spec)) == spec
    # no eval_sampler and no day mode: evaluation gets a fresh instance of the training sampler
    only = ExperimentSpec(
        weather=WeatherSpec(start_day=None, sampler=CallableSpec(target, {"pairs": pairs}))
    )
    ev = build_sampler(only, "eval")
    assert ev.sample(np.random.default_rng(0)).source == "bleiswijk_2014"
    with pytest.raises(ValueError, match="needs a sampler"):
        WeatherSpec(eval_sampler=CallableSpec(target))
    with pytest.raises(ValueError, match="at most one"):
        WeatherSpec(start_days=(1.0,), sampler=CallableSpec(target))
    with pytest.raises(TypeError, match="not a WeatherSampler"):
        build_sampler(
            ExperimentSpec(
                weather=WeatherSpec(
                    start_day=None,
                    sampler=CallableSpec(
                        "lettuce_greenhouse_gym.model.parameters:FixedParameterProvider",
                        {"base": None},
                    ),
                )
            )
        )


# ---- wrappers ---------------------------------------------------------------------------------------


def test_wrappers_round_trip_and_wrap_the_built_env(tmp_path):
    from lettuce_greenhouse_gym.experiment import apply_wrappers

    spec = ExperimentSpec(
        env=EnvConfig(episode_days=0.5),
        weather=WeatherSpec(source="synthetic", start_day=3.0),
        wrappers=(
            CallableSpec(
                "lettuce_greenhouse_gym.wrappers:ParameterObservation",
                {"names": ["leak"], "ranges": {"leak": [0.0, 1.0e-4]}},
            ),
        ),
    )
    assert from_dict(to_dict(spec)) == spec
    saved = save_yaml(spec, tmp_path / "spec.yaml")
    assert "!!python" not in saved.read_text()
    assert load_yaml(saved) == spec
    bare = build_env(spec)
    env = apply_wrappers(spec, bare)
    assert env.unwrapped is bare
    assert env.observation_space.shape[0] == bare.observation_space.shape[0] + 1
    obs, _ = env.reset(seed=0)
    assert obs[-1] == pytest.approx(0.75e-5 / 1.0e-4)
    log = run_episode(env, AllOff(), seed=0)
    assert log.x.shape[0] == log.u.shape[0] + 1
    assert apply_wrappers(ExperimentSpec(), bare) is bare


def test_wrappers_are_validated_and_anchored(tmp_path):
    from lettuce_greenhouse_gym.experiment import anchor_files, apply_wrappers

    with pytest.raises(TypeError, match="tuple of CallableSpec"):
        ExperimentSpec(wrappers=[CallableSpec("m:f")])
    bad = ExperimentSpec(wrappers=(CallableSpec("builtins:id"),))
    with pytest.raises(TypeError, match="not a gymnasium"):
        apply_wrappers(bad, build_env(bad))
    d = anchor_files({"wrappers": [{"target": "wrap.py:make", "kwargs": {}}]}, tmp_path)
    assert d["wrappers"][0]["target"] == f"{(tmp_path / 'wrap.py').resolve()}:make"
