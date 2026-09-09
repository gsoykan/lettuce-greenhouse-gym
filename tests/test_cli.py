"""The CLI's override logic and its SB3-free subcommand."""

import pytest

from lettuce_greenhouse_gym.experiment import ExperimentSpec, to_dict
from lettuce_greenhouse_gym.train.cli import _assign, main, merge

pytest.importorskip("yaml")  # merge parses values with the YAML parser


def test_assign_creates_nested_dicts_and_replaces_null():
    tree = {"train": {"wandb": None}}
    _assign(tree, "train.wandb.project", "p")
    _assign(tree, "new.deep.key", 1)
    assert tree == {"train": {"wandb": {"project": "p"}}, "new": {"deep": {"key": 1}}}


def test_merge_precedence_and_yaml_typed_values():
    base = to_dict(ExperimentSpec())
    out = merge(
        base,
        [
            "train.algo=sac",
            "parameters.ranges.leak=[2e-5, 3e-5]",
            "train.normalize_obs=false",
            "train.seed=7",
        ],
        {"train.seed": 3, "train.algo": None},  # a flag beats --set; None means not given
    )
    assert out["train"]["algo"] == "sac"
    assert out["train"]["seed"] == 3
    assert out["train"]["normalize_obs"] is False
    assert out["parameters"]["ranges"] == {"leak": [2e-05, 3e-05]}
    assert base["train"]["seed"] == 0  # the input is not mutated


def test_merge_rejects_malformed_set():
    with pytest.raises(ValueError, match=r"key\.path=value"):
        merge({}, ["train.algo"], {})


def test_print_config_shows_the_merged_spec(capsys):
    assert main(["train", "--print-config", "--algo", "sac", "--set", "env.episode_days=2"]) == 0
    out = capsys.readouterr().out
    assert "algo: sac" in out
    assert "episode_days: 2" in out


def test_baselines_subcommand_runs_without_sb3(capsys):
    code = main(["baselines", "--controllers", "all_off,heuristic", "--set", "env.episode_days=1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "all_off" in out
    assert "heuristic" in out
    assert "return" in out


def test_baselines_rejects_unknown_controller():
    with pytest.raises(SystemExit, match="unknown controller"):
        main(["baselines", "--controllers", "nope"])


def test_cadence_flags_map_to_the_spec(capsys):
    assert (
        main(["train", "--print-config", "--eval-every", "5000", "--checkpoint-every", "20000"])
        == 0
    )
    out = capsys.readouterr().out
    assert "eval_every: 5000" in out
    assert "checkpoint_every: 20000" in out


def test_weather_list_shows_packaged_and_spec_sources(capsys):
    assert main(["weather", "list"]) == 0
    out = capsys.readouterr().out
    assert "bleiswijk_2014" in out
    assert "synthetic" in out
    assert (
        main(
            [
                "weather",
                "list",
                "--set",
                "weather.loaders.gen.target=lettuce_greenhouse_gym.weather:WeatherSeries.synthetic",
            ]
        )
        == 0
    )
    assert "gen" in capsys.readouterr().out.split()


def test_year_ranges_parse():
    from lettuce_greenhouse_gym.train.cli import _parse_years

    assert _parse_years("2010") == [2010]
    assert _parse_years("2001-2003") == [2001, 2002, 2003]
    assert _parse_years("2001,2005-2006") == [2001, 2005, 2006]
