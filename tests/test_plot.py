"""The season figure renders headlessly and covers every panel."""

import pytest

from lettuce_greenhouse_gym import EnvConfig, LettuceGreenhouseEnv
from lettuce_greenhouse_gym.baselines import GrowerHeuristic, run_episode

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_plot_episode_renders_all_panels(tmp_path):
    from lettuce_greenhouse_gym.plot import plot_episode

    config = EnvConfig(episode_days=1.0)
    log = run_episode(LettuceGreenhouseEnv(config), GrowerHeuristic(), seed=0)
    fig = plot_episode(log, config, title="one day")
    assert len(fig.axes) == 10  # 3x3 grid plus the twin weather axis
    out = tmp_path / "season.png"
    fig.savefig(out)
    assert out.stat().st_size > 10_000
    matplotlib.pyplot.close(fig)


def test_plot_episode_defaults_to_the_benchmark_config():
    from lettuce_greenhouse_gym.plot import plot_episode

    log = run_episode(LettuceGreenhouseEnv(EnvConfig(episode_days=0.5)), GrowerHeuristic(), seed=0)
    fig = plot_episode(log)  # dt from EnvConfig(): the time axis still spans 0.5 days
    assert fig.axes[0].get_xlim()[1] >= 0.5
    matplotlib.pyplot.close(fig)
