"""One figure per season: observables against their comfort bands, controls, weather, reward."""

import importlib

import numpy as np

from .._optional import require
from ..baselines import EpisodeLog
from ..config import EnvConfig
from ..variables.controls import CONTROL
from ..variables.exogenous import EXOGENOUS
from ..variables.observables import OBSERVABLE

_SECONDS_PER_DAY = 86400.0


def plot_episode(log: EpisodeLog, config: EnvConfig | None = None, *, title: str | None = None):
    """A 3x3 figure of one season; the caller saves or shows it.

    ``config`` supplies what the log does not carry: the step length for the time axis, the
    comfort bands (shaded) and the actuator limits (dashed). It defaults to the benchmark.
    """
    require("matplotlib")
    plt = importlib.import_module("matplotlib.pyplot")
    cfg = config if config is not None else EnvConfig()
    n = log.u.shape[0]
    days = np.arange(n + 1) * cfg.dt / _SECONDS_PER_DAY  # states/observables have n + 1 points
    step_days = days[:-1]  # controls, weather and rewards have n
    bands = {b.name: b for b in cfg.reward.bounds}
    u_lo, u_hi = cfg.effective_control_bounds()
    y_units = OBSERVABLE.units
    u_units = CONTROL.units

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)

    def observable(ax, name: str) -> None:
        i = OBSERVABLE.idx(name)
        ax.plot(days, log.y[:, i], lw=1.2)
        if name in bands:
            ax.axhspan(bands[name].lo, bands[name].hi, color="tab:green", alpha=0.12, lw=0)
        ax.set_ylabel(f"{name} [{y_units[i]}]")

    def control(ax, name: str) -> None:
        i = CONTROL.idx(name)
        ax.step(step_days, log.u[:, i], where="post", lw=1.2, color="tab:orange")
        for limit in (u_lo[i], u_hi[i]):
            ax.axhline(limit, ls="--", lw=0.8, color="gray")
        ax.set_ylabel(f"{name} [{u_units[i]}]")

    for ax, name in zip(axes[:, 0], ("dry_weight", "indoor_temp", "co2_ppm"), strict=True):
        observable(ax, name)
    for ax, name in zip(axes[:, 1], ("heating", "ventilation", "co2_supply"), strict=True):
        control(ax, name)
    observable(axes[0, 2], "rh")

    weather = axes[1, 2]
    weather.plot(step_days, log.v[:, EXOGENOUS.idx("rad")], color="tab:olive", lw=1.0)
    weather.set_ylabel("radiation [W/m2]")
    twin = weather.twinx()
    twin.plot(step_days, log.v[:, EXOGENOUS.idx("out_temp")], color="tab:blue", lw=1.0)
    twin.set_ylabel("outdoor temp [degC]")

    reward = axes[2, 2]
    for key, sign in (("revenue", 1), ("energy_cost", -1), ("co2_cost", -1), ("penalty", -1)):
        series = sign * np.cumsum([step[key] for step in log.info])
        reward.plot(step_days, series, lw=1.2, label=key)
    reward.plot(step_days, np.cumsum(log.reward), color="black", lw=1.6, label="return")
    reward.set_ylabel("cumulative [EUR/m2]")
    reward.legend(fontsize=8, loc="upper left")

    for ax in axes[2]:
        ax.set_xlabel("day")
    fig.suptitle(title or f"{n} steps, return {log.total_return:.3f}")
    fig.tight_layout()
    return fig
