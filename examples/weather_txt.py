"""Bring your own weather, minimal version: a loader for a plain-text logger export.

The file this reads is the kind a weather station or a data portal hands out: a header line, then
one row per hour with a date, a time and a few columns of readings::

    date time temp_C rh_pct rad_Wm2
    2023-03-01 00:00 4.0 83 0
    2023-03-01 01:00 3.4 84 0

Every custom source is the same three moves this file makes: parse the readings, put them on a
uniform grid, and return ``WeatherSeries.from_channels``. The longer example ``weather_epw.py`` does
the same for EnergyPlus EPW files; ``docs/custom_weather.md`` walks through both.

    python examples/weather_txt.py station.txt    # load it and run the grower heuristic
    python examples/weather_txt.py                # the same on a generated 5-day file
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from lettuce_greenhouse_gym import (
    EnvConfig,
    FixedWeatherSampler,
    LettuceGreenhouseEnv,
    WeatherRepository,
    WeatherScenario,
    WeatherSeries,
)
from lettuce_greenhouse_gym.baselines import GrowerHeuristic, run_episode
from lettuce_greenhouse_gym.units import co2_ppm_to_density, rh_to_vapour_density


def load_txt(path: str | Path, *, dt: float = 300.0, co2_ppm: float = 400.0) -> WeatherSeries:
    """``station.txt`` as a ``WeatherSeries`` on a ``dt`` grid.

    The file has no CO2 column, so outdoor CO2 is the constant ``co2_ppm``. ``epoch_day`` is read
    off the first timestamp: 1 March is day 60 of a non-leap year.
    """
    path = Path(path)
    rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no data rows")
    stamps = [datetime.strptime(f"{r[0]} {r[1]}", "%Y-%m-%d %H:%M") for r in rows]
    seconds = np.array([(s - stamps[0]).total_seconds() for s in stamps])
    temp, rh, rad = (np.array([float(r[i]) for r in rows]) for i in (2, 3, 4))

    grid = np.arange(0.0, seconds[-1] + 1e-9, dt)
    temp, rh, rad = (np.interp(grid, seconds, c) for c in (temp, rh, rad))
    first = stamps[0]
    epoch_day = first.timetuple().tm_yday + (first.hour * 3600 + first.minute * 60) / 86400.0
    return WeatherSeries.from_channels(
        path.stem,
        dt,
        epoch_day,
        rad=np.maximum(rad, 0.0),
        out_temp=temp,
        out_vapor=rh_to_vapour_density(temp, rh),
        out_co2=co2_ppm_to_density(temp, np.full_like(temp, co2_ppm)),
    )


def write_demo_txt(path: Path, days: int = 5) -> Path:
    """A small file in the format above with a plausible diurnal cycle, starting 1 March 2023."""
    lines = ["date time temp_C rh_pct rad_Wm2"]
    for h in range(days * 24):
        t = datetime(2023, 3, 1) + timedelta(hours=h)
        hh = t.hour + 0.5
        temp = 8.0 + 5.0 * np.sin(np.pi * (hh - 9.0) / 12.0)
        rh = 75.0 - 10.0 * np.sin(np.pi * (hh - 9.0) / 12.0)
        rad = max(0.0, 400.0 * np.sin(np.pi * (hh - 7.0) / 11.0)) if 7.0 <= hh <= 18.0 else 0.0
        lines.append(f"{t:%Y-%m-%d %H:%M} {temp:.1f} {rh:.0f} {rad:.0f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str]) -> int:
    if argv:
        txt = Path(argv[0])
    else:
        txt = write_demo_txt(Path(tempfile.mkdtemp()) / "station.txt")
        print(f"no file given; wrote a 5-day demonstration file to {txt}")

    series = load_txt(txt)
    print(
        f"{series.name}: {series.n_samples} samples at {series.dt:.0f} s, "
        f"days {series.epoch_day:.2f}-{series.last_day:.2f} of the year"
    )
    repo = WeatherRepository({"station": lambda: series})
    episode_days = min(3.0, series.span_days - 1.0)
    env = LettuceGreenhouseEnv(
        EnvConfig(episode_days=episode_days),
        weather_repository=repo,
        weather_sampler=FixedWeatherSampler(WeatherScenario("station", series.epoch_day + 1.0)),
    )
    log = run_episode(env, GrowerHeuristic(), seed=0)
    print(f"grower heuristic over {episode_days:g} day(s): return {log.total_return:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
