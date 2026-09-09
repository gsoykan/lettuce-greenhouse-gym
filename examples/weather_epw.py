"""Bring your own weather: a loader for EnergyPlus EPW files.

A weather source is one function that returns a ``WeatherSeries``: read whatever you have, convert
to W/m2, degC, % and ppm, put the channels on a uniform grid, and call
``WeatherSeries.from_channels``. This script does that for EPW, the hourly format EnergyPlus
publishes for thousands of sites (https://energyplus.net/weather), so a season can run on weather
from anywhere in the world.

Three ways to use it::

    python examples/weather_epw.py site.epw    # load it, run the grower heuristic for a season
    python examples/weather_epw.py             # the same on a small demonstration file

    # in an experiment spec (paths relative to the YAML):
    weather:
      source: site
      start_day: 40.0
      loaders:
        site: {target: examples/weather_epw.py:load_epw, kwargs: {path: site.epw}}

The pattern is the whole point; the format is incidental. A database query or a generator is the
same function without the file.
"""

import sys
import tempfile
from datetime import date, timedelta
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

# EPW data records (EnergyPlus Auxiliary Programs, "Weather Data Period" fields), 0-based columns
YEAR, MONTH, DAY, HOUR = 0, 1, 2, 3
DRY_BULB, RELATIVE_HUMIDITY, GLOBAL_HORIZONTAL, WIND_SPEED = 6, 8, 13, 21
# the sentinels EPW uses for a missing value in those columns
MISSING = {DRY_BULB: 99.9, RELATIVE_HUMIDITY: 999.0, GLOBAL_HORIZONTAL: 9999.0, WIND_SPEED: 999.0}


def _fill(values: np.ndarray) -> np.ndarray:
    """Linear interpolation over missing hours; edges take the nearest valid value."""
    ok = np.isfinite(values)
    if not ok.any():
        raise ValueError("a channel has no valid observations")
    idx = np.arange(values.size)
    return np.interp(idx, idx[ok], values[ok])


def load_epw(path: str | Path, *, dt: float = 300.0, co2_ppm: float = 400.0) -> WeatherSeries:
    """An EPW file as a ``WeatherSeries`` on a ``dt`` grid.

    EPW rows are hourly and stamped with the hour that *ends* at ``HOUR`` (1-24), so a row is placed
    at HOUR:00 and the hourly values are interpolated onto the grid. Global horizontal radiation is
    in Wh/m2 per hour, numerically the mean W/m2 of that hour. EPW has no CO2 channel, so the
    outdoor CO2 is the constant ``co2_ppm``. ``epoch_day`` is the day of year of the first row.
    """
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split(",")
        # header records (LOCATION, DESIGN CONDITIONS, ...) start with a word; data rows with a year
        if len(parts) >= 22 and parts[YEAR].strip().isdigit():
            rows.append(parts)
    if not rows:
        raise ValueError(f"{path}: no EPW data rows found")

    def column(index: int) -> np.ndarray:
        values = np.array([float(r[index]) for r in rows])
        values[values == MISSING[index]] = np.nan
        return _fill(values)

    stamps = np.array(
        [
            (date(int(r[YEAR]), int(r[MONTH]), int(r[DAY])).timetuple().tm_yday - 1) * 86400.0
            + int(r[HOUR]) * 3600.0
            for r in rows
        ]
    )
    grid = np.arange(stamps[0], stamps[-1] + 1e-9, dt)
    temp = np.interp(grid, stamps, column(DRY_BULB))
    rh = np.clip(np.interp(grid, stamps, column(RELATIVE_HUMIDITY)), 0.0, 100.0)
    rad = np.maximum(np.interp(grid, stamps, column(GLOBAL_HORIZONTAL)), 0.0)
    return WeatherSeries.from_channels(
        path.stem,
        dt,
        1.0 + stamps[0] / 86400.0,
        rad=rad,
        out_co2=co2_ppm_to_density(temp, np.full_like(temp, co2_ppm)),
        out_temp=temp,
        out_vapor=rh_to_vapour_density(temp, rh),
    )


def write_demo_epw(path: Path, days: int = 3) -> Path:
    """A small EPW-shaped file with a plausible diurnal cycle, for trying the loader out."""
    last = date(2001, 1, 1) + timedelta(days=days - 1)
    header = [
        "LOCATION,Demo,-,-,Synthetic,000000,52.0,4.5,1.0,0.0",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,synthetic demonstration data written by examples/weather_epw.py",
        "COMMENTS 2,",
        f"DATA PERIODS,1,1,Data,Monday,1/1,{last.month}/{last.day}",
    ]
    lines = list(header)
    for d in range(days):
        day = date(2001, 1, 1) + timedelta(days=d)
        for hour in range(1, 25):
            h = hour - 0.5  # mid-hour, for the diurnal shapes
            temp = 6.0 + 4.0 * np.sin(np.pi * (h - 9.0) / 12.0)
            rh = 80.0 - 10.0 * np.sin(np.pi * (h - 9.0) / 12.0)
            ghi = max(0.0, 350.0 * np.sin(np.pi * (h - 7.0) / 10.0)) if 7.0 <= h <= 17.0 else 0.0
            fields = ["0"] * 35
            fields[YEAR], fields[MONTH], fields[DAY] = str(day.year), str(day.month), str(day.day)
            fields[HOUR] = str(hour)
            fields[4] = "0"
            fields[5] = "?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9"
            fields[DRY_BULB] = f"{temp:.1f}"
            fields[7] = f"{temp - 3.0:.1f}"
            fields[RELATIVE_HUMIDITY] = f"{rh:.0f}"
            fields[9] = "101300"
            fields[GLOBAL_HORIZONTAL] = f"{ghi:.0f}"
            fields[WIND_SPEED] = "3.0"
            lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str]) -> int:
    if argv:
        epw = Path(argv[0])
    else:
        epw = write_demo_epw(Path(tempfile.mkdtemp()) / "demo.epw")
        print(f"no file given; wrote a 3-day demonstration file to {epw}")

    series = load_epw(epw)
    print(
        f"{series.name}: {series.n_samples} samples at {series.dt:.0f} s, "
        f"days {series.epoch_day:.2f}-{series.last_day:.2f} of the year"
    )

    # Register the loader under a name and point the env at it. In a spec the `loaders` block does
    # exactly this; here it is two lines of Python.
    repo = WeatherRepository({"site": lambda: series})
    episode_days = min(1.0, series.span_days - 0.5)  # a full season needs 40 days of data
    env = LettuceGreenhouseEnv(
        EnvConfig(episode_days=episode_days),
        weather_sampler=FixedWeatherSampler(WeatherScenario("site", series.epoch_day)),
        weather_repository=repo,
    )
    log = run_episode(env, GrowerHeuristic(), seed=0)
    print(f"grower heuristic over {episode_days:g} day(s): return {log.total_return:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
