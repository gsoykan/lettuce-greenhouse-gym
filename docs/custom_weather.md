# Using your own weather data

The environment ships one weather trace, the 2014 Bleiswijk season the benchmark is defined on, and
a fetcher for KNMI station-years. Everything else, a text export from your own station, an
EnergyPlus EPW file, a NetCDF reanalysis, a database, a generator, goes through the same small
protocol. This guide walks through it once on a text file, then covers the choices and pitfalls
that apply to any format.

## The protocol in one paragraph

A weather source is a **callable that takes no arguments and returns a `WeatherSeries`**. A
`WeatherSeries` is four channels on a uniform time grid, in the model's units, plus two numbers
that place the grid in time: `dt`, the grid step in seconds, and `epoch_day`, the day of the year
of the first sample (1.0 is 00:00 on 1 January; 60.0 is 00:00 on 1 March of a non-leap year). The
four channels are outdoor global radiation in W/m², outdoor temperature in °C, outdoor CO₂ and
outdoor vapour as densities in kg/m³. You never compute the densities yourself:
`WeatherSeries.from_channels` takes radiation, temperature, and whatever `rh_to_vapour_density` and
`co2_ppm_to_density` return from your relative humidity in % and CO₂ in ppm.

That is all the environment knows about weather. A loader is registered under a name in a
`WeatherRepository`, a sampler names it in a `WeatherScenario`, and each episode slices the series
at the control step from its start day. The repository calls the loader once and caches the result.

## Walkthrough: a text file from a station logger

Suppose a logger exported this, one row per hour:

```text
date time temp_C rh_pct rad_Wm2
2023-03-01 00:00 4.0 83 0
2023-03-01 01:00 3.4 84 0
2023-03-01 02:00 3.0 85 0
```

### Step 1: write the loader

Three moves: parse the readings, put them on a uniform grid, return `from_channels`. This is
`examples/weather_txt.py`, which you can run as is:

```python
from datetime import datetime
from pathlib import Path

import numpy as np

from lettuce_greenhouse_gym import WeatherSeries
from lettuce_greenhouse_gym.units import co2_ppm_to_density, rh_to_vapour_density


def load_txt(path, *, dt=300.0, co2_ppm=400.0) -> WeatherSeries:
    path = Path(path)
    rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
    stamps = [datetime.strptime(f"{r[0]} {r[1]}", "%Y-%m-%d %H:%M") for r in rows]
    seconds = np.array([(s - stamps[0]).total_seconds() for s in stamps])
    temp, rh, rad = (np.array([float(r[i]) for r in rows]) for i in (2, 3, 4))

    grid = np.arange(0.0, seconds[-1] + 1e-9, dt)                      # a uniform grid
    temp, rh, rad = (np.interp(grid, seconds, c) for c in (temp, rh, rad))
    first = stamps[0]
    epoch_day = first.timetuple().tm_yday + (first.hour * 3600 + first.minute * 60) / 86400

    return WeatherSeries.from_channels(
        path.stem, dt, epoch_day,
        rad=np.maximum(rad, 0.0),
        out_temp=temp,
        out_vapor=rh_to_vapour_density(temp, rh),
        out_co2=co2_ppm_to_density(temp, np.full_like(temp, co2_ppm)),  # no CO2 sensor: a constant
    )
```

The decisions in it are the ones the file cannot make for you: the grid step, what to do about a
channel you do not measure, and the calendar date of the first row. Everything else is bookkeeping.

### Step 2: point the environment at it

Pick whichever of the three routes fits how you work.

**From Python.** Register the function under a name and name it in the sampler:

```python
from functools import partial

from lettuce_greenhouse_gym import (
    EnvConfig, FixedWeatherSampler, LettuceGreenhouseEnv, WeatherRepository, WeatherScenario,
)
from my_weather import load_txt

env = LettuceGreenhouseEnv(
    EnvConfig(episode_days=3.0),
    weather_repository=WeatherRepository({"station": partial(load_txt, "station.txt")}),
    weather_sampler=FixedWeatherSampler(WeatherScenario("station", start_day=61.0)),  # 2 March
)
```

Any sampler works: `RandomWeatherSampler("station", start_days=[61.0, 62.0])` draws a season start
per episode, and `split_start_days(series, ...)` gives disjoint training and evaluation days.

**From an experiment spec**, which is what the command line and training use. The spec names the
callable and its keyword arguments; the environment is rebuilt from the spec in every process that
needs it, including subprocess training workers and a saved run being reloaded:

```yaml
env: {episode_days: 3.0}
weather:
  source: station
  start_day: 61.0
  loaders:
    station: {target: my_weather.py:load_txt, kwargs: {path: station.txt}}
```

```bash
lettuce-gym baselines --config spec.yaml --controllers heuristic
lettuce-gym train --config spec.yaml
```

`target` is `path/to/script.py:function` for a script, or `package.module:function` for anything
importable, with attribute chains allowed (`package.module:Reader.load`). A relative script path
and a `path` entry in `kwargs` are taken relative to the YAML file; other keyword arguments pass
through verbatim. Note what this implies: a spec with a `loaders` block runs the code it names when
the environment is built, so treat a spec from someone else as you would treat their script.

**As a file, once.** If you would rather not carry a loader around, convert once and let the
package read the result natively. `lettuce_greenhouse_gym.weather.write_csv` writes the package's
CSV from physical channels:

```python
from lettuce_greenhouse_gym.weather import write_csv

write_csv("station.csv", time_s=grid, rad=rad, temp=temp, rh=rh, co2_ppm=400.0)
```

and the spec then needs no code at all:

```yaml
weather:
  source: station
  start_day: 61.0
  files:
    station: {path: station.csv, epoch_day: 60.0}
```

The CSV is a headerless table with six columns; `WeatherSeries.from_csv(path, epoch_day)` reads it:

| column | meaning | unit |
|---|---|---|
| `time` | seconds from the first row, one uniform step | s |
| `Io` | outdoor global radiation | W/m² |
| `To` | outdoor air temperature | °C |
| `RH` | outdoor relative humidity | % |
| `Vo` | wind speed (read and ignored: the model has no wind term) | m/s |
| `CO2ppm` | outdoor CO₂ | ppm |

The file does not know its own calendar date, which is why `epoch_day` travels next to it.

## Other formats

The loader changes; nothing after it does.

- **EnergyPlus EPW** (hourly files for thousands of sites worldwide): `examples/weather_epw.py`
  is a complete loader. EPW stamps each row with the hour that *ends* at the given hour, uses
  sentinel values such as 999 for a missing reading, and reports radiation in Wh/m² per hour, which
  is numerically the mean W/m² of that hour. The example handles all three.
- **KNMI station-years**: `lettuce-gym weather fetch-knmi` downloads and converts them to the CSV
  above (see the README), so they go in under `files`. The conversion module,
  `lettuce_greenhouse_gym.knmi`, is itself a loader written the same way and a good model for
  another national weather service.
- **NetCDF, GRIB, a database, an API**: read with whatever library you use, produce the arrays, and
  finish with `from_channels`. Nothing in the package needs to know how the arrays were obtained.
- **A generator**: `WeatherSeries.synthetic` is the reference; a loader that returns a series built
  from formulas is as valid as one that reads a file.

## Choices you will have to make

**Grid step.** Episodes sample the series at the control step (1800 s in the benchmark) by picking
the nearest grid sample, without interpolating. Hourly data left at `dt=3600` therefore gives
half-hour steps that repeat each hour's value twice. Interpolating onto a finer grid in the loader,
as both examples do with `dt=300`, gives smooth forcing. There is no single right answer, but be
aware which one you chose.

**Missing channels.** The van Henten model needs outdoor CO₂ and vapour. Few stations measure CO₂:
a constant near 400 ppm is what the KNMI conversion and both examples use, and what the reference
literature does. Vapour is derived from relative humidity; if you have dew point instead, convert
to RH first. Wind is not used.

**The first day.** `epoch_day` anchors the file to the calendar, and `start_day` in a scenario is a
day of the year, not an offset into the file. That is what lets one spec run a season "from
9 February" on traces that begin on different dates. Get it from the first timestamp, and remember
that day-of-year counts from 1.

**Several years.** One trace per year, each registered under its own name, and a spec with
`source: [y2009, y2010, ...]` draws a year per episode at a fixed `start_day`. `eval_source` and
`eval_start_days` let evaluation use other traces or days. The README's *Weather data* section
shows the YAML. Any other rule for choosing a season, pairing each trace with its own day, say, is a
`WeatherSampler` subclass returning a `WeatherScenario` per episode; `weather.sampler` names it in a
spec the way `loaders` names a reader.

## Changing the weather after loading

Loading is where a trace enters; two hook points on the environment are where a study reshapes it,
and both are the identity unless you say otherwise. Subclass `WeatherPerturbation`:

- `realise(rng, weather, dt)` decides what the **plant** experiences. It runs once per episode with
  the trace from the season's first step to the end of the trace at the control step, shape
  `(4, n)`, and returns the weather the season runs on. A stochastic realisation around the measured
  trace, a warmer year, a cloudier week all go here.
- `forecast(rng, step_index, weather, dt)` decides what a **controller** is told. It runs on every
  forecast the environment hands out, the observation's weather window and `weather_forecast(n)`
  alike, with the realised weather from the current step onward, and returns what the controller
  sees. Forecast error growing with lead time goes here; the plant never sees it.

```python
class ForecastNoise(WeatherPerturbation):
    def __init__(self, sigma_per_step):
        self.sigma = np.asarray(sigma_per_step)          # one value per channel, per step of lead time

    def forecast(self, rng, step_index, weather, dt):
        lead = np.arange(1, weather.shape[1] + 1)
        noisy = weather + rng.normal(size=weather.shape) * self.sigma[:, None] * np.sqrt(lead)
        noisy[0] = np.maximum(noisy[0], 0.0)              # radiation stays non-negative
        return noisy
```

Both hooks receive the environment's RNG, so one seed still fixes the whole episode. Both must
return an array of the same shape whose values lie within the physical bounds, or the environment
raises rather than run a season on weather that cannot exist. Pass an instance as
`weather_perturbation=` to the environment, or name it in a spec under `weather.perturbation` with
the same `target` and `kwargs` form as a loader.

## What the loader will refuse

`WeatherSeries` validates on construction, so mistakes surface when the loader runs, not as a
plausible-looking episode with nonsense weather:

- **Channel values outside their physical bounds** raise, naming the channel. The usual cause is
  columns in the wrong order (temperature in the radiation slot goes negative at night).
- **A non-uniform time column** in a CSV raises, since the grid step would otherwise be silently
  averaged. Resample in the loader rather than deleting rows.
- **A loader that returns something other than a `WeatherSeries`** raises a `TypeError` from the
  repository.
- **An episode that runs past the end of the trace** raises at `reset`, with the number of days
  missing. Check `series.viable_start_days(n_steps, dt)` when choosing start days.

A returned series takes the name it was registered under, so `info["weather_source"]` and error
messages use the spec's vocabulary rather than the loader's.
