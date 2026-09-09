"""Weather: the exogenous input ``v`` an episode runs on.

Three concerns, kept apart: a :class:`WeatherSeries` is *data* (one loaded trace in model units), a
:class:`WeatherScenario` names *which slice* an episode gets, and a sampler decides *which scenario*
this episode is. The repository caches series across episodes; the sampler runs once per reset.

``start_day`` is a **day of year**, not an offset into the file. Sources begin on different dates --
the standard Bleiswijk trace starts on 10 January while KNMI years start on 1 January, so a
file-relative offset silently shifts the season by days when you switch source. Each series carries
its own epoch and does that arithmetic itself.

Any weather source plugs in through one protocol: a :data:`WeatherLoader`, a zero-argument callable
that returns a :class:`WeatherSeries`, registered under a name in a :class:`WeatherRepository`. The
packaged trace, the CSV reader and the KNMI files are all loaders; a reader for another format, a
database query or a generator is one function that ends in :meth:`WeatherSeries.from_channels`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .units import co2_ppm_to_density, rh_to_vapour_density
from .variables.exogenous import EXOGENOUS

SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True, slots=True)
class WeatherScenario:
    """Which weather an episode runs on: a named source and a start day of year."""

    source: str
    start_day: float
    """Day of the year the season starts, 1-based (1.0 = 1 January)."""

    def __post_init__(self) -> None:
        if not 1.0 <= self.start_day <= 367.0:
            raise ValueError(f"start_day must be a day of year in [1, 367], got {self.start_day}")


@dataclass(frozen=True, slots=True)
class WeatherSeries:
    """One loaded weather trace, already in model units and EXOGENOUS order.

    ``values`` is ``(4, N)``: rad [W/m2], out_co2 [kg/m3], out_temp [degC], out_vapor [kg/m3]. The
    trace keeps its own native resolution ``dt``; episodes sample it at the control step, so the
    control step can change without touching the data.
    """

    name: str
    values: NDArray[np.float64]
    """``(4, N)`` in EXOGENOUS order. Prefer :meth:`from_channels`, which derives the order from the
    group rather than asking you to remember it."""
    dt: float
    """The trace's own sample period in seconds (300 s for the reference CSVs)."""
    epoch_day: float
    """Day of year of the first sample, 1-based."""

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[0] != EXOGENOUS.size:
            raise ValueError(
                f"values must be ({EXOGENOUS.size}, N) in EXOGENOUS order, got {self.values.shape}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        # The shape check above cannot tell a mis-ordered trace from a correct one, and a swapped
        # row produces a physically nonsensical episode that still runs. Checking each row against
        # its own declared physical bounds catches the realistic cases without inventing thresholds:
        # outdoor temperature goes below zero, so temperature in the radiation row fails `rad >= 0`.
        lo, hi = EXOGENOUS.bounds()
        offending = (self.values < lo[:, None]).any(axis=1) | (self.values > hi[:, None]).any(
            axis=1
        )
        if offending.any():
            names = [EXOGENOUS.names[i] for i in np.flatnonzero(offending)]
            raise ValueError(
                f"{self.name}: values fall outside the physical bounds of {names}. "
                f"Are the rows in EXOGENOUS order {EXOGENOUS.names}?"
            )

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[1])

    @property
    def span_days(self) -> float:
        return self.n_samples * self.dt / SECONDS_PER_DAY

    @property
    def last_day(self) -> float:
        """Day of year of the final sample."""
        return self.epoch_day + self.span_days

    def episode_slice(
        self, start_day: float, n_steps: int, step_dt: float, lookahead: int = 0
    ) -> NDArray[np.float64]:
        """The weather for one episode, resampled to the control step.

        Returns ``(4, n_steps + lookahead)``: the value at each control step, plus ``lookahead``
        further steps so an observation can include a forecast window. Raises if the requested
        window runs past the end of the trace; silently clamping would leave an episode running on
        frozen weather that looks perfectly plausible.
        """
        offset_s = (start_day - self.epoch_day) * SECONDS_PER_DAY
        if offset_s < 0:
            raise ValueError(
                f"{self.name}: start_day {start_day} precedes the trace's first day {self.epoch_day}"
            )
        total = n_steps + lookahead
        rows = np.rint((offset_s + np.arange(total) * step_dt) / self.dt).astype(int)
        if rows[-1] >= self.n_samples:
            needed = (rows[-1] + 1 - self.n_samples) * self.dt / SECONDS_PER_DAY
            raise ValueError(
                f"{self.name}: episode of {n_steps} steps from day {start_day} runs "
                f"{needed:.1f} days past the end of the trace (ends day {self.last_day:.1f})"
            )
        return self.values[:, rows]

    def steps_available(self, start_day: float, step_dt: float) -> int:
        """How many control steps of ``step_dt`` fit between ``start_day`` and the end of the trace."""
        offset_s = (start_day - self.epoch_day) * SECONDS_PER_DAY
        if offset_s < 0:
            raise ValueError(
                f"{self.name}: start_day {start_day} precedes the trace's first day {self.epoch_day}"
            )
        last_s = (self.n_samples - 1) * self.dt
        return int(np.floor((last_s - offset_s) / step_dt)) + 1 if last_s >= offset_s else 0

    def viable_start_days(
        self, n_steps: int, step_dt: float, lookahead: int = 0
    ) -> NDArray[np.float64]:
        """Whole days from which a full episode (plus lookahead) fits inside this trace."""
        span_s = (n_steps + lookahead - 1) * step_dt
        last = self.epoch_day + (self.n_samples - 1) * self.dt / SECONDS_PER_DAY
        latest = last - span_s / SECONDS_PER_DAY
        return np.arange(np.ceil(self.epoch_day), np.floor(latest) + 1, dtype=np.float64)

    @classmethod
    def from_channels(
        cls, name: str, dt: float, epoch_day: float, **channels: NDArray[np.float64]
    ) -> "WeatherSeries":
        """Build a series from **named** channels, so the row order cannot be got wrong.

        Mirrors :meth:`VarGroup.pack`: you supply ``rad=``, ``out_co2=``, ``out_temp=`` and
        ``out_vapor=``, and the group's declaration order decides the layout. Constructing
        ``values`` by hand is still allowed, but then the ordering is yours to get right.
        """
        missing = set(EXOGENOUS.names) - channels.keys()
        extra = channels.keys() - set(EXOGENOUS.names)
        if missing or extra:
            raise ValueError(
                f"channels: missing={sorted(missing)} extra={sorted(extra)}; "
                f"expected exactly {EXOGENOUS.names}"
            )
        values = np.vstack([np.asarray(channels[n], dtype=np.float64) for n in EXOGENOUS.names])
        return cls(name=name, values=values, dt=dt, epoch_day=epoch_day)

    @classmethod
    def synthetic(
        cls,
        days: float = 120.0,
        dt: float = 300.0,
        name: str = "synthetic",
        epoch_day: float = 1.0,
    ) -> "WeatherSeries":
        """A smooth day/night trace for tests and examples; no data files required.

        An *alternative constructor*, like ``ParameterSet.from_defaults``: it builds a series rather
        than acting on one, which is why it takes ``cls`` and not ``self``. Its ``dt``, ``name`` and
        ``epoch_day`` are its own parameters: a synthetic trace has no file to read them from, so
        it declares them, with defaults that make ``WeatherSeries.synthetic()`` usable bare.

        Radiation is a half-sine over daylight hours and exactly zero at night; temperature follows a
        daily cycle; outdoor CO2 and vapour are constant. Deterministic, so tests are reproducible.
        """
        n = round(days * SECONDS_PER_DAY / dt)
        t = np.arange(n) * dt
        time_of_day = (t % SECONDS_PER_DAY) / SECONDS_PER_DAY

        daylight = (time_of_day > 0.25) & (time_of_day < 0.75)
        rad = np.where(daylight, 800.0 * np.sin(np.pi * (time_of_day - 0.25) / 0.5), 0.0)
        out_temp = 10.0 + 8.0 * np.sin(2.0 * np.pi * (time_of_day - 0.25))
        out_co2 = np.full(n, 7.5e-4)
        out_vapor = np.full(n, 0.008)

        return cls.from_channels(
            name=name,
            dt=dt,
            epoch_day=epoch_day,
            rad=rad,
            out_co2=out_co2,
            out_temp=out_temp,
            out_vapor=out_vapor,
        )

    @classmethod
    def from_csv(
        cls, path: str | Path, epoch_day: float, name: str | None = None
    ) -> "WeatherSeries":
        """Load a headerless weather CSV: ``time, Io, To, RH, Vo, CO2ppm``.

        The file records what instruments read: radiation [W/m2], temperature [degC], relative
        humidity [%], wind [m/s], CO2 [ppm]. Humidity and CO2 are converted here to the densities
        the model consumes, using the same formulas ``measurement`` inverts, so the two directions
        cannot drift. Wind is dropped: the four-state model has no wind term.

        ``epoch_day`` must be supplied because the ``time`` column counts seconds from the file's
        own start, not a calendar date, so the file alone cannot say which day of the year it
        begins on.
        """
        raw = np.loadtxt(path, delimiter=",")
        if raw.ndim != 2 or raw.shape[1] < 6:
            raise ValueError(f"{path}: expected 6 columns (time, Io, To, RH, Vo, CO2ppm)")

        times = raw[:, 0]
        steps = np.diff(times)
        dt = float(np.round(np.mean(steps)))
        if not np.allclose(steps, dt, rtol=0, atol=1e-6):
            raise ValueError(f"{path}: time column is not uniformly spaced (mean step {dt} s)")

        out_temp = raw[:, 2]
        rad = raw[:, 1].copy()
        # clamp night-time sensor noise to exact zero: the photosynthesis term has a distinct
        # branch at rad == 0, and a residual 1e-9 would step around it
        rad[rad < 1e-6] = 0.0
        out_co2 = co2_ppm_to_density(out_temp, raw[:, 5])
        out_vapor = rh_to_vapour_density(out_temp, raw[:, 3])

        return cls.from_channels(
            name=name or Path(path).stem,
            dt=dt,
            epoch_day=epoch_day,
            rad=rad,
            out_co2=out_co2,
            out_temp=out_temp,
            out_vapor=out_vapor,
        )


def write_csv(
    path: str | Path,
    *,
    time_s: NDArray[np.float64],
    rad: NDArray[np.float64],
    temp: NDArray[np.float64],
    rh: NDArray[np.float64],
    co2_ppm: NDArray[np.float64] | float,
    wind: NDArray[np.float64] | float = 0.0,
) -> Path:
    """Write a trace in the package's CSV format, the inverse of :meth:`WeatherSeries.from_csv`.

    Channels are what instruments read: ``time_s`` seconds from the first row (uniformly spaced),
    ``rad`` W/m2, ``temp`` degC, ``rh`` %, ``co2_ppm`` ppm and ``wind`` m/s; the last two may be a
    single constant. Use it to share a trace as a file; a reader that stays in Python can skip the
    file and return :meth:`WeatherSeries.from_channels` directly.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must be a 1-D array with at least two rows")
    steps = np.diff(time_s)
    if not np.allclose(steps, steps[0], rtol=0, atol=1e-6) or steps[0] <= 0:
        raise ValueError("time_s must be uniformly spaced and increasing")
    columns = [time_s, rad, temp, rh, wind, co2_ppm]
    table = np.column_stack(
        [np.broadcast_to(np.asarray(c, dtype=np.float64), time_s.shape) for c in columns]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, table, delimiter=",", fmt=["%.0f", "%.4f", "%.3f", "%.2f", "%.2f", "%.1f"])
    return path


WeatherLoader = Callable[[], WeatherSeries]
"""The protocol every weather source satisfies: call it, get a :class:`WeatherSeries`.

Nothing else is assumed: not a file, not a format, not a resolution. ``functools.partial`` turns
any function with arguments (a path, a year, a station) into a loader.
"""


class WeatherRepository:
    """Named weather sources, loaded on demand and cached for the life of the process.

    Sources register as :data:`WeatherLoader` callables rather than paths, so a synthetic or
    in-memory source plugs in like a file-backed one. A loaded series takes the name it was
    registered under, so logs and error messages use the caller's vocabulary.
    """

    def __init__(self, sources: Mapping[str, WeatherLoader] | None = None) -> None:
        self._loaders: dict[str, WeatherLoader] = dict(sources or {})
        self._cache: dict[str, WeatherSeries] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._loaders)

    def register(self, name: str, loader: WeatherLoader) -> None:
        if name in self._loaders:
            raise ValueError(f"weather source {name!r} is already registered")
        self._loaders[name] = loader

    def load(self, name: str) -> WeatherSeries:
        if name not in self._cache:
            if name not in self._loaders:
                raise KeyError(f"unknown weather source {name!r}; registered: {self.names}")
            series = self._loaders[name]()
            if not isinstance(series, WeatherSeries):
                raise TypeError(
                    f"weather loader {name!r} returned {type(series).__name__}, not a WeatherSeries"
                )
            self._cache[name] = series if series.name == name else replace(series, name=name)
        return self._cache[name]


SYNTHETIC = "synthetic"
BLEISWIJK_2014 = "bleiswijk_2014"

# The first sample of the packaged trace is 10 January 2014. The CSV cannot say so itself, since its
# time column counts seconds from the file start, so the date is recovered from the datenums in
# the original MATLAB artifact and supplied here. See data/SOURCE.md.
_BLEISWIJK_EPOCH_DAY = 10.0

BENCHMARK_SCENARIO = WeatherScenario(BLEISWIJK_2014, start_day=40.0)
"""The season the published RL/MPC benchmark runs on: 40 days from 9 February 2014."""


def bleiswijk_2014() -> WeatherSeries:
    """The packaged 2014 Bleiswijk trace (WUR Glas, NL). See ``data/SOURCE.md`` for attribution."""
    resource = resources.files("lettuce_greenhouse_gym.data") / "outdoorWeatherWurGlas2014.csv"
    with resources.as_file(resource) as path:
        return WeatherSeries.from_csv(path, epoch_day=_BLEISWIJK_EPOCH_DAY, name=BLEISWIJK_2014)


def default_repository() -> WeatherRepository:
    """The sources shipped with the package: real measured weather, plus a synthetic fallback."""
    return WeatherRepository({SYNTHETIC: WeatherSeries.synthetic, BLEISWIJK_2014: bleiswijk_2014})


def _apply_options(
    scenario: WeatherScenario, options: Mapping[str, Any] | None
) -> WeatherScenario:
    """Let ``env.reset(options=...)`` pin the weather, overriding whatever the sampler chose."""
    if not options:
        return scenario
    if isinstance(options.get("weather"), WeatherScenario):
        return options["weather"]
    if "start_day" in options:
        return WeatherScenario(scenario.source, float(options["start_day"]))
    return scenario


class WeatherPerturbation:
    """Two hook points between the weather trace and the parties that consume it.

    The environment owns the points, not what happens at them: both methods default to the
    identity, and a study overrides one or both. Each receives the env's RNG, so a seed still
    determines the whole episode.

    :meth:`realise` shapes what the **plant** experiences. It is called once per episode, at reset,
    with the trace from the season's first step to the end of the trace at the control step,
    ``(4, n)`` in EXOGENOUS order, and returns the weather the plant runs on, for example a
    stochastic realisation around the measured trace.

    :meth:`forecast` shapes what a **controller** is told. It is called on every forecast the env
    hands out (the observation's weather window and :meth:`LettuceGreenhouseEnv.weather_forecast`)
    with the realised weather from ``step_index`` onward, ``(4, n)``, and returns what the
    controller should see, for example noise growing with the lead time. The plant is unaffected.

    Returned arrays must keep their shape and stay within ``EXOGENOUS.bounds()``; the env checks
    both and raises, rather than running a season on weather that does not exist.
    """

    def realise(
        self, rng: np.random.Generator, weather: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        return weather

    def forecast(
        self, rng: np.random.Generator, step_index: int, weather: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        return weather


class WeatherSampler(ABC):
    """Chooses the weather scenario for one episode. Receives the env's RNG; never owns one."""

    @abstractmethod
    def sample(
        self, rng: np.random.Generator, options: Mapping[str, Any] | None = None
    ) -> WeatherScenario:
        """Return the scenario for the episode about to start."""


class FixedWeatherSampler(WeatherSampler):
    """Always the same scenario, which is what reproducing a published experiment needs."""

    def __init__(self, scenario: WeatherScenario) -> None:
        self._scenario = scenario

    def sample(self, rng, options=None) -> WeatherScenario:
        return _apply_options(self._scenario, options)


def _sources(source: str | Sequence[str]) -> tuple[str, ...]:
    sources = (source,) if isinstance(source, str) else tuple(source)
    if not sources:
        raise ValueError("at least one weather source is required")
    return sources


def _days(start_days: Sequence[float]) -> NDArray[np.float64]:
    days = np.asarray(start_days, dtype=np.float64)
    if days.size == 0:
        raise ValueError("start_days must not be empty")
    return days


class RandomWeatherSampler(WeatherSampler):
    """A source and a start day drawn uniformly from explicit lists.

    The day list is supplied rather than derived, so a train/test split is *visible* at the call
    site: build one sampler over the training days and another over the held-out ones (see
    :func:`split_start_days`). Several sources (e.g. one trace per year) share the day list; the
    source is drawn first, then the day.
    """

    def __init__(self, source: str | Sequence[str], start_days: Sequence[float]) -> None:
        self._sources = _sources(source)
        self._start_days = _days(start_days)

    @property
    def sources(self) -> tuple[str, ...]:
        return self._sources

    @property
    def start_days(self) -> tuple[float, ...]:
        return tuple(float(d) for d in self._start_days)

    def sample(self, rng, options=None) -> WeatherScenario:
        # a single source draws nothing for itself, so seeded single-source runs are unchanged
        source = (
            self._sources[rng.choice(len(self._sources))]
            if len(self._sources) > 1
            else self._sources[0]
        )
        chosen = WeatherScenario(source, float(rng.choice(self._start_days)))
        return _apply_options(chosen, options)


class CyclingWeatherSampler(WeatherSampler):
    """Walks every (source, start day) pair in order, repeating: deterministic evaluation.

    Ignores the RNG on purpose: an evaluation run should visit every held-out season exactly, not
    sample them. Sources are visited one after another, each over all days. Call :meth:`reset` to
    restart the cycle.
    """

    def __init__(self, source: str | Sequence[str], start_days: Sequence[float]) -> None:
        self._sources = _sources(source)
        self._start_days = _days(start_days)
        self._cycle = [
            WeatherScenario(s, float(d)) for s in self._sources for d in self._start_days
        ]
        self._i = 0

    @property
    def sources(self) -> tuple[str, ...]:
        return self._sources

    @property
    def start_days(self) -> tuple[float, ...]:
        return tuple(float(d) for d in self._start_days)

    def reset(self) -> None:
        self._i = 0

    def sample(self, rng, options=None) -> WeatherScenario:
        chosen = self._cycle[self._i % len(self._cycle)]
        self._i += 1
        return _apply_options(chosen, options)


def split_start_days(
    series: WeatherSeries,
    n_steps: int,
    step_dt: float,
    rng: np.random.Generator,
    train_fraction: float = 0.8,
    lookahead: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Disjoint train / test start days over a single trace.

    One year of weather already contains many distinct seasons, so held-out *start days* give a
    genuine generalisation test without needing several years of data. Shuffle once, cut, and keep
    the halves apart; the caller then builds one sampler per half, so which season went where is
    explicit rather than hidden behind a mode flag.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    days = series.viable_start_days(n_steps, step_dt, lookahead)
    if days.size < 2:
        raise ValueError(
            f"{series.name}: only {days.size} viable start day(s) for a {n_steps}-step episode; "
            "cannot split"
        )
    shuffled = days.copy()
    rng.shuffle(shuffled)
    cut = int(train_fraction * shuffled.size)
    cut = max(1, min(cut, shuffled.size - 1))  # both halves non-empty
    return np.sort(shuffled[:cut]), np.sort(shuffled[cut:])
