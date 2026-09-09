"""KNMI hourly weather observations as weather traces for this environment.

The Royal Netherlands Meteorological Institute (KNMI) publishes hourly station observations under
CC BY 4.0. :func:`fetch_year` downloads one station-year, converts it to this package's CSV format
(``time, Io, To, RH, Vo, CO2ppm``, 5-minute rows by default), and writes a ``SOURCE.md`` with the
attribution the licence requires. Load the result with ``WeatherSeries.from_csv(path, epoch_day=1.0)``
or list it under ``weather.files`` in an experiment spec.

Two facts of the conversion to know: KNMI has no CO2 channel, so the CO2 column is a constant you
choose (default 400 ppm); and hourly values are linearly interpolated onto the output grid, which
starts at 00:00 on 1 January by holding the first observation, so ``epoch_day`` is exactly 1.0.
"""

import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .units import co2_ppm_to_density, rh_to_vapour_density
from .weather import SECONDS_PER_DAY, WeatherSeries
from .weather import write_csv as write_weather_csv

ENDPOINT = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
VARIABLES = "Q:T:U:FH"  # global radiation, temperature, relative humidity, wind speed
DEFAULT_STATION = 344  # Rotterdam Airport, the KNMI station nearest to Bleiswijk
ATTRIBUTION = "Source: Royal Netherlands Meteorological Institute (KNMI), licence CC BY 4.0."

_STATION_LINE = re.compile(r"^#\s+(\d{3})\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(.+?)\s*$")


def seconds_in_year(year: int) -> float:
    """Length of the calendar year in seconds."""
    return (date(year + 1, 1, 1) - date(year, 1, 1)).days * SECONDS_PER_DAY


@dataclass(frozen=True)
class HourlyRecords:
    """One station-year of hourly observations, in physical units, stamped at the end of each hour."""

    station: int
    year: int
    # e.g. "344  4.447  51.962  -4.30  Rotterdam Airport", from the response header
    station_line: str
    seconds: NDArray[np.float64]  # since 00:00 on 1 January, end of each hourly interval
    rad: NDArray[np.float64]  # W/m2
    temp: NDArray[np.float64]  # degC
    rh: NDArray[np.float64]  # %
    wind: NDArray[np.float64]  # m/s


def fetch_hourly(station: int, year: int, *, timeout: float = 60.0) -> str:
    """Download one station-year from KNMI's hourly-observations service, as returned (text)."""
    query = urllib.parse.urlencode(
        {"stns": station, "vars": VARIABLES, "start": f"{year}010101", "end": f"{year}123124"}
    ).encode()
    with urllib.request.urlopen(
        "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens", data=query, timeout=timeout
    ) as response:
        return response.read().decode("utf-8", errors="replace")


def _fill_gaps(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear interpolation over missing hours; edges take the nearest valid value."""
    ok = np.isfinite(values)
    if ok.all():
        return values
    if not ok.any():
        raise ValueError("a channel has no valid observations")
    idx = np.arange(values.size)
    return np.interp(idx, idx[ok], values[ok])


def parse_hourly(text: str) -> HourlyRecords:
    """Parse the service's text: ``#`` comment lines, then ``STN, YYYYMMDD, HH, Q, T, U, FH`` rows.

    Units are converted here: ``Q`` J/cm2 per hour -> W/m2, ``T`` 0.1 degC -> degC, ``FH`` 0.1 m/s ->
    m/s; ``U`` is already %. KNMI's ``HH`` is the hour that *ends* at HH (05 = 04:00-05:00 UT), so a
    row is stamped at HH:00. Missing values (blank fields) are interpolated linearly.
    """
    station_line = ""
    rows: list[
        tuple[int, int, int, float, float, float, float]
    ] = []  # STN, YYYYMMDD, HH, Q, T, U, FH
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            m = _STATION_LINE.match(line)
            if m and not station_line:
                station_line = "  ".join(m.groups())
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            raise ValueError(f"unexpected KNMI row: {line!r}")
        stn, ymd, hh = int(parts[0]), int(parts[1]), int(parts[2])
        obs = [float(p) if p else np.nan for p in parts[3:7]]
        rows.append((stn, ymd, hh, obs[0], obs[1], obs[2], obs[3]))
    if not rows:
        raise ValueError("no observation rows in the KNMI response")

    stations = {r[0] for r in rows}
    if len(stations) != 1:
        raise ValueError(f"expected one station, got {sorted(stations)}")
    year = rows[0][1] // 10000
    origin = datetime(year, 1, 1)
    seconds = np.array(
        [
            (datetime.strptime(str(ymd), "%Y%m%d") + timedelta(hours=hh) - origin).total_seconds()
            for _, ymd, hh, *_ in rows
        ]
    )
    q, t, u, fh = (np.array([r[3 + i] for r in rows]) for i in range(4))
    # the service returns rows past the requested end; keep the calendar year (last stamp = 1 Jan
    # 00:00 of the next year, which is the end of the year's final hour)
    keep = seconds <= seconds_in_year(year)
    seconds, q, t, u, fh = (a[keep] for a in (seconds, q, t, u, fh))
    return HourlyRecords(
        station=stations.pop(),
        year=year,
        station_line=station_line,
        seconds=seconds,
        rad=np.maximum(_fill_gaps(q) * 1e4 / 3600.0, 0.0),
        temp=_fill_gaps(t) / 10.0,
        rh=_fill_gaps(u),
        wind=_fill_gaps(fh) / 10.0,
    )


def _grid(records: HourlyRecords, dt: float) -> NDArray[np.float64]:
    """Uniform grid from 00:00 on 1 January to the last observation."""
    return np.arange(0.0, records.seconds[-1] + 1e-9, dt)


def _on_grid(records: HourlyRecords, dt: float) -> dict[str, NDArray[np.float64]]:
    grid = _grid(records, dt)
    return {
        name: np.interp(grid, records.seconds, getattr(records, name))
        for name in ("rad", "temp", "rh", "wind")
    } | {"time": grid}


def to_series(
    records: HourlyRecords, *, dt: float = 300.0, co2_ppm: float = 400.0, name: str | None = None
) -> WeatherSeries:
    """The station-year as a :class:`WeatherSeries` on a ``dt`` grid, ``epoch_day`` 1.0."""
    g = _on_grid(records, dt)
    return WeatherSeries.from_channels(
        name or f"knmi_{records.station}_{records.year}",
        dt,
        1.0,
        rad=g["rad"],
        out_co2=co2_ppm_to_density(g["temp"], np.full_like(g["temp"], co2_ppm)),
        out_temp=g["temp"],
        out_vapor=rh_to_vapour_density(g["temp"], g["rh"]),
    )


def write_csv(
    records: HourlyRecords, path: str | Path, *, dt: float = 300.0, co2_ppm: float = 400.0
) -> Path:
    """Write the package's headerless CSV (``time, Io, To, RH, Vo, CO2ppm``); reload with epoch_day 1.0."""
    g = _on_grid(records, dt)
    return write_weather_csv(
        path,
        time_s=g["time"],
        rad=g["rad"],
        temp=g["temp"],
        rh=g["rh"],
        co2_ppm=co2_ppm,
        wind=g["wind"],
    )


def write_source_note(
    out_dir: str | Path, entries: list[tuple[str, HourlyRecords]], *, dt: float, co2_ppm: float
) -> Path:
    """``SOURCE.md`` with the attribution CC BY 4.0 requires and the conversions applied."""
    lines = [
        "# KNMI weather traces",
        "",
        ATTRIBUTION,
        f"Fetched {date.today().isoformat()} from {ENDPOINT} (hourly observations, variables "
        f"{VARIABLES}). Please credit KNMI in work that uses these files.",
        "",
        "| file | station (STN, lon, lat, alt m, name) | year | rows | hours missing (interpolated) |",
        "|---|---|---|---|---|",
    ]
    for filename, r in entries:
        expected = round(r.seconds[-1] / 3600.0)
        missing = expected - r.seconds.size
        lines.append(
            f"| `{filename}` | {r.station_line or r.station} | {r.year} | "
            f"{_grid(r, dt).size} | {missing} |"
        )
    lines += [
        "",
        "Conversions: global radiation `Q` J/cm2 per hour -> W/m2 (x 10000/3600); temperature `T` "
        "0.1 degC -> degC; relative humidity `U` in %; wind `FH` 0.1 m/s -> m/s. Rows are stamped at "
        "the end of each hourly interval and linearly interpolated onto a "
        f"{dt:.0f} s grid starting 00:00 on 1 January (first hour held), so `epoch_day` is 1.0. KNMI "
        f"has no CO2 channel: the CO2 column is a constant {co2_ppm:g} ppm.",
        "",
    ]
    path = Path(out_dir) / "SOURCE.md"
    path.write_text("\n".join(lines))
    return path


def fetch_year(
    station: int,
    year: int,
    out_dir: str | Path,
    *,
    dt: float = 300.0,
    co2_ppm: float = 400.0,
    timeout: float = 60.0,
) -> Path:
    """Download, convert and write ``knmi_<station>_<year>.csv`` into ``out_dir``; returns its path."""
    records = parse_hourly(fetch_hourly(station, year, timeout=timeout))
    filename = f"knmi_{station}_{year}.csv"
    path = write_csv(records, Path(out_dir) / filename, dt=dt, co2_ppm=co2_ppm)
    write_source_note(out_dir, [(filename, records)], dt=dt, co2_ppm=co2_ppm)
    return path
