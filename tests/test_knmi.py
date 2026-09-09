"""KNMI conversion, on a recorded two-day response; the network call itself is not exercised here."""

import os
from pathlib import Path

import numpy as np
import pytest

from lettuce_greenhouse_gym.knmi import (
    fetch_hourly,
    parse_hourly,
    to_series,
    write_csv,
    write_source_note,
)
from lettuce_greenhouse_gym.variables.exogenous import EXOGENOUS
from lettuce_greenhouse_gym.weather import WeatherSeries

FIXTURE = (Path(__file__).parent / "data" / "knmi_344_2010_jan1-2.txt").read_text()


def test_parse_reads_rows_units_and_station():
    r = parse_hourly(FIXTURE)
    assert r.station == 344
    assert r.year == 2010
    assert r.station_line.startswith("344")
    assert "Rotterdam" in r.station_line
    # first row: 344,20100101,1,0,7,78,70 -> 01:00, 0 W/m2, 0.7 degC, 78 %, 7.0 m/s
    assert r.seconds[0] == 3600.0
    assert r.rad[0] == 0.0
    assert r.temp[0] == pytest.approx(0.7)
    assert r.rh[0] == 78.0
    assert r.wind[0] == pytest.approx(7.0)
    assert np.all(np.diff(r.seconds) == 3600.0)


def test_radiation_unit_conversion_and_gap_filling():
    text = (
        "# header\n"
        "344,20100101,    1,  100,   -62,   80,   50\n"
        "344,20100101,    2,     ,      ,     ,     \n"  # a missing hour
        "344,20100101,    3,  300,   -22,   90,   50\n"
    )
    r = parse_hourly(text)
    assert r.rad[0] == pytest.approx(100 * 1e4 / 3600)  # 277.78 W/m2
    assert r.temp[0] == pytest.approx(-6.2)
    assert r.rad[1] == pytest.approx((r.rad[0] + r.rad[2]) / 2)  # interpolated
    assert r.temp[1] == pytest.approx(-4.2)


def test_to_series_is_a_valid_5_minute_trace_from_day_one():
    s = to_series(parse_hourly(FIXTURE), dt=300.0, co2_ppm=420.0)
    assert s.dt == 300.0
    assert s.epoch_day == 1.0
    assert s.n_samples == int(parse_hourly(FIXTURE).seconds[-1] / 300) + 1
    lo, hi = EXOGENOUS.bounds()
    assert np.all(s.values >= lo[:, None] - 1e-12)
    assert np.all(s.values <= hi[:, None] + 1e-12)
    # the first hour is held: rows 0..12 carry the 01:00 observation
    np.testing.assert_allclose(s.values[2, :13], s.values[2, 0])


def test_csv_round_trip_matches_to_series(tmp_path):
    r = parse_hourly(FIXTURE)
    path = write_csv(r, tmp_path / "knmi_344_2010.csv", dt=300.0, co2_ppm=400.0)
    reloaded = WeatherSeries.from_csv(path, epoch_day=1.0)
    direct = to_series(r, dt=300.0, co2_ppm=400.0)
    assert reloaded.dt == 300.0
    np.testing.assert_allclose(
        reloaded.values, direct.values, rtol=1e-3, atol=1e-3
    )  # CSV precision
    assert reloaded.name == "knmi_344_2010"


def test_source_note_carries_the_attribution(tmp_path):
    r = parse_hourly(FIXTURE)
    note = write_source_note(tmp_path, [("knmi_344_2010.csv", r)], dt=300.0, co2_ppm=400.0)
    text = note.read_text()
    assert "KNMI" in text
    assert "CC BY 4.0" in text
    assert "400 ppm" in text
    assert "Rotterdam" in text


def test_parse_rejects_mixed_stations_and_empty_input():
    with pytest.raises(ValueError, match="one station"):
        parse_hourly("344,20100101,1,0,7,78,70\n260,20100101,1,0,7,78,70\n")
    with pytest.raises(ValueError, match="no observation rows"):
        parse_hourly("# only comments\n")


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("LETTUCE_NETWORK_TESTS"), reason="set LETTUCE_NETWORK_TESTS=1"
)
def test_fetch_hourly_returns_the_documented_format():
    text = fetch_hourly(344, 2010)
    r = parse_hourly(text)
    assert r.year == 2010
    assert r.seconds[-1] >= 364 * 86400


def test_rows_past_the_calendar_year_are_dropped():
    text = (
        "344,20101231,   23,  0,  10,  80,  10\n"
        "344,20101231,   24,  0,  10,  80,  10\n"  # 1 Jan 00:00: the year's last hour, kept
        "344,20110101,    1,  0,  10,  80,  10\n"  # next year: dropped
    )
    r = parse_hourly(text)
    assert r.seconds.size == 2
    assert r.seconds[-1] == 365 * 86400.0
