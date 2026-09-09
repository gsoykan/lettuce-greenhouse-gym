"""Tests for the weather layer: series indexing, the repository, and the samplers.

No data files are touched: everything runs on ``WeatherSeries.synthetic``, which is the point of
having it: the env can be built and tested before any loader exists.
"""

import numpy as np
import pytest

from lettuce_greenhouse_gym.units import co2_density_to_ppm, vapour_density_to_rh
from lettuce_greenhouse_gym.variables.exogenous import EXOGENOUS
from lettuce_greenhouse_gym.weather import (
    BENCHMARK_SCENARIO,
    BLEISWIJK_2014,
    SECONDS_PER_DAY,
    CyclingWeatherSampler,
    FixedWeatherSampler,
    RandomWeatherSampler,
    WeatherRepository,
    WeatherScenario,
    WeatherSeries,
    default_repository,
    split_start_days,
    write_csv,
)

N_STEPS = 1920  # 40 days at 1800 s, the benchmark season
STEP_DT = 1800.0


# ---- WeatherScenario -----------------------------------------------------------------------------
@pytest.mark.parametrize("start_day", [0.0, -1.0, 400.0], ids=["zero", "negative", "past_year"])
def test_scenario_rejects_a_start_day_outside_the_year(start_day):
    with pytest.raises(ValueError, match="day of year"):
        WeatherScenario("synthetic", start_day)


# ---- WeatherSeries -------------------------------------------------------------------------------
def test_synthetic_series_has_the_expected_shape_and_span():
    s = WeatherSeries.synthetic(days=120.0, dt=300.0)
    assert s.values.shape == (EXOGENOUS.size, 34560)
    assert s.span_days == pytest.approx(120.0)
    assert s.dt == 300.0


def test_series_rejects_values_that_are_not_in_exogenous_order():
    with pytest.raises(ValueError, match="EXOGENOUS order"):
        WeatherSeries(name="bad", values=np.zeros((3, 10)), dt=300.0, epoch_day=1.0)


def test_series_rejects_a_row_that_breaks_its_own_physical_bounds():
    """Partial protection against a mis-ordered trace.

    Outdoor temperature dips below zero, so temperature landing in the radiation row breaks
    ``rad >= 0``. It cannot catch every permutation (swapping the two densities is invisible this
    way, since both are non-negative and unbounded above), which is why ``from_channels`` exists.
    """
    temp = np.array([-1.5, 5.0, 10.0])
    zeros = np.zeros(3)
    with pytest.raises(ValueError, match="physical bounds"):
        WeatherSeries(
            name="swapped", values=np.vstack([temp, zeros, zeros, zeros]), dt=300.0, epoch_day=1.0
        )


def test_from_channels_does_not_care_what_order_the_keywords_arrive_in():
    """The group's declaration order decides the layout, so a caller cannot get it wrong."""
    rad, co2, temp, vapour = (
        np.full(4, 100.0),
        np.full(4, 7e-4),
        np.full(4, 12.0),
        np.full(4, 0.008),
    )
    a = WeatherSeries.from_channels(
        name="a", dt=300.0, epoch_day=1.0, rad=rad, out_co2=co2, out_temp=temp, out_vapor=vapour
    )
    b = WeatherSeries.from_channels(
        name="b", dt=300.0, epoch_day=1.0, out_vapor=vapour, out_temp=temp, out_co2=co2, rad=rad
    )
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.values[EXOGENOUS.idx("out_temp")], temp)


@pytest.mark.parametrize(
    "channels",
    [{"rad": np.zeros(3)}, {"rad": np.zeros(3), "wind": np.zeros(3)}],
    ids=["missing", "extra"],
)
def test_from_channels_rejects_an_incomplete_or_unknown_channel_set(channels):
    with pytest.raises(ValueError, match="expected exactly"):
        WeatherSeries.from_channels(name="x", dt=300.0, epoch_day=1.0, **channels)


def test_synthetic_radiation_is_exactly_zero_at_night_and_positive_by_day():
    """Exact zeros matter: they make the rad == 0 branch of the photosynthesis term reachable."""
    s = WeatherSeries.synthetic(days=2.0)
    rad = s.values[EXOGENOUS.idx("rad")]
    assert np.any(rad == 0.0)
    assert rad.max() > 500.0
    assert rad.min() == 0.0


def test_episode_slice_has_one_column_per_step_plus_lookahead():
    s = WeatherSeries.synthetic()
    assert s.episode_slice(10.0, N_STEPS, STEP_DT).shape == (EXOGENOUS.size, N_STEPS)
    assert s.episode_slice(10.0, N_STEPS, STEP_DT, lookahead=12).shape == (
        EXOGENOUS.size,
        N_STEPS + 12,
    )


def test_episode_slice_starts_at_the_requested_day():
    """start_day is a day of year, so the offset is measured from the series' own epoch."""
    s = WeatherSeries.synthetic(epoch_day=10.0)
    first = s.episode_slice(start_day=20.0, n_steps=1, step_dt=STEP_DT)[:, 0]
    expected_row = round(10.0 * SECONDS_PER_DAY / s.dt)  # 10 days past the epoch
    np.testing.assert_array_equal(first, s.values[:, expected_row])


def test_episode_slice_respects_a_non_default_epoch():
    """Two series differing only in epoch must return the same weather for the same day of year."""
    a = WeatherSeries.synthetic(epoch_day=1.0)
    b = WeatherSeries.synthetic(epoch_day=1.0, name="shifted")
    shifted = WeatherSeries(name="b", values=b.values[:, 288:], dt=b.dt, epoch_day=2.0)
    np.testing.assert_allclose(
        a.episode_slice(30.0, 48, STEP_DT), shifted.episode_slice(30.0, 48, STEP_DT)
    )


def test_episode_slice_raises_rather_than_clamping_past_the_end():
    """Clamping would leave the episode running on frozen weather and looking entirely plausible."""
    s = WeatherSeries.synthetic(days=50.0)
    with pytest.raises(ValueError, match="past the end"):
        s.episode_slice(start_day=30.0, n_steps=N_STEPS, step_dt=STEP_DT)


def test_episode_slice_raises_before_the_epoch():
    s = WeatherSeries.synthetic(epoch_day=10.0)
    with pytest.raises(ValueError, match="precedes"):
        s.episode_slice(start_day=5.0, n_steps=10, step_dt=STEP_DT)


def test_viable_start_days_are_exactly_those_that_fit():
    s = WeatherSeries.synthetic(days=120.0)
    days = s.viable_start_days(N_STEPS, STEP_DT)
    s.episode_slice(float(days[-1]), N_STEPS, STEP_DT)  # the last viable day must work
    with pytest.raises(ValueError, match="past the end"):
        s.episode_slice(float(days[-1]) + 1.0, N_STEPS, STEP_DT)


# ---- WeatherRepository ---------------------------------------------------------------------------
def test_repository_caches_a_source_and_loads_it_once():
    calls = {"n": 0}

    def loader() -> WeatherSeries:
        calls["n"] += 1
        return WeatherSeries.synthetic(days=10.0)

    repo = WeatherRepository({"s": loader})
    assert repo.load("s") is repo.load("s")
    assert calls["n"] == 1


def test_repository_rejects_an_unknown_source():
    with pytest.raises(KeyError, match="unknown weather source"):
        default_repository().load("no_such_source")


def test_repository_rejects_a_duplicate_registration():
    repo = default_repository()
    with pytest.raises(ValueError, match="already registered"):
        repo.register("synthetic", WeatherSeries.synthetic)


# ---- samplers ------------------------------------------------------------------------------------
def test_fixed_sampler_always_returns_the_same_scenario():
    scenario = WeatherScenario("synthetic", 40.0)
    sampler = FixedWeatherSampler(scenario)
    rng = np.random.default_rng(0)
    assert {sampler.sample(rng) for _ in range(5)} == {scenario}


def test_random_sampler_only_draws_from_the_given_days():
    days = [5.0, 10.0, 15.0]
    sampler = RandomWeatherSampler("synthetic", days)
    rng = np.random.default_rng(1)
    drawn = {sampler.sample(rng).start_day for _ in range(50)}
    assert drawn <= set(days)


def test_random_sampler_is_reproducible_from_a_seed():
    sampler = RandomWeatherSampler("synthetic", [1.0, 2.0, 3.0, 4.0, 5.0])
    a = [sampler.sample(np.random.default_rng(7)).start_day for _ in range(3)]
    b = [sampler.sample(np.random.default_rng(7)).start_day for _ in range(3)]
    assert a == b


def test_random_sampler_rejects_an_empty_day_list():
    with pytest.raises(ValueError, match="must not be empty"):
        RandomWeatherSampler("synthetic", [])


def test_cycling_sampler_walks_the_days_in_order_and_repeats():
    sampler = CyclingWeatherSampler("synthetic", [3.0, 6.0, 9.0])
    rng = np.random.default_rng(0)
    assert [sampler.sample(rng).start_day for _ in range(7)] == [3.0, 6.0, 9.0, 3.0, 6.0, 9.0, 3.0]


def test_cycling_sampler_ignores_the_rng():
    """Evaluation must visit every held-out season, not sample them."""
    a = CyclingWeatherSampler("synthetic", [3.0, 6.0])
    b = CyclingWeatherSampler("synthetic", [3.0, 6.0])
    assert [a.sample(np.random.default_rng(0)).start_day for _ in range(4)] == [
        b.sample(np.random.default_rng(999)).start_day for _ in range(4)
    ]


def test_cycling_sampler_can_be_restarted():
    sampler = CyclingWeatherSampler("synthetic", [3.0, 6.0])
    rng = np.random.default_rng(0)
    sampler.sample(rng)
    sampler.reset()
    assert sampler.sample(rng).start_day == 3.0


@pytest.mark.parametrize(
    "options",
    [{"start_day": 99.0}, {"weather": WeatherScenario("synthetic", 99.0)}],
    ids=["start_day", "scenario"],
)
def test_reset_options_override_whatever_the_sampler_chose(options):
    sampler = RandomWeatherSampler("synthetic", [1.0, 2.0])
    assert sampler.sample(np.random.default_rng(0), options).start_day == 99.0


# ---- the train / test split ----------------------------------------------------------------------
def test_split_start_days_is_disjoint_and_covers_every_viable_day():
    s = WeatherSeries.synthetic(days=120.0)
    rng = np.random.default_rng(3)
    train, test = split_start_days(s, N_STEPS, STEP_DT, rng)
    assert set(train).isdisjoint(set(test))
    assert set(train) | set(test) == set(s.viable_start_days(N_STEPS, STEP_DT))


def test_split_start_days_honours_the_fraction_and_leaves_both_halves_non_empty():
    s = WeatherSeries.synthetic(days=120.0)
    rng = np.random.default_rng(3)
    train, test = split_start_days(s, N_STEPS, STEP_DT, rng, train_fraction=0.8)
    total = len(train) + len(test)
    assert len(train) == pytest.approx(0.8 * total, abs=1)
    for fraction in (0.01, 0.99):
        tr, te = split_start_days(s, N_STEPS, STEP_DT, np.random.default_rng(0), fraction)
        assert len(tr) >= 1
        assert len(te) >= 1


def test_split_start_days_rejects_a_trace_too_short_to_split():
    # 40.5 days leaves exactly one viable start day, too few to divide into two non-empty halves
    s = WeatherSeries.synthetic(days=40.5)
    with pytest.raises(ValueError, match="cannot split"):
        split_start_days(s, N_STEPS, STEP_DT, np.random.default_rng(0))


# ---- the packaged Bleiswijk trace -----------------------------------------------------------------
@pytest.fixture(scope="module")
def bleiswijk():
    """Loaded once: parsing 93k CSV rows per test would be wasteful."""
    return default_repository().load(BLEISWIJK_2014)


def test_packaged_source_is_registered_by_default():
    assert BLEISWIJK_2014 in default_repository().names


def test_bleiswijk_has_the_documented_shape_and_calendar(bleiswijk):
    assert bleiswijk.values.shape == (EXOGENOUS.size, 93311)
    assert bleiswijk.dt == 300.0
    assert bleiswijk.epoch_day == 10.0  # 10 January 2014, per data/SOURCE.md
    assert bleiswijk.span_days == pytest.approx(324.0, abs=0.1)


def test_benchmark_scenario_yields_a_full_season(bleiswijk):
    """40 days from 9 February 2014 at a 30-minute step: the published benchmark setup."""
    assert BENCHMARK_SCENARIO.source == BLEISWIJK_2014
    assert BENCHMARK_SCENARIO.start_day == 40.0
    assert bleiswijk.episode_slice(BENCHMARK_SCENARIO.start_day, N_STEPS, STEP_DT).shape == (
        EXOGENOUS.size,
        N_STEPS,
    )


def test_loader_converts_humidity_and_co2_into_model_units(bleiswijk):
    """The CSV records RH% and ppm; the model consumes densities. Convert back and the recorded
    ranges must reappear; this is what ties the loader to `units` rather than to a copied formula.
    """
    temp = bleiswijk.values[EXOGENOUS.idx("out_temp")]
    rh = vapour_density_to_rh(temp, bleiswijk.values[EXOGENOUS.idx("out_vapor")])
    ppm = co2_density_to_ppm(temp, bleiswijk.values[EXOGENOUS.idx("out_co2")])
    assert rh.min() >= 27.0  # recorded RH spans 27.1 - 99.6 %
    assert rh.max() <= 100.0
    assert ppm.min() >= 339.0  # recorded CO2 spans 339.9 - 702 ppm
    assert ppm.max() <= 703.0


def test_loader_clamps_night_radiation_to_exact_zero(bleiswijk):
    """A residual 1e-9 would step around the rad == 0 branch of the photosynthesis term."""
    rad = bleiswijk.values[EXOGENOUS.idx("rad")]
    assert (rad == 0.0).any()
    assert rad.min() == 0.0


def test_from_csv_rejects_a_file_with_the_wrong_column_count(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("0,1,2\n300,4,5\n")
    with pytest.raises(ValueError, match="expected 6 columns"):
        WeatherSeries.from_csv(bad, epoch_day=1.0)


# ---- several sources ------------------------------------------------------------------------
def test_random_sampler_covers_every_source_and_is_seed_reproducible():
    sampler = RandomWeatherSampler(("a", "b", "c"), [1.0, 2.0])
    draws = [sampler.sample(np.random.default_rng(0)) for _ in range(1)]  # one draw per rng seed
    rng = np.random.default_rng(0)
    draws = [sampler.sample(rng) for _ in range(200)]
    assert {d.source for d in draws} == {"a", "b", "c"}
    assert {d.start_day for d in draws} == {1.0, 2.0}
    rng2 = np.random.default_rng(0)
    assert [sampler.sample(rng2) for _ in range(200)] == draws
    assert sampler.sources == ("a", "b", "c")


def test_single_source_random_sampler_stream_is_unchanged_by_the_multi_source_support():
    """One source draws nothing for itself, so seeded single-source runs keep their sequence."""
    rng = np.random.default_rng(7)
    expected = [float(rng.choice(np.array([3.0, 6.0, 9.0]))) for _ in range(5)]
    sampler = RandomWeatherSampler("bleiswijk_2014", [3.0, 6.0, 9.0])
    rng = np.random.default_rng(7)
    assert [sampler.sample(rng).start_day for _ in range(5)] == expected


def test_cycling_sampler_walks_the_source_major_product():
    sampler = CyclingWeatherSampler(("a", "b"), [3.0, 6.0])
    rng = np.random.default_rng(0)
    seen = [(s.source, s.start_day) for s in (sampler.sample(rng) for _ in range(5))]
    assert seen == [("a", 3.0), ("a", 6.0), ("b", 3.0), ("b", 6.0), ("a", 3.0)]


def test_samplers_reject_empty_sources():
    with pytest.raises(ValueError, match="at least one"):
        RandomWeatherSampler((), [1.0])
    with pytest.raises(ValueError, match="at least one"):
        CyclingWeatherSampler([], [1.0])


def test_from_csv_rejects_a_gappy_time_column(tmp_path):
    path = tmp_path / "gap.csv"
    rows = np.array(
        [[0, 100, 10, 60, 1, 400], [300, 100, 10, 60, 1, 400], [900, 100, 10, 60, 1, 400]]
    )
    np.savetxt(path, rows, delimiter=",")
    with pytest.raises(ValueError, match="uniformly spaced"):
        WeatherSeries.from_csv(path, epoch_day=1.0)


# ---- write_csv and the loader protocol ------------------------------------------------------------


def test_write_csv_round_trips_through_from_csv(tmp_path):
    time_s = np.arange(0.0, 6 * 3600.0, 300.0)
    temp = 5.0 + 3.0 * np.sin(time_s / 3600.0)
    rh = 70.0 + 5.0 * np.cos(time_s / 3600.0)
    rad = np.maximum(0.0, 300.0 * np.sin(np.pi * time_s / time_s[-1]))
    path = write_csv(tmp_path / "t.csv", time_s=time_s, rad=rad, temp=temp, rh=rh, co2_ppm=420.0)
    s = WeatherSeries.from_csv(path, epoch_day=100.0)
    assert s.dt == 300.0
    assert s.n_samples == time_s.size
    np.testing.assert_allclose(s.values[EXOGENOUS.idx("out_temp")], temp, atol=1e-3)
    np.testing.assert_allclose(s.values[EXOGENOUS.idx("rad")], rad, atol=1e-3)
    np.testing.assert_allclose(
        vapour_density_to_rh(temp, s.values[EXOGENOUS.idx("out_vapor")]), rh, atol=0.05
    )
    np.testing.assert_allclose(
        co2_density_to_ppm(temp, s.values[EXOGENOUS.idx("out_co2")]), 420.0, atol=0.1
    )
    raw = np.loadtxt(path, delimiter=",")
    assert raw.shape == (time_s.size, 6)
    assert np.all(raw[:, 4] == 0.0)  # wind defaults to a zero column


def test_write_csv_rejects_a_non_uniform_grid(tmp_path):
    with pytest.raises(ValueError, match="uniformly spaced"):
        write_csv(
            tmp_path / "t.csv",
            time_s=np.array([0.0, 300.0, 900.0]),
            rad=np.zeros(3),
            temp=np.zeros(3),
            rh=np.zeros(3),
            co2_ppm=400.0,
        )


def test_repository_enforces_the_loader_protocol_and_names_the_series():
    repo = WeatherRepository(
        {
            "mine": lambda: WeatherSeries.synthetic(days=3.0, name="whatever"),
            "broken": lambda: np.zeros((4, 10)),
        }
    )
    assert repo.load("mine").name == "mine"  # the registered name wins over the loader's
    with pytest.raises(TypeError, match="not a WeatherSeries"):
        repo.load("broken")
