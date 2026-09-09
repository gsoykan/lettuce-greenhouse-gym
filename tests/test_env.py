"""Tests for the Gymnasium environment.

Episodes here are one simulated day (48 steps at 1800 s) rather than the 40-day benchmark, so the
suite stays fast; the season length is config, not behaviour, and `n_steps` is covered separately.
"""

import itertools
import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from lettuce_greenhouse_gym.config import ActionMode, ControlOverride, EnvConfig, TimestepEncoding
from lettuce_greenhouse_gym.envs.control_env import EnvState, LettuceGreenhouseEnv
from lettuce_greenhouse_gym.model.parameters import (
    MODEL_COEFFS,
    ParameterSet,
    RandomizedParameterProvider,
)
from lettuce_greenhouse_gym.weather import (
    BLEISWIJK_2014,
    RandomWeatherSampler,
    WeatherPerturbation,
    WeatherScenario,
)

SHORT = EnvConfig(episode_days=1.0)  # 48 steps at 1800 s


def make_env(config: EnvConfig | None = None, **kwargs) -> LettuceGreenhouseEnv:
    return LettuceGreenhouseEnv(config or SHORT, **kwargs)


# ---- Gymnasium conformance -----------------------------------------------------------------------
def test_passes_the_gymnasium_env_checker():
    """The authority on whether this is a well-formed env, so it gets its own test."""
    check_env(make_env(), skip_render_check=True)


def test_reset_returns_an_observation_inside_the_declared_space():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_the_five_tuple_with_an_in_space_observation():
    env = make_env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---- episode boundaries --------------------------------------------------------------------------
def test_terminates_exactly_at_the_end_of_the_season_and_never_truncates():
    """The season ends in harvest, so it is a terminal state of the MDP, not a time limit."""
    env = make_env()
    env.reset(seed=0)
    for step in range(1, env.config.n_steps + 1):
        _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
        assert truncated is False
        assert terminated is (step == env.config.n_steps)


def test_stepping_past_the_end_of_an_episode_raises():
    env = make_env()
    env.reset(seed=0)
    for _ in range(env.config.n_steps):
        env.step(np.zeros(env.action_space.shape))
    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step(np.zeros(env.action_space.shape))


def test_reset_puts_the_crop_back_to_a_transplant():
    """start_day moves the calendar, not the crop: every episode plants a fresh seedling."""
    env = make_env()
    env.reset(seed=0)
    for _ in range(20):
        env.step(env.action_space.sample())
    grown = env._x[0]
    env.reset(seed=1)
    assert env._x[0] == pytest.approx(env.config.effective_x0()[0])
    assert env._x[0] != grown


# ---- determinism ---------------------------------------------------------------------------------
def test_the_same_seed_reproduces_the_same_episode():
    """Randomised weather and parameters both draw from self.np_random, so one seed fixes both."""
    days = [20.0, 40.0, 60.0]

    def rollout(seed: int):
        env = LettuceGreenhouseEnv(
            SHORT,
            weather_sampler=RandomWeatherSampler(BLEISWIJK_2014, days),
            parameter_provider=RandomizedParameterProvider(
                ParameterSet.from_defaults(MODEL_COEFFS), {"leak": (1e-5, 2e-5)}
            ),
        )
        obs, info = env.reset(seed=seed)
        rewards = [env.step(np.full(env.action_space.shape, 0.1))[1] for _ in range(10)]
        return obs, info["params"], rewards

    a_obs, a_params, a_rewards = rollout(3)
    b_obs, b_params, b_rewards = rollout(3)
    np.testing.assert_array_equal(a_obs, b_obs)
    np.testing.assert_array_equal(a_params, b_params)
    assert a_rewards == b_rewards


def test_a_returned_observation_does_not_alias_environment_state():
    env = make_env()
    obs, _ = env.reset(seed=0)
    obs[:] = 999.0
    assert not np.any(env._observation() == 999.0)


# ---- action decoding -----------------------------------------------------------------------------
def test_absolute_mode_maps_the_action_range_onto_the_control_bounds():
    env = make_env(EnvConfig(episode_days=1.0, action_mode=ActionMode.ABSOLUTE))
    env.reset(seed=0)
    lo, hi = env.config.effective_control_bounds()
    np.testing.assert_allclose(env.decode_action(-np.ones(3)), lo)
    np.testing.assert_allclose(env.decode_action(np.ones(3)), hi)
    np.testing.assert_allclose(env.decode_action(np.zeros(3)), 0.5 * (lo + hi))


def test_delta_mode_moves_the_previous_control_by_at_most_du_max():
    env = make_env(EnvConfig(episode_days=1.0, action_mode=ActionMode.DELTA))
    env.reset(seed=0)
    u0 = env.config.effective_u0()
    np.testing.assert_allclose(env.decode_action(np.zeros(3)), u0)  # zero action holds
    expected = np.clip(u0 + env.config.effective_du_max(), *env.config.effective_control_bounds())
    np.testing.assert_allclose(env.decode_action(np.ones(3)), expected)


def test_an_out_of_range_action_is_clipped_rather_than_rejected():
    """Gaussian policies routinely emit values just outside [-1, 1]."""
    env = make_env(EnvConfig(episode_days=1.0, action_mode=ActionMode.ABSOLUTE))
    env.reset(seed=0)
    _, hi = env.config.effective_control_bounds()
    np.testing.assert_allclose(env.decode_action(np.full(3, 5.0)), hi)


def test_decoding_respects_overridden_control_bounds():
    config = EnvConfig(
        episode_days=1.0,
        action_mode=ActionMode.ABSOLUTE,
        control_overrides=(ControlOverride("heating", hi=200.0),),
    )
    env = make_env(config)
    env.reset(seed=0)
    assert env.decode_action(np.ones(3))[env.model.controls.idx("heating")] == pytest.approx(200.0)


# ---- the observation layout ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("previous_control", "timestep", "window"),
    list(itertools.product([True, False], [True, False], [0, 1, 13])),
    ids=lambda v: str(v),
)
def test_observation_matches_the_declared_space_for_every_layout(
    previous_control, timestep, window
):
    """Four independently-toggled parts: the declared shape and the built vector must agree in all
    combinations, which is the failure mode when a space and an assembler are written separately."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # include_timestep=False warns on purpose
        env = make_env(
            EnvConfig(
                episode_days=1.0,
                include_previous_control=previous_control,
                include_timestep=timestep,
                weather_window=window,
            )
        )
        obs, _ = env.reset(seed=0)
        expected = 4 + (3 if previous_control else 0) + (1 if timestep else 0) + 4 * window
        assert obs.shape == (expected,)
        assert env.observation_space.contains(obs)
        # and still correct at the terminal step, where the forecast window reaches furthest
        for _ in range(env.config.n_steps):
            obs, _, _, _, _ = env.step(np.zeros(3))
        assert env.observation_space.contains(obs)


def test_disabling_the_timestep_warns_because_termination_needs_observable_time():
    with pytest.warns(UserWarning, match="not Markov"):
        make_env(EnvConfig(episode_days=1.0, include_timestep=False))


# ---- info: the exo seam --------------------------------------------------------------------------
def test_info_exposes_the_episode_parameters_raw_and_named():
    env = make_env()
    _, info = env.reset(seed=0)
    np.testing.assert_array_equal(info["params"], env.model.constants.to_array())
    assert info["param_names"] == env.model.constants.names
    assert len(info["params"]) == len(info["param_names"])


def test_info_parameters_follow_a_randomising_provider():
    """The seam exo depends on: whatever the provider drew must be visible to a wrapper."""
    provider = RandomizedParameterProvider(
        ParameterSet.from_defaults(MODEL_COEFFS), {"leak": (1e-5, 2e-5)}
    )
    env = LettuceGreenhouseEnv(SHORT, parameter_provider=provider)
    seen = set()
    for seed in range(5):
        _, info = env.reset(seed=seed)
        seen.add(float(info["params"][env.model.constants.idx("leak")]))
    assert len(seen) > 1  # the drawn value actually varies between episodes
    assert all(1e-5 <= v <= 2e-5 for v in seen)


def test_info_carries_the_reward_breakdown_after_a_step():
    env = make_env()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert reward == pytest.approx(
        info["revenue"] - info["energy_cost"] - info["co2_cost"] - info["penalty"]
    )


def test_reset_options_pin_the_weather():
    env = make_env(weather_sampler=RandomWeatherSampler(BLEISWIJK_2014, [20.0, 60.0]))
    _, info = env.reset(seed=0, options={"start_day": 100.0})
    assert info["start_day"] == 100.0
    _, info = env.reset(seed=0, options={"weather": WeatherScenario(BLEISWIJK_2014, 150.0)})
    assert info["start_day"] == 150.0


# ---- lifecycle guards and render_mode --------------------------------------------------------------
def test_step_before_reset_raises():
    with pytest.raises(RuntimeError, match="call reset"):
        make_env().step(np.zeros(3))


def test_unsupported_render_mode_is_rejected():
    with pytest.raises(ValueError, match="render_mode"):
        make_env(render_mode="human")
    assert make_env().render_mode is None


# ---- the public control API --------------------------------------------------------------------
def test_state_and_control_are_copies():
    env = make_env()
    env.reset(seed=0)
    env.state[:] = 999.0
    env.control[:] = 999.0
    assert not np.any(env.state == 999.0)
    assert not np.any(env.control == 999.0)


def test_exposed_integrator_reproduces_the_environments_own_step():
    """A planner using env.integrator plans with exactly the dynamics the env runs."""
    env = make_env()
    env.reset(seed=0)
    x, action = env.state, np.full(3, 0.3)
    v = env.weather_forecast(1)[:, 0]
    u = env.decode_action(action)
    predicted = np.asarray(env.integrator(x, u, v, env.parameters.to_array())).ravel()
    env.step(action)
    np.testing.assert_allclose(env.state, predicted, rtol=0, atol=1e-12)


def test_exposed_measurement_matches_the_observation():
    env = make_env()
    obs, _ = env.reset(seed=0)
    y = np.asarray(env.measurement(env.state)).ravel()
    np.testing.assert_allclose(obs[:4], y.astype(np.float32))


def test_parameters_property_matches_info():
    env = make_env()
    _, info = env.reset(seed=0)
    np.testing.assert_array_equal(env.parameters.to_array(), info["params"])


def test_weather_forecast_has_the_requested_horizon_and_starts_now():
    env = make_env()
    env.reset(seed=0)
    forecast = env.weather_forecast(16)
    assert forecast.shape == (4, 16)
    np.testing.assert_array_equal(forecast[:, 0], env._weather[:, 0])
    env.step(np.zeros(3))
    np.testing.assert_array_equal(env.weather_forecast(1)[:, 0], env._weather[:, 1])


def test_weather_forecast_may_extend_past_the_season_but_not_the_trace():
    env = make_env()
    env.reset(seed=0, options={"start_day": 330.0})  # the trace ends on day 333
    env.weather_forecast(env.config.n_steps + 10)  # past the 1-day season: fine
    with pytest.raises(ValueError, match="past the end"):
        env.weather_forecast(10_000)


def test_weather_forecast_before_reset_raises():
    with pytest.raises(RuntimeError, match="call reset"):
        make_env().weather_forecast(1)


# ---- encode_control: the inverse map controllers use ----------------------------------------
@pytest.mark.parametrize("mode", [ActionMode.ABSOLUTE, ActionMode.DELTA])
def test_encode_control_inverts_decode_action(mode):
    env = make_env(EnvConfig(episode_days=1.0, action_mode=mode))
    env.reset(seed=0)
    lo, hi = env.config.effective_control_bounds()
    # in delta mode only targets within one du_max of u_prev are reachable in a single step
    reach = env.config.effective_du_max() if mode is ActionMode.DELTA else hi - lo
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.clip(env.control + rng.uniform(-1, 1, size=3) * reach, lo, hi)
        np.testing.assert_allclose(env.decode_action(env.encode_control(u)), u, atol=1e-12)


def test_encode_control_saturates_at_the_bounds():
    env = make_env(EnvConfig(episode_days=1.0, action_mode=ActionMode.ABSOLUTE))
    env.reset(seed=0)
    _, hi = env.config.effective_control_bounds()
    np.testing.assert_allclose(env.decode_action(env.encode_control(hi * 2)), hi)


def test_encode_control_before_reset_raises():
    with pytest.raises(RuntimeError, match="call reset"):
        make_env().encode_control(np.zeros(3))


# ---- the season clock -------------------------------------------------------------------------
def test_timestep_index_encoding_counts_steps():
    env = make_env(EnvConfig(episode_days=1.0, timestep_encoding=TimestepEncoding.INDEX))
    obs, _ = env.reset(seed=0)
    assert obs[7] == 0.0
    for k in range(1, 4):
        obs, *_ = env.step(np.zeros(3))
        assert obs[7] == float(k)
    assert env.observation_space.high[7] == env.config.n_steps


def test_timestep_progress_encoding_is_a_fraction():
    env = make_env(EnvConfig(episode_days=1.0))
    env.reset(seed=0)
    obs, *_ = env.step(np.zeros(3))
    assert obs[7] == pytest.approx(1 / env.config.n_steps)
    assert env.observation_space.high[7] == 1.0


# ---- per-step parameter noise -------------------------------------------------------------------
def test_per_step_provider_redraws_before_every_transition():
    provider = RandomizedParameterProvider.relative(
        ParameterSet.from_defaults(MODEL_COEFFS), 0.05, per_step=True
    )
    env = LettuceGreenhouseEnv(SHORT, parameter_provider=provider)
    _, info0 = env.reset(seed=0)
    _, _, _, _, info1 = env.step(np.zeros(3))
    _, _, _, _, info2 = env.step(np.zeros(3))
    assert not np.array_equal(info1["params"], info0["params"])
    assert not np.array_equal(info2["params"], info1["params"])
    np.testing.assert_array_equal(info2["params"], env.parameters.to_array())


def test_per_episode_provider_holds_parameters_within_an_episode():
    provider = RandomizedParameterProvider.relative(ParameterSet.from_defaults(MODEL_COEFFS), 0.05)
    env = LettuceGreenhouseEnv(SHORT, parameter_provider=provider)
    _, info0 = env.reset(seed=0)
    _, _, _, _, info1 = env.step(np.zeros(3))
    np.testing.assert_array_equal(info1["params"], info0["params"])


def test_a_custom_step_hook_can_drift_a_coefficient():
    """The env honours any provider's step(): here leakage grows linearly over the episode."""
    from lettuce_greenhouse_gym.model.parameters import ParameterProvider

    class Drift(ParameterProvider):
        def __init__(self, base, n_steps):
            self.base, self.n_steps = base, n_steps

        def sample(self, rng):
            return self.base

        def step(self, rng, step_index, current):
            return self.base.override(leak=self.base.get("leak") * (1 + step_index / self.n_steps))

    base = ParameterSet.from_defaults(MODEL_COEFFS)
    env = LettuceGreenhouseEnv(SHORT, parameter_provider=Drift(base, SHORT.n_steps))
    env.reset(seed=0)
    leak = env.model.constants.idx("leak")
    seen = []
    for _ in range(3):
        _, _, _, _, info = env.step(np.zeros(3))
        seen.append(info["params"][leak])
    expected = [base.get("leak") * (1 + k / SHORT.n_steps) for k in range(3)]
    np.testing.assert_allclose(seen, expected)
    assert env.parameters.get("leak") == pytest.approx(expected[-1])


# ---- branching an episode: snapshot / restore / observe -------------------------------------------


def _run(env, actions):
    out = []
    for a in actions:
        obs, r, *_ = env.step(a)
        out.append((obs, r))
    return out


def test_restore_replays_the_same_future_from_the_same_moment():
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=1.0))
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1, 1, size=(6, 3))
    _run(env, actions[:2])
    snap = env.snapshot()
    first = _run(env, actions[2:])
    obs_back = env.restore(snap)
    np.testing.assert_array_equal(obs_back, env.observe(snap.x, snap.u_prev, snap.step_index))
    second = _run(env, actions[2:])
    for (o1, r1), (o2, r2) in zip(first, second, strict=True):
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2


def test_snapshot_is_plain_data_and_restore_validates_it():
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=1.0))
    env.reset(seed=0)
    snap = env.snapshot()
    assert isinstance(snap, EnvState)
    assert snap.step_index == 0
    assert snap.scenario.start_day == 40.0
    from dataclasses import replace

    moved = replace(snap, x=np.array([0.01, 0.001, 18.0, 0.009]), step_index=10)
    obs = env.restore(moved)
    np.testing.assert_allclose(env.state, moved.x)
    assert obs[7] == pytest.approx(10 / env.config.n_steps)  # the clock moved with the state
    with pytest.raises(ValueError, match="state bounds"):
        env.restore(replace(snap, x=np.array([-1.0, 0.001, 18.0, 0.009])))
    with pytest.raises(ValueError, match="step_index"):
        env.restore(replace(snap, step_index=env.config.n_steps + 1))
    with pytest.raises(ValueError, match="belongs to"):
        env.restore(replace(snap, scenario=WeatherScenario(BLEISWIJK_2014, 41.0)))
    fresh = LettuceGreenhouseEnv(EnvConfig(episode_days=1.0))
    with pytest.raises(RuntimeError, match="reset"):
        fresh.snapshot()


def test_observe_matches_the_emitted_observation_and_leaves_the_state_alone():
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=1.0, weather_window=4))
    obs0, _ = env.reset(seed=3)
    np.testing.assert_array_equal(env.observe(env.state, env.control, 0), obs0)
    obs1, *_ = env.step(np.array([0.5, -0.2, 0.1]))
    hypothetical = env.observe(np.array([0.004, 0.0012, 16.0, 0.0085]), env.control, 5)
    assert hypothetical.shape == obs1.shape
    assert hypothetical[7] == pytest.approx(5 / env.config.n_steps)
    np.testing.assert_array_equal(env.observe(env.state, env.control, 1), obs1)  # state untouched
    with pytest.raises(ValueError, match="shape"):
        env.observe(np.zeros(3), env.control, 1)
    with pytest.raises(ValueError, match="step_index"):
        env.observe(env.state, env.control, env.config.n_steps + 1)


# ---- absolute mode with a rate limit ---------------------------------------------------------------


def test_absolute_rate_limit_moves_at_most_du_max_per_step():
    cfg = EnvConfig(episode_days=1.0, action_mode=ActionMode.ABSOLUTE, absolute_rate_limit=True)
    env = LettuceGreenhouseEnv(cfg)
    env.reset(seed=0)
    _, hi = cfg.effective_control_bounds()
    du = cfg.effective_du_max()
    u0 = env.control
    env.step(np.ones(3))  # ask for every actuator's maximum
    np.testing.assert_allclose(env.control, np.minimum(u0 + du, hi))
    plain = LettuceGreenhouseEnv(EnvConfig(episode_days=1.0, action_mode=ActionMode.ABSOLUTE))
    plain.reset(seed=0)
    plain.step(np.ones(3))
    np.testing.assert_allclose(plain.control, hi)
    with pytest.raises(ValueError, match="absolute action mode only"):
        EnvConfig(action_mode=ActionMode.DELTA, absolute_rate_limit=True)


# ---- the weather hooks -----------------------------------------------------------------------------


class ScaleRadiation(WeatherPerturbation):
    """What the plant experiences: radiation halved."""

    def realise(self, rng, weather, dt):
        out = weather.copy()
        out[0] *= 0.5
        return out


class NoisyForecast(WeatherPerturbation):
    """What a controller is told: temperature offset growing with lead time; the plant is unaffected."""

    def forecast(self, rng, step_index, weather, dt):
        out = weather.copy()
        out[2] += 0.1 * np.arange(1, weather.shape[1] + 1)
        return out


class WrongShape(WeatherPerturbation):
    def realise(self, rng, weather, dt):
        return weather[:, :-1]


class OutOfBounds(WeatherPerturbation):
    def forecast(self, rng, step_index, weather, dt):
        out = weather.copy()
        out[0] -= 1.0e6
        return out


def test_realise_hook_changes_what_the_plant_and_every_forecast_see():
    cfg = EnvConfig(episode_days=1.0)
    plain, scaled = (
        LettuceGreenhouseEnv(cfg),
        LettuceGreenhouseEnv(cfg, weather_perturbation=ScaleRadiation()),
    )
    o_plain, _ = plain.reset(seed=0)
    o_scaled, _ = scaled.reset(seed=0)
    assert o_scaled[8] == pytest.approx(0.5 * o_plain[8])  # radiation in the weather block
    np.testing.assert_allclose(scaled.weather_forecast(6)[0], 0.5 * plain.weather_forecast(6)[0])
    np.testing.assert_allclose(scaled.weather_forecast(6)[1:], plain.weather_forecast(6)[1:])
    a = np.array([0.3, 0.0, 0.2])
    for _ in range(20):
        plain.step(a)
        scaled.step(a)
    assert not np.allclose(plain.state, scaled.state)  # less light, different crop


def test_forecast_hook_changes_only_what_controllers_are_told():
    cfg = EnvConfig(episode_days=1.0)
    plain, noisy = (
        LettuceGreenhouseEnv(cfg),
        LettuceGreenhouseEnv(cfg, weather_perturbation=NoisyForecast()),
    )
    o_plain, _ = plain.reset(seed=0)
    o_noisy, _ = noisy.reset(seed=0)
    assert o_noisy[10] == pytest.approx(o_plain[10] + 0.1)  # outdoor temperature, one step ahead
    f_plain, f_noisy = plain.weather_forecast(5), noisy.weather_forecast(5)
    np.testing.assert_allclose(f_noisy[2] - f_plain[2], 0.1 * np.arange(1, 6))
    a = np.array([0.3, 0.0, 0.2])
    for _ in range(20):
        plain.step(a)
        noisy.step(a)
    np.testing.assert_array_equal(plain.state, noisy.state)  # the plant ran on the true weather


def test_weather_hooks_that_break_the_contract_are_refused():
    cfg = EnvConfig(episode_days=1.0)
    with pytest.raises(ValueError, match="realise returned shape"):
        LettuceGreenhouseEnv(cfg, weather_perturbation=WrongShape()).reset(seed=0)
    with pytest.raises(ValueError, match="physical bounds"):
        LettuceGreenhouseEnv(cfg, weather_perturbation=OutOfBounds()).reset(seed=0)


def test_forecast_runs_to_the_end_of_the_trace_and_no_further():
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=1.0))
    env.reset(seed=0)
    available = env._weather.shape[1]
    assert env.weather_forecast(available).shape == (4, available)
    with pytest.raises(ValueError, match="past the end of the trace"):
        env.weather_forecast(available + 1)


# ---- the parameter-observation wrapper --------------------------------------------------------------


def test_parameter_observation_appends_named_coefficients_and_follows_per_step_changes():
    from lettuce_greenhouse_gym import ParameterObservation

    base = ParameterSet.from_defaults(MODEL_COEFFS)
    provider = RandomizedParameterProvider.relative(base, 0.5, names=("leak",), per_step=True)
    inner = LettuceGreenhouseEnv(EnvConfig(episode_days=0.5), parameter_provider=provider)
    env = ParameterObservation(inner, ["leak", "sat_vp1"], {"leak": (0.5e-5, 1.5e-5)})
    assert env.observation_space.shape == (inner.observation_space.shape[0] + 2,)
    obs, info = env.reset(seed=0)
    leak, sat = info["params"][base.idx("leak")], info["params"][base.idx("sat_vp1")]
    assert obs[-2] == pytest.approx((leak - 0.5e-5) / 1.0e-5)
    assert obs[-1] == pytest.approx(sat)
    obs, _, _, _, info = env.step(np.zeros(3))
    assert obs[-2] == pytest.approx((info["params"][base.idx("leak")] - 0.5e-5) / 1.0e-5)
    with pytest.raises(ValueError, match="unknown coefficient"):
        ParameterObservation(inner, ["nope"])
    with pytest.raises(ValueError, match="not appended"):
        ParameterObservation(inner, ["leak"], {"sat_vp1": (0.0, 1.0)})
