"""Reference controllers: the runner's shapes, and pinned season returns as a regression gate."""

import casadi
import numpy as np
import pytest

from lettuce_greenhouse_gym import (
    CONTROL,
    MODEL_COEFFS,
    ParameterSet,
    RandomizedParameterProvider,
)
from lettuce_greenhouse_gym.baselines import (
    AllOff,
    ConstantControl,
    GrowerHeuristic,
    GrowerRules,
    MPCSettings,
    NominalMPC,
    run_episode,
)
from lettuce_greenhouse_gym.config import ActionMode, ControlOverride, EnvConfig, InitialState
from lettuce_greenhouse_gym.envs.control_env import LettuceGreenhouseEnv
from lettuce_greenhouse_gym.rewards import (
    PenaltyBound,
    RewardConfig,
    benchmark_reward_config,
    economic_stage_reward,
)

SHORT = EnvConfig(episode_days=1.0)


def test_run_episode_returns_consistent_shapes():
    env = LettuceGreenhouseEnv(SHORT)
    log = run_episode(env, AllOff(), seed=0)
    n = env.config.n_steps
    assert log.x.shape == (n + 1, 4)
    assert log.y.shape == (n + 1, 4)
    assert log.u.shape == (n, 3)
    assert log.v.shape == (n, 4)
    assert log.reward.shape == (n,)
    assert len(log.info) == n
    assert log.total_return == pytest.approx(log.reward.sum())


def test_run_episode_starts_from_x0_and_logs_the_applied_control():
    """The log holds what the env applied: in delta mode the heater ramps down from u0 at du_max."""
    env = LettuceGreenhouseEnv(SHORT)
    log = run_episode(env, AllOff(), seed=0)
    np.testing.assert_array_equal(log.x[0], env.config.effective_x0())
    np.testing.assert_allclose(log.u[:4, 2], [35.0, 20.0, 5.0, 0.0])
    np.testing.assert_array_equal(log.u[4:], 0.0)


def test_all_off_is_immediate_in_absolute_mode():
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=1.0, action_mode=ActionMode.ABSOLUTE))
    log = run_episode(env, AllOff(), seed=0)
    np.testing.assert_array_equal(log.u, 0.0)


@pytest.mark.parametrize("mode", [ActionMode.ABSOLUTE, ActionMode.DELTA])
def test_constant_control_is_reached_and_held_in_both_modes(mode):
    """In delta mode the target is reached over a few steps, then held exactly."""
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=1.0, action_mode=mode))
    target = np.array([0.6, 3.0, 100.0])
    log = run_episode(env, ConstantControl(target), seed=0)
    settled = np.all(np.isclose(log.u, target), axis=1)
    assert settled[-1]
    assert settled.sum() >= len(settled) - 10


def test_constant_control_rejects_the_wrong_width():
    with pytest.raises(ValueError, match="expected 3 controls"):
        ConstantControl([0.0, 0.0])


# ---- pinned benchmark-season returns ---------------------------------------------------------
# One number per controller checks dynamics, reward, and weather loading together. A change here
# means the benchmark itself changed.
@pytest.mark.parametrize(
    ("controller", "expected"),
    [
        (AllOff(), -64.53472461089223),
        (ConstantControl([0.0, 0.0, 50.0]), -73.57157594819853),  # hold the benchmark's u0
        (GrowerHeuristic(), -4.351605460692962),
    ],
    ids=["all_off", "hold_u0", "grower_heuristic"],
)
def test_benchmark_season_return_is_pinned(controller, expected):
    log = run_episode(LettuceGreenhouseEnv(), controller, seed=0)
    assert log.reward.shape == (1920,)
    assert log.total_return == pytest.approx(expected, rel=1e-6)


# ---- the grower heuristic ---------------------------------------------------------------------
def _heuristic_control(y, rad, rules=None):
    """Feed one observable vector and one radiation value through the rules on a fresh env."""
    env = LettuceGreenhouseEnv(SHORT)
    obs, _ = env.reset(seed=0)
    ctrl = GrowerHeuristic(rules)
    ctrl.reset(env)
    fake_obs = obs.copy()
    fake_obs[:4] = y
    forecast = np.array([[rad], [env.weather_forecast(1)[1, 0]], [10.0], [0.005]])
    env.weather_forecast = lambda n: forecast  # only the current weather is read
    return CONTROL.unpack(ctrl.control(fake_obs, env))


def test_cold_and_dark_heats_and_does_not_dose():
    u = _heuristic_control([0.01, 800.0, 4.0, 60.0], rad=0.0)
    assert u.heating == 150.0  # 6 degC below the 10 degC night setpoint saturates the heater
    assert u.co2_supply == 0.0
    assert u.ventilation == 0.0


def test_mild_deficit_heats_proportionally():
    u = _heuristic_control([0.01, 800.0, 12.0, 60.0], rad=300.0)  # 2 degC below the day setpoint
    assert u.heating == pytest.approx(30.0 * 2.0)
    assert u.ventilation == 0.0


def test_hot_vents_proportionally():
    u = _heuristic_control([0.01, 800.0, 18.0, 60.0], rad=300.0)  # 2 degC over 14 + deadband 2
    assert u.heating == 0.0
    assert u.ventilation == pytest.approx(2.0 * 2.0)


def test_humid_at_setpoint_vents():
    u = _heuristic_control([0.01, 800.0, 14.0, 90.0], rad=300.0)
    assert u.ventilation == pytest.approx(0.5 * 10.0)


def test_bright_and_co2_poor_doses_unless_vents_are_open():
    dosing = _heuristic_control([0.01, 600.0, 14.0, 60.0], rad=300.0)
    assert dosing.co2_supply == pytest.approx(0.006 * 150.0)
    venting = _heuristic_control([0.01, 600.0, 25.0, 60.0], rad=300.0)
    assert venting.ventilation == 7.5  # saturated, well past the 2 mm/s cutoff
    assert venting.co2_supply == 0.0


def test_dosing_saturates_at_the_actuator_limit():
    u = _heuristic_control([0.01, 300.0, 14.0, 60.0], rad=300.0)  # 450 ppm deficit -> 2.7 mg/m2/s
    assert u.co2_supply == 1.2


def test_rules_respect_overridden_actuator_limits():
    """Clipping uses the episode's effective bounds, so a bigger boiler is actually used."""
    env = LettuceGreenhouseEnv(
        EnvConfig(episode_days=1.0, control_overrides=(ControlOverride("heating", hi=200.0),))
    )
    obs, _ = env.reset(seed=0)
    ctrl = GrowerHeuristic()
    ctrl.reset(env)
    obs[:4] = [0.01, 800.0, 2.0, 60.0]  # 8 degC deficit -> 240 W/m2 requested
    assert CONTROL.unpack(ctrl.control(obs, env)).heating == 200.0


def test_rules_reject_negative_gains():
    with pytest.raises(ValueError, match="non-negative"):
        GrowerRules(heat_gain=-1.0)


def test_grower_heuristic_beats_all_off_over_the_benchmark_season():
    """The point of a reference controller: if it cannot beat 'off', something is wrong."""
    heuristic = run_episode(LettuceGreenhouseEnv(), GrowerHeuristic(), seed=0)
    off = run_episode(LettuceGreenhouseEnv(), AllOff(), seed=0)
    assert heuristic.total_return > off.total_return


# ---- the nominal MPC ---------------------------------------------------------------------------
def test_stage_reward_matches_the_env_reward_on_a_logged_episode():
    """The symbolic objective is the env's reward, not an approximation of it."""
    env = LettuceGreenhouseEnv(SHORT)
    log = run_episode(env, GrowerHeuristic(), seed=0)
    for k in range(env.config.n_steps):
        symbolic = economic_stage_reward(
            casadi.DM(log.x[k]),
            casadi.DM(log.x[k + 1]),
            casadi.DM(log.u[k]),
            casadi.DM(log.y[k + 1]),
            env.config.dt,
            env.config.reward,
        )
        assert float(symbolic) == pytest.approx(log.reward[k], abs=1e-9)


@pytest.mark.parametrize("mode", [ActionMode.DELTA, ActionMode.ABSOLUTE])
def test_mpc_respects_bounds_and_rate_limits(mode):
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=0.5, action_mode=mode))
    mpc = NominalMPC(MPCSettings(horizon=6))
    log = run_episode(env, mpc, seed=0)
    lo, hi = env.config.effective_control_bounds()
    assert np.all(log.u >= lo - 1e-9)
    assert np.all(log.u <= hi + 1e-9)
    if mode is ActionMode.DELTA:
        du_max = env.config.effective_du_max()
        assert np.all(np.abs(log.u[0] - env.config.effective_u0()) <= du_max + 1e-9)
        assert np.all(np.abs(np.diff(log.u, axis=0)) <= du_max + 1e-9)
    assert mpc.n_failures == 0
    assert np.all(np.isfinite(log.reward))


def test_mpc_uses_overridden_actuator_limits():
    """The plan is bounded by the episode's effective limits, not the declared ones.

    Cheap energy, a 5 degC start and a 20 degC comfort floor make ~250 W/m2 optimal for the first
    step: reachable only if the overridden 300 W/m2 limit is what the optimiser sees.
    """
    warm_floor = RewardConfig(
        economics=benchmark_reward_config(energy_cost=1e-4).economics,
        bounds=(PenaltyBound("indoor_temp", 20.0, 20.0, 3e-3, 5e-3),),
    )
    env = LettuceGreenhouseEnv(
        EnvConfig(
            episode_days=0.25,
            action_mode=ActionMode.ABSOLUTE,
            reward=warm_floor,
            control_overrides=(ControlOverride("heating", hi=300.0),),
            initial_state=(InitialState("indoor_temp", 5.0),),
        )
    )
    log = run_episode(env, NominalMPC(MPCSettings(horizon=4)), seed=0)
    heating = log.u[:, CONTROL.idx("heating")]
    assert np.all(heating <= 300.0 + 1e-9)
    assert heating[0] > 150.0


def test_mpc_settings_reject_an_empty_horizon_or_budget():
    with pytest.raises(ValueError, match="horizon"):
        MPCSettings(horizon=0)
    with pytest.raises(ValueError, match="max_iter"):
        MPCSettings(max_iter=0)


def test_mpc_plans_with_nominal_or_true_parameters():
    provider = RandomizedParameterProvider(
        ParameterSet.from_defaults(MODEL_COEFFS), {"leak": (2e-5, 3e-5)}
    )
    env = LettuceGreenhouseEnv(SHORT, parameter_provider=provider)
    env.reset(seed=0)
    nominal, oracle = NominalMPC(), NominalMPC(MPCSettings(use_true_parameters=True))
    nominal.reset(env)
    oracle.reset(env)
    np.testing.assert_array_equal(nominal._c, env.model.constants.to_array())
    np.testing.assert_array_equal(oracle._c, env.parameters.to_array())
    assert (
        nominal._c[env.model.constants.idx("leak")] != oracle._c[env.model.constants.idx("leak")]
    )


def test_mpc_ipopt_options_reach_the_solver():
    """A one-iteration budget cannot converge, so every solve is counted as a failure."""
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=0.25))
    starved = NominalMPC(MPCSettings(horizon=3, ipopt_options={"max_iter": 1}))
    run_episode(env, starved, seed=0)
    assert starved.n_failures == env.config.n_steps


def test_mpc_is_privileged_and_the_heuristic_is_not():
    assert NominalMPC.privileged is True
    assert GrowerHeuristic.privileged is False


@pytest.mark.slow
def test_mpc_beats_the_grower_heuristic_over_the_benchmark_season():
    ctrl = NominalMPC()
    mpc = run_episode(LettuceGreenhouseEnv(), ctrl, seed=0)
    heuristic = run_episode(LettuceGreenhouseEnv(), GrowerHeuristic(), seed=0)
    assert ctrl.n_failures == 0
    assert mpc.total_return > heuristic.total_return
    # loose pin: the value depends on the IPOPT build, unlike the closed-form controllers above
    assert mpc.total_return == pytest.approx(3.6194, abs=0.05)


# ---- planning with other coefficients or bands, and solve statistics ------------------------------


def test_mpc_records_solve_statistics_and_the_runner_stores_them():
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=0.25))
    mpc = NominalMPC(MPCSettings(horizon=4))
    log = run_episode(env, mpc, seed=0)
    assert len(mpc.solves) == env.config.n_steps
    assert all(s.wall_time > 0 and s.iterations > 0 for s in mpc.solves)
    assert [s.step_index for s in mpc.solves] == list(range(env.config.n_steps))
    stats = log.info[-1]["controller"]
    assert stats["status"]
    assert isinstance(stats["success"], bool)
    assert "controller" not in run_episode(env, GrowerHeuristic(), seed=0).info[0]


def test_mpc_can_plan_with_supplied_coefficients_and_tightened_bands():
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=0.25))
    nominal = env.model.constants
    wrong = nominal.override(leak=nominal.get("leak") * 3.0)
    tight = tuple(
        PenaltyBound(b.name, b.lo, b.hi - 2.0 if b.name == "rh" else b.hi, b.w_lo, b.w_hi)
        for b in env.config.reward.bounds
    )
    reference = run_episode(env, NominalMPC(MPCSettings(horizon=4)), seed=0)
    mis_specified = run_episode(env, NominalMPC(MPCSettings(horizon=4, parameters=wrong)), seed=0)
    tightened = run_episode(env, NominalMPC(MPCSettings(horizon=4, bounds=tight)), seed=0)
    assert not np.allclose(mis_specified.u, reference.u)
    assert not np.allclose(tightened.u, reference.u)
    assert np.all(np.isfinite(tightened.reward))
    other = ParameterSet.from_defaults(nominal.constants[:-1])
    with pytest.raises(ValueError, match="name the model's coefficients"):
        run_episode(env, NominalMPC(MPCSettings(horizon=4, parameters=other)), seed=0)


def test_episode_log_exports_a_flat_table(tmp_path):
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=0.25))
    log = run_episode(env, NominalMPC(MPCSettings(horizon=4)), seed=0)
    rows = log.to_records(dt=env.config.dt)
    assert len(rows) == env.config.n_steps
    assert rows[1]["time_s"] == env.config.dt
    assert rows[0]["x_dry_weight"] == log.x[1, 0]
    assert rows[0]["u_heating"] == log.u[0, CONTROL.idx("heating")]
    assert rows[0]["revenue"] == log.info[0]["revenue"]
    assert rows[0]["controller_iterations"] == log.info[0]["controller"]["iterations"]
    assert "params" not in rows[0]  # arrays stay out of the flat table
    path = log.write_csv(tmp_path / "season.csv", dt=env.config.dt)
    header, first = path.read_text().splitlines()[:2]
    assert header.startswith("step,time_s,x_dry_weight")
    assert len(first.split(",")) == len(rows[0])


def test_mpc_plans_with_a_day_night_temperature_band():
    from lettuce_greenhouse_gym.baselines.mpc import band_limits

    base = benchmark_reward_config()
    day_night = tuple(
        PenaltyBound(
            b.name, 10.0, 15.0, b.w_lo, b.w_hi, day_lo=15.0, day_hi=20.0, day_radiation=10.0
        )
        if b.name == "indoor_temp"
        else b
        for b in base.bounds
    )
    env = LettuceGreenhouseEnv(
        EnvConfig(episode_days=0.5, reward=RewardConfig(base.economics, day_night))
    )
    mpc = NominalMPC(MPCSettings(horizon=6))
    log = run_episode(env, mpc, seed=0)
    assert np.all(np.isfinite(log.reward))
    assert all(s.success for s in mpc.solves)
    lo, hi = band_limits(day_night, log.v.T)
    temp_row = [b.name for b in day_night].index("indoor_temp")
    night = log.v[:, 0] <= 10.0
    assert set(lo[temp_row, night]) == {10.0}
    assert set(hi[temp_row, ~night]) == {20.0}


# ---- scenario MPC --------------------------------------------------------------------------------


def test_scenario_mpc_with_one_fixed_scenario_is_the_nominal_mpc():
    from lettuce_greenhouse_gym.baselines import ScenarioMPC, ScenarioMPCSettings
    from lettuce_greenhouse_gym.model.parameters import FixedParameterProvider

    env = LettuceGreenhouseEnv(EnvConfig(episode_days=0.25))
    nominal = run_episode(env, NominalMPC(MPCSettings(horizon=4)), seed=0)
    one = ScenarioMPC(
        FixedParameterProvider(env.model.constants), ScenarioMPCSettings(horizon=4, n_scenarios=1)
    )
    scenario = run_episode(env, one, seed=0)
    np.testing.assert_allclose(scenario.u, nominal.u, rtol=1e-6, atol=1e-6)
    assert scenario.total_return == pytest.approx(nominal.total_return, abs=1e-6)


def test_scenario_mpc_hedges_over_a_per_step_provider_and_reports_solves():
    from lettuce_greenhouse_gym.baselines import ScenarioMPC, ScenarioMPCSettings

    belief = RandomizedParameterProvider.relative(
        ParameterSet.from_defaults(MODEL_COEFFS), 0.1, per_step=True
    )
    env = LettuceGreenhouseEnv(EnvConfig(episode_days=0.25))
    smpc = ScenarioMPC(belief, ScenarioMPCSettings(horizon=4, n_scenarios=3, seed=1))
    log = run_episode(env, smpc, seed=0)
    assert np.all(np.isfinite(log.reward))
    assert len(smpc.solves) == env.config.n_steps
    assert log.info[0]["controller"]["iterations"] > 0
    drawn = smpc.draw_scenarios()
    assert len(drawn) == 3
    assert drawn[0].shape == (len(MODEL_COEFFS), 4)
    assert not np.allclose(drawn[0][:, 0], drawn[0][:, 1])  # per-step: redrawn along the horizon
    # the controller's own RNG: the same seed draws the same scenarios after reset
    twin = ScenarioMPC(belief, ScenarioMPCSettings(horizon=4, n_scenarios=3, seed=1))
    twin.reset(env)
    smpc.reset(env)
    np.testing.assert_array_equal(twin.draw_scenarios()[0], smpc.draw_scenarios()[0])
