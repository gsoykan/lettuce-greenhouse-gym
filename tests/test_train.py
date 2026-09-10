"""SB3 adapter: env construction, policy-as-controller, and a full evaluation of an untrained model."""

import glob

import numpy as np
import pytest

from lettuce_greenhouse_gym import EnvConfig
from lettuce_greenhouse_gym.experiment import ExperimentSpec, ParameterSpec, TrainSpec, WeatherSpec
from lettuce_greenhouse_gym.train.sb3 import (
    PolicyController,
    evaluate,
    load,
    make_model,
    make_vec_env,
    return_components,
    train,
    train_seeds,
)

sb3 = pytest.importorskip("stable_baselines3")

# a quarter day on synthetic weather: 12 steps, no data loading, seconds not minutes
SMOKE = ExperimentSpec(
    env=EnvConfig(episode_days=0.25),
    weather=WeatherSpec(source="synthetic", start_day=5.0),
    train=TrainSpec(n_envs=1, seed=0),
)


def test_make_vec_env_single_process_is_normalised():
    venv = make_vec_env(SMOKE)
    try:
        assert isinstance(venv, sb3.common.vec_env.VecNormalize)
        assert isinstance(venv.venv, sb3.common.vec_env.DummyVecEnv)
        assert venv.observation_space.shape == (12,)
    finally:
        venv.close()


@pytest.mark.parametrize("algo", ["ppo", "sac", "ddpg", "td3"])
def test_untrained_model_evaluates_a_full_episode(algo):
    spec = ExperimentSpec(
        SMOKE.env, SMOKE.weather, ParameterSpec(), TrainSpec(algo=algo, n_envs=1)
    )
    venv = make_vec_env(spec)
    try:
        model = make_model(spec, venv)
        log = evaluate(model, spec, vecnormalize=venv, seed=0)
    finally:
        venv.close()
    assert log.reward.shape == (spec.env.n_steps,)
    assert np.all(np.isfinite(log.reward))
    comps = return_components(log)
    assert comps["return"] == pytest.approx(
        comps["revenue"] - comps["energy_cost"] - comps["co2_cost"] - comps["penalty"]
    )


def test_net_arch_merges_with_user_policy_kwargs():
    """A custom policy option must not be clobbered by the net_arch shorthand."""
    spec = ExperimentSpec(
        SMOKE.env,
        SMOKE.weather,
        ParameterSpec(),
        TrainSpec(
            n_envs=1, net_arch=(32, 32), algo_kwargs={"policy_kwargs": {"log_std_init": -1.0}}
        ),
    )
    venv = make_vec_env(spec)
    try:
        model = make_model(spec, venv)
    finally:
        venv.close()
    assert model.policy_kwargs["net_arch"] == [32, 32]
    assert model.policy_kwargs["log_std_init"] == -1.0


def test_policy_controller_output_is_what_the_env_applies():
    """decode then encode is the identity in absolute mode, so the log holds the policy's own u."""
    from lettuce_greenhouse_gym.config import ActionMode
    from lettuce_greenhouse_gym.experiment import build_env

    spec = ExperimentSpec(
        EnvConfig(episode_days=0.25, action_mode=ActionMode.ABSOLUTE),
        SMOKE.weather,
        ParameterSpec(),
        SMOKE.train,
    )
    venv = make_vec_env(spec)
    try:
        model = make_model(spec, venv)
    finally:
        venv.close()
    env = build_env(spec, "eval")
    obs, _ = env.reset(seed=0)
    ctrl = PolicyController(model)
    ctrl.reset(env)
    u = ctrl.control(obs, env)
    env.step(env.encode_control(u))
    np.testing.assert_allclose(env.control, u, atol=1e-12)


@pytest.mark.parametrize(
    ("algo", "kwargs", "steps"),
    [
        ("ppo", {"n_steps": 16, "batch_size": 16, "n_epochs": 1}, 64),
        ("sac", {"learning_starts": 8, "train_freq": 1, "batch_size": 8, "buffer_size": 64}, 32),
    ],
)
def test_train_writes_a_complete_run_directory_and_evaluates(tmp_path, algo, kwargs, steps):
    spec = ExperimentSpec(
        SMOKE.env,
        SMOKE.weather,
        ParameterSpec(),
        TrainSpec(
            algo=algo, n_envs=1, total_timesteps=steps, algo_kwargs=kwargs, log_dir=str(tmp_path)
        ),
    )
    result = train(spec, run_dir=tmp_path / algo)
    assert result.model_path.exists()
    assert result.vecnormalize_path is not None
    assert result.vecnormalize_path.exists()
    assert (result.run_dir / "spec.yaml").exists()
    assert glob.glob(str(result.run_dir / "monitor" / "*.monitor.csv"))

    loaded_spec, model, vecnormalize = load(result.run_dir)
    assert loaded_spec == spec
    log = evaluate(model, loaded_spec, vecnormalize=vecnormalize, seed=0)
    assert log.reward.shape == (spec.env.n_steps,)


def test_train_without_normalisation_saves_no_statistics(tmp_path):
    spec = ExperimentSpec(
        SMOKE.env,
        SMOKE.weather,
        ParameterSpec(),
        TrainSpec(
            n_envs=1,
            total_timesteps=32,
            normalize_obs=False,
            algo_kwargs={"n_steps": 16, "batch_size": 16, "n_epochs": 1},
        ),
    )
    result = train(spec, run_dir=tmp_path / "raw")
    assert result.vecnormalize_path is None
    _, _, vecnormalize = load(result.run_dir)
    assert vecnormalize is None


def test_wandb_branch_logs_the_spec_as_config(tmp_path, monkeypatch):
    """The W&B run must receive the whole spec as its config; no network is touched here."""
    import types

    from lettuce_greenhouse_gym.experiment import WandbSpec, to_dict
    from lettuce_greenhouse_gym.train import sb3 as sb3_module

    wandb = pytest.importorskip("wandb")
    seen: dict = {}

    rows: list[tuple[int, dict]] = []

    def fake_init(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(
            log=lambda data, step: rows.append((step, data)),
            finish=lambda: seen.setdefault("finished", True),
        )

    class NoOpCallback(sb3.common.callbacks.BaseCallback):
        def _on_step(self) -> bool:
            return True

    real_import = sb3_module.importlib.import_module
    fake_module = types.SimpleNamespace(WandbCallback=NoOpCallback)

    def import_module(name, *args):
        return fake_module if name == "wandb.integration.sb3" else real_import(name, *args)

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(sb3_module.importlib, "import_module", import_module)
    spec = ExperimentSpec(
        SMOKE.env,
        SMOKE.weather,
        ParameterSpec(),
        TrainSpec(
            n_envs=1,
            total_timesteps=32,
            algo_kwargs={"n_steps": 16, "batch_size": 16, "n_epochs": 1},
            wandb=WandbSpec(project="p", tags=("a", "b")),
        ),
    )
    train(spec, run_dir=tmp_path / "wb")
    assert seen["project"] == "p"
    assert seen["tags"] == ["a", "b"]
    assert seen["config"] == to_dict(spec)
    assert seen["finished"] is True
    # metrics are written directly, one complete row per logger dump, with monotonic steps
    logged = {k for _, data in rows for k in data}
    assert {"episode/penalty", "rollout/ep_rew_mean", "train/loss"} <= logged
    steps = [step for step, _ in rows]
    assert steps == sorted(steps)


def test_eval_and_checkpoint_callbacks_write_their_artifacts(tmp_path):
    """Periodic evaluation keeps a best model; checkpoints land with their normalisation stats."""
    from lettuce_greenhouse_gym.experiment import SplitSpec

    spec = ExperimentSpec(
        SMOKE.env,
        WeatherSpec(source="synthetic", start_day=None, split=SplitSpec(0.5, 0)),
        ParameterSpec(),
        TrainSpec(
            n_envs=1,
            total_timesteps=64,
            algo_kwargs={"n_steps": 16, "batch_size": 16, "n_epochs": 1},
            eval_every=32,
            checkpoint_every=32,
        ),
    )
    result = train(spec, run_dir=tmp_path / "periodic")
    assert (result.run_dir / "best" / "best_model.zip").exists()
    assert (result.run_dir / "eval" / "evaluations.npz").exists()
    checkpoints = sorted((result.run_dir / "checkpoints").glob("*.zip"))
    assert len(checkpoints) >= 2  # model + vecnormalize at least once
    assert any("vecnormalize" in c.name for c in (result.run_dir / "checkpoints").glob("*.pkl"))


def test_train_seeds_makes_one_run_per_seed(tmp_path):
    spec = ExperimentSpec(
        SMOKE.env,
        SMOKE.weather,
        ParameterSpec(),
        TrainSpec(
            n_envs=1,
            total_timesteps=32,
            algo_kwargs={"n_steps": 16, "batch_size": 16, "n_epochs": 1},
            log_dir=str(tmp_path),
            run_name="multi",
        ),
    )
    results = train_seeds(spec, [3, 4])
    assert [r.run_dir.name for r in results] == ["multi_seed3", "multi_seed4"]
    assert [load(r.run_dir)[0].train.seed for r in results] == [3, 4]


def test_train_spec_rejects_non_positive_cadences():
    with pytest.raises(ValueError, match="eval_every"):
        TrainSpec(eval_every=0)
    with pytest.raises(ValueError, match="checkpoint_every"):
        TrainSpec(checkpoint_every=-5)


def test_plain_data_spellings_become_sb3_objects():
    spec = ExperimentSpec(
        SMOKE.env,
        SMOKE.weather,
        ParameterSpec(),
        TrainSpec(
            algo="sac",
            n_envs=1,
            action_noise_sigma=0.05,
            algo_kwargs={
                "learning_rate": "lin_5e-3",
                "policy_kwargs": {"activation_fn": "relu", "net_arch": {"pi": [8], "qf": [8]}},
                "buffer_size": 64,
            },
        ),
    )
    venv = make_vec_env(spec)
    try:
        model = make_model(spec, venv)
    finally:
        venv.close()
    torch = pytest.importorskip("torch")
    noise = pytest.importorskip("stable_baselines3.common.noise")
    assert model.lr_schedule(1.0) == pytest.approx(5e-3)
    assert model.lr_schedule(0.5) == pytest.approx(2.5e-3)
    assert model.policy_kwargs["activation_fn"] is torch.nn.ReLU
    assert isinstance(model.action_noise, noise.NormalActionNoise)
    np.testing.assert_allclose(model.action_noise._sigma, 0.05)


def test_action_noise_is_off_policy_only_and_lr_strings_are_validated():
    with pytest.raises(ValueError, match="applies to"):
        TrainSpec(algo="ppo", action_noise_sigma=0.1)
    with pytest.raises(ValueError, match="action_noise must be"):
        TrainSpec(algo="sac", action_noise="pink")
    with pytest.raises(ValueError, match="algo must be"):
        TrainSpec(algo="dqn")
    spec = ExperimentSpec(
        SMOKE.env,
        SMOKE.weather,
        ParameterSpec(),
        TrainSpec(n_envs=1, algo_kwargs={"learning_rate": "cosine"}),
    )
    venv = make_vec_env(spec)
    try:
        with pytest.raises(ValueError, match="lin_"):
            make_model(spec, venv)
    finally:
        venv.close()


def test_ornstein_uhlenbeck_noise_reaches_an_off_policy_model():
    noise = pytest.importorskip("stable_baselines3.common.noise")
    spec = ExperimentSpec(
        SMOKE.env,
        SMOKE.weather,
        ParameterSpec(),
        TrainSpec(
            algo="ddpg", n_envs=1, action_noise_sigma=0.3, action_noise="ornstein_uhlenbeck"
        ),
    )
    venv = make_vec_env(spec)
    try:
        model = make_model(spec, venv)
    finally:
        venv.close()
    assert isinstance(model.action_noise, noise.OrnsteinUhlenbeckActionNoise)
    np.testing.assert_allclose(model.action_noise._sigma, 0.3)


def test_evaluation_logs_the_reward_components(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from lettuce_greenhouse_gym.train.callbacks import ComponentEvalCallback

    spec = ExperimentSpec(SMOKE.env, SMOKE.weather, ParameterSpec(), TrainSpec(n_envs=1))
    venv = make_vec_env(spec)
    eval_env = make_vec_env(spec, role="eval", n_envs=1, seed=1)
    try:
        model = make_model(spec, venv)
        model.set_logger(sb3.common.logger.configure(str(tmp_path), ["csv"]))
        n = spec.env.n_steps
        callback = ComponentEvalCallback(eval_env, eval_freq=n, n_eval_episodes=1, verbose=0)
        model.learn(total_timesteps=n, callback=callback)
    finally:
        venv.close()
        eval_env.close()
    header = (tmp_path / "progress.csv").read_text().splitlines()[0].split(",")
    assert {"eval/mean_reward", "eval/revenue", "eval/penalty", "eval/energy_cost"} <= set(header)


def test_training_and_evaluation_see_the_same_wrapped_observation(tmp_path):
    from dataclasses import replace

    from lettuce_greenhouse_gym.experiment import CallableSpec

    spec = replace(
        SMOKE,
        parameters=ParameterSpec(ranges={"leak": (0.5e-5, 1.5e-5)}),
        wrappers=(
            CallableSpec(
                "lettuce_greenhouse_gym.wrappers:ParameterObservation",
                {"names": ["leak"], "ranges": {"leak": [0.5e-5, 1.5e-5]}},
            ),
        ),
        train=replace(SMOKE.train, algo="sac", total_timesteps=12, log_dir=str(tmp_path)),
    )
    venv = make_vec_env(spec)
    assert venv.observation_space.shape[0] == make_vec_env(SMOKE).observation_space.shape[0] + 1
    venv.close()
    result = train(spec)
    loaded_spec, model, vecnormalize = load(result.run_dir)
    assert loaded_spec == spec
    log = evaluate(model, loaded_spec, vecnormalize=vecnormalize)
    assert log.u.shape == (spec.env.n_steps, 3)
