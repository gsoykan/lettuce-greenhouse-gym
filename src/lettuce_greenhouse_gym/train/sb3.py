"""Stable-Baselines3 adapter: vectorised envs, PPO/SAC construction, policy evaluation."""

import importlib
import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._optional import require
from ..baselines import Controller, EpisodeLog, run_episode
from ..envs.control_env import LettuceGreenhouseEnv
from ..experiment import ExperimentSpec, Role, build_env, load_yaml, save_yaml, to_dict
from ..provenance import write_run_metadata


def _make_env(spec: ExperimentSpec, role: Role) -> LettuceGreenhouseEnv:
    """Module-level so ``SubprocVecEnv`` can pickle it; a lambda would fail on macOS's spawn start."""
    return build_env(spec, role)


def make_vec_env(
    spec: ExperimentSpec,
    *,
    role: Role = "train",
    n_envs: int | None = None,
    seed: int | None = None,
    monitor_dir: str | Path | None = None,
):
    """``n_envs`` parallel copies of the spec's env; one process when ``n_envs == 1``."""
    env_util = require("stable_baselines3").common.env_util
    vec_env = require("stable_baselines3").common.vec_env
    n = n_envs if n_envs is not None else spec.train.n_envs
    venv = env_util.make_vec_env(
        partial(_make_env, spec, role),
        n_envs=n,
        seed=seed if seed is not None else spec.train.seed,
        monitor_dir=None if monitor_dir is None else str(monitor_dir),
        vec_env_cls=vec_env.DummyVecEnv if n == 1 else vec_env.SubprocVecEnv,
    )
    if spec.train.normalize_obs:
        venv = vec_env.VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv


class LinearSchedule:
    """``initial * progress_remaining``: decays to zero at the end of training.

    A class rather than a closure so a saved model reloads with plain pickle.
    """

    def __init__(self, initial: float) -> None:
        self.initial = initial

    def __call__(self, progress_remaining: float) -> float:
        return self.initial * progress_remaining


_ACTIVATIONS = {"relu": "ReLU", "tanh": "Tanh", "elu": "ELU", "leaky_relu": "LeakyReLU"}


def _resolve_algo_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the spec's plain-data spellings into the objects SB3 wants.

    ``learning_rate: "lin_5e-3"`` -> a linear schedule from 5e-3 to 0;
    ``policy_kwargs: {activation_fn: relu}`` -> ``torch.nn.ReLU``. Anything else passes through.
    """
    out = dict(kwargs)
    lr = out.get("learning_rate")
    if isinstance(lr, str):
        if not lr.startswith("lin_"):
            raise ValueError(f"learning_rate strings must look like 'lin_5e-3', got {lr!r}")
        out["learning_rate"] = LinearSchedule(float(lr[4:]))
    policy_kwargs = dict(out.get("policy_kwargs") or {})
    activation = policy_kwargs.get("activation_fn")
    if isinstance(activation, str):
        nn = require("torch").nn
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation_fn {activation!r}; one of {sorted(_ACTIVATIONS)}"
            )
        policy_kwargs["activation_fn"] = getattr(nn, _ACTIVATIONS[activation])
        out["policy_kwargs"] = policy_kwargs
    return out


def _algorithm(name: str):
    """The SB3 class for a ``TrainSpec.algo`` name."""
    sb3 = require("stable_baselines3")
    return {"ppo": sb3.PPO, "sac": sb3.SAC, "ddpg": sb3.DDPG, "td3": sb3.TD3}[name]


def make_model(spec: ExperimentSpec, venv, *, tensorboard_log: str | Path | None = None):
    """PPO, SAC, DDPG or TD3 on ``spec.train.policy``, with ``algo_kwargs`` passed through unchanged.

    ``net_arch`` is merged into ``policy_kwargs`` rather than replacing them, so a custom feature
    extractor or any other policy option given through ``algo_kwargs`` survives.
    """
    kwargs = _resolve_algo_kwargs(spec.train.algo_kwargs)
    if spec.train.net_arch is not None:
        kwargs["policy_kwargs"] = {
            **kwargs.get("policy_kwargs", {}),
            "net_arch": list(spec.train.net_arch),
        }
    if spec.train.action_noise_sigma is not None:
        n_actions = venv.action_space.shape[0]
        noise = importlib.import_module("stable_baselines3.common.noise")
        noise_cls = {
            "normal": noise.NormalActionNoise,
            "ornstein_uhlenbeck": noise.OrnsteinUhlenbeckActionNoise,
        }[spec.train.action_noise]
        kwargs["action_noise"] = noise_cls(
            mean=np.zeros(n_actions), sigma=spec.train.action_noise_sigma * np.ones(n_actions)
        )
    return _algorithm(spec.train.algo)(
        spec.train.policy,
        venv,
        seed=spec.train.seed,
        device=spec.train.device,
        tensorboard_log=None if tensorboard_log is None else str(tensorboard_log),
        **kwargs,
    )


class PolicyController(Controller):
    """An SB3 policy driving the env through the same runner as the reference controllers.

    ``vecnormalize`` is the statistics object saved with the model; observations are normalised
    with it exactly as during training. Not privileged: it sees only the observation.
    """

    def __init__(self, model, vecnormalize=None, *, deterministic: bool = True) -> None:
        self.model = model
        self.vecnormalize = vecnormalize
        self.deterministic = deterministic

    def control(self, obs: NDArray[np.float32], env: LettuceGreenhouseEnv) -> NDArray[np.float64]:
        if self.vecnormalize is not None:
            obs = self.vecnormalize.normalize_obs(obs[None])[0]
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return env.decode_action(action)


def evaluate(
    model,
    spec: ExperimentSpec,
    *,
    vecnormalize=None,
    seed: int = 0,
    deterministic: bool = True,
    options: dict[str, Any] | None = None,
) -> EpisodeLog:
    """One full season of the policy on the spec's evaluation env."""
    controller = PolicyController(model, vecnormalize, deterministic=deterministic)
    return run_episode(build_env(spec, "eval"), controller, seed=seed, options=options)


def return_components(log: EpisodeLog) -> dict[str, float]:
    """Season totals of the reward's parts, for tables."""
    keys = ("revenue", "energy_cost", "co2_cost", "penalty")
    return {k: float(sum(step[k] for step in log.info)) for k in keys} | {
        "return": log.total_return
    }


@dataclass(frozen=True)
class TrainResult:
    """Where a run left its artifacts."""

    run_dir: Path
    model_path: Path
    vecnormalize_path: Path | None
    spec: ExperimentSpec
    timesteps: int


def _run_dir(spec: ExperimentSpec) -> Path:
    name = (
        spec.train.run_name
        or f"{spec.train.algo}_{spec.train.seed}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    return Path(spec.train.log_dir) / name


def train(spec: ExperimentSpec, *, run_dir: str | Path | None = None) -> TrainResult:
    """Train per ``spec`` and save model, normalisation statistics and the spec itself to one dir."""
    require("stable_baselines3")
    from .callbacks import ReturnComponentsCallback, WandbLoggerCallback

    run_dir = Path(run_dir) if run_dir is not None else _run_dir(spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    # the experiment record, written before anything can fail: what was run, and with which build
    save_yaml(spec, run_dir / "spec.yaml")
    write_run_metadata(run_dir)

    tb = str(run_dir) if importlib.util.find_spec("tensorboard") else None
    venv = make_vec_env(spec, monitor_dir=run_dir / "monitor")
    eval_env = None
    wandb_run = None
    try:
        model = make_model(spec, venv, tensorboard_log=tb)
        callbacks: list[Any] = [ReturnComponentsCallback()]
        callbacks += _periodic_callbacks(spec, run_dir)
        if spec.train.eval_every is not None:
            eval_env, eval_callback = _eval_callback(spec, run_dir)
            callbacks.append(eval_callback)
        if spec.train.wandb is not None:
            wandb = require("wandb")
            wandb_run = wandb.init(
                project=spec.train.wandb.project,
                entity=spec.train.wandb.entity,
                tags=list(spec.train.wandb.tags),
                name=run_dir.name,
                config=to_dict(spec),
            )
            # a subpackage wandb does not import eagerly
            sb3_integration = importlib.import_module("wandb.integration.sb3")
            callbacks += [WandbLoggerCallback(wandb_run), sb3_integration.WandbCallback()]
        model.learn(total_timesteps=spec.train.total_timesteps, callback=callbacks)
        model_path = run_dir / "model.zip"
        model.save(model_path)
        vecnormalize_path = None
        if spec.train.normalize_obs:
            vecnormalize_path = run_dir / "vecnormalize.pkl"
            venv.save(vecnormalize_path)
        write_run_metadata(
            run_dir,
            finished=datetime.now(UTC).isoformat(timespec="seconds"),
            timesteps=spec.train.total_timesteps,
        )
    finally:
        venv.close()  # SubprocVecEnv workers must be shut down even when learn() raises
        if eval_env is not None:
            eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()
    return TrainResult(run_dir, model_path, vecnormalize_path, spec, spec.train.total_timesteps)


def _per_env_freq(every: int, n_envs: int) -> int:
    """SB3 counts callback frequency in calls, i.e. per vectorised step, not per env step."""
    return max(1, every // n_envs)


def _periodic_callbacks(spec: ExperimentSpec, run_dir: Path) -> list:
    sb3 = require("stable_baselines3")
    if spec.train.checkpoint_every is None:
        return []
    return [
        sb3.common.callbacks.CheckpointCallback(
            save_freq=_per_env_freq(spec.train.checkpoint_every, spec.train.n_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="model",
            save_vecnormalize=spec.train.normalize_obs,
        )
    ]


def _eval_callback(spec: ExperimentSpec, run_dir: Path):
    """Evaluation on the spec's *eval* weather (the held-out days under a split).

    The eval env carries its own ``VecNormalize`` in inference mode; SB3 copies the training
    statistics into it before each evaluation, so the policy sees observations as it will at test.
    """
    if spec.train.eval_every is None:
        raise ValueError("eval_every is not set")
    eval_env = make_vec_env(spec, role="eval", n_envs=1, seed=spec.train.seed + 10_000)
    if spec.train.normalize_obs:
        eval_env.training = False
        eval_env.norm_reward = False
    from .callbacks import ComponentEvalCallback

    callback = ComponentEvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=_per_env_freq(spec.train.eval_every, spec.train.n_envs),
        n_eval_episodes=spec.train.n_eval_episodes,
        deterministic=True,
        verbose=0,
    )
    return eval_env, callback


def train_seeds(spec: ExperimentSpec, seeds: Sequence[int]) -> list[TrainResult]:
    """One run per seed, sequentially, as ``<run_name>_seed<k>`` under the spec's ``log_dir``."""
    base = spec.train.run_name or f"{spec.train.algo}_{datetime.now():%Y%m%d-%H%M%S}"
    results = []
    for seed in seeds:
        train_spec = replace(spec.train, seed=int(seed), run_name=f"{base}_seed{seed}")
        results.append(train(replace(spec, train=train_spec)))
    return results


def load(run_dir: str | Path):
    """``(spec, model, vecnormalize)`` from a directory written by :func:`train`."""
    sb3 = require("stable_baselines3")
    run_dir = Path(run_dir)
    spec = load_yaml(run_dir / "spec.yaml")
    model = _algorithm(spec.train.algo).load(run_dir / "model.zip", device=spec.train.device)
    vecnormalize = None
    pkl = run_dir / "vecnormalize.pkl"
    if pkl.exists():
        dummy = sb3.common.vec_env.DummyVecEnv([partial(_make_env, spec, "eval")])
        vecnormalize = sb3.common.vec_env.VecNormalize.load(str(pkl), dummy)
        vecnormalize.training = False
    return spec, model, vecnormalize
