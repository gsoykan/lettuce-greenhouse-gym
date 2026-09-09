"""SB3 callbacks: reward components per episode, and a direct Weights & Biases writer."""

import importlib
from typing import TYPE_CHECKING, Any

import numpy as np

from .._optional import require

if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.logger import KVWriter
else:  # the train extra is optional: resolve the bases only when this module is imported
    BaseCallback = require("stable_baselines3").common.callbacks.BaseCallback
    EvalCallback = require("stable_baselines3").common.callbacks.EvalCallback
    KVWriter = importlib.import_module("stable_baselines3.common.logger").KVWriter

_KEYS = ("revenue", "energy_cost", "co2_cost", "penalty")


class ReturnComponentsCallback(BaseCallback):
    """Log season totals of revenue, costs and penalty, averaged over the episodes of each rollout.

    ``info`` carries per-step values; a season total is the sum over the episode, so a running sum
    per env is kept and flushed on ``done``. ``VecMonitor`` cannot do this: it records only the
    final step's value of an info key.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sums = np.zeros((0, len(_KEYS)))  # sized once the vector env is known

    def _on_training_start(self) -> None:
        self._sums = np.zeros((self.training_env.num_envs, len(_KEYS)))

    def _on_step(self) -> bool:
        for i, (info, done) in enumerate(
            zip(self.locals["infos"], self.locals["dones"], strict=True)
        ):
            self._sums[i] += [info[k] for k in _KEYS]
            if done:
                for k, total in zip(_KEYS, self._sums[i], strict=True):
                    self.logger.record_mean(f"episode/{k}", float(total))
                self._sums[i] = 0.0
        return True


class ComponentEvalCallback(EvalCallback):
    """SB3's evaluation callback, also logging the season totals of the reward components.

    Evaluation otherwise reports the return alone; a season's revenue, costs and penalty say *why*
    it changed. Sums per episode are collected during ``evaluate_policy`` and logged as
    ``eval/<component>`` means at the same timestep as ``eval/mean_reward``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._running: dict[int, np.ndarray] = {}
        self._totals: list[np.ndarray] = []

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)
        i, info = locals_["i"], locals_["info"]
        acc = self._running.setdefault(i, np.zeros(len(_KEYS)))
        acc += [info.get(k, 0.0) for k in _KEYS]
        if locals_["done"]:
            self._totals.append(acc.copy())
            acc[:] = 0.0

    def _on_step(self) -> bool:
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluating:
            self._running, self._totals = {}, []
        continue_training = super()._on_step()
        if evaluating and self._totals:
            means = np.mean(self._totals, axis=0)
            for k, m in zip(_KEYS, means, strict=True):
                self.logger.record(f"eval/{k}", float(m))
            self.logger.dump(self.num_timesteps)
        return continue_training


class _WandbWriter(KVWriter):
    """One complete W&B history row per SB3 logger dump."""

    def __init__(self, run) -> None:
        self._run = run

    def write(
        self, key_values: dict[str, Any], key_excluded: dict[str, Any], step: int = 0
    ) -> None:
        scalars = {
            k: float(v) for k, v in key_values.items() if isinstance(v, int | float | np.number)
        }
        if scalars:
            self._run.log(scalars, step=step)

    def close(self) -> None:
        return None


class WandbLoggerCallback(BaseCallback):
    """Mirror every SB3 logger dump (rollout/*, train/*, episode/*) to a W&B run.

    Written directly rather than through W&B's TensorBoard syncing: the rows are then ordinary
    history records, so offline runs upload with a plain ``wandb sync``, and TensorBoard need not be
    installed for W&B logging to work.
    """

    def __init__(self, run) -> None:
        super().__init__()
        self._run = run

    def _on_training_start(self) -> None:
        self.logger.output_formats.append(_WandbWriter(self._run))

    def _on_step(self) -> bool:
        return True
