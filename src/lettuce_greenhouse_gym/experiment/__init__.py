"""A serialisable experiment: env config, weather choice, parameter randomisation, training.

``ExperimentSpec()`` is the benchmark. Everything here is plain data, so a spec can be written to
YAML as the record of a run and read back to rebuild the same environment (see ``build_env``).
Every key is optional and defaults to the benchmark; the canonical form written by ``save_yaml``::

    env:                       # EnvConfig, field for field
      dt: 1800.0
      episode_days: 40.0
      action_mode: delta       # or absolute
      absolute_rate_limit: false   # absolute mode only: clip the level to du_max around the previous control
      include_previous_control: true
      include_timestep: true
      terminal_at_harvest: true    # false: report the season's end as truncated (bootstrap through it)
      timestep_encoding: progress   # or index (raw step counter)
      weather_window: 1
      reward:
        prices: {energy_cost: 0.1281, co2_cost: 0.1906, product_price_1: 0.8168, product_price_2: 22.285125}
        bounds:                # comfort bands; omit to keep the benchmark's
          - {name: indoor_temp, lo: 10.0, hi: 20.0, w_lo: 0.003, w_hi: 0.005}
          #   optional day band: day_lo: 15.0, day_hi: 20.0, day_radiation: 10.0 (W/m2 threshold)
      control_overrides: [{name: heating, lo: null, hi: 200.0, du_max: null}]
      initial_state: []
      initial_control: []
    weather:                   # exactly one of start_day / start_days / split
      source: bleiswijk_2014   # a packaged name, a key of files, or a list of them (random per episode)
      start_day: 40.0
      start_days: null
      split: null              # or {train_fraction: 0.8, seed: 0}
      files: {}                # user CSV traces: {knmi_344_2010: {path: weather/knmi_344_2010.csv, epoch_day: 1.0}}
      eval_source: null        # evaluation trace(s); null = the training ones
      eval_start_days: null    # evaluation days; null = the training days
      loaders: {}              # any other source: {epw: {target: my_weather:load_epw, kwargs: {path: site.epw}}}
      perturbation: null       # a WeatherPerturbation: {target: my_study:ForecastNoise, kwargs: {sigma: 0.1}}
      sampler: null            # a WeatherSampler replacing the start-day modes for training
      eval_sampler: null       # ... and for evaluation (else the day modes, else `sampler` afresh)
    parameters:                # how the dynamics coefficients vary; all empty/null = nominal
      ranges: {}               # absolute per-name [lo, hi], e.g. {leak: [2.0e-05, 3.0e-05]}, or
      relative: null           # every coefficient x (1 + U(-relative, relative)), e.g. 0.05
      names: null              # restrict `relative` to these coefficients
      per_step: false          # redraw before every transition instead of once per episode
      provider: null           # any other scheme: {target: my_study:PoolProvider, kwargs: {...}},
                               # called as target(constants, **kwargs) -> ParameterProvider
    train:
      algo: ppo                # or sac, ddpg, td3
      policy: MlpPolicy        # any SB3 policy name
      total_timesteps: 1000000
      n_envs: 4
      seed: 0
      normalize_obs: true
      net_arch: null           # or [64, 64]
      algo_kwargs: {}          # passed to the SB3 constructor unchanged
      device: auto
      log_dir: runs
      run_name: null
      wandb: null              # or {project: my-project, entity: null, tags: []}
      eval_every: null         # env steps between evaluations on the eval weather; keeps <run>/best/
      n_eval_episodes: 1
      checkpoint_every: null   # env steps between checkpoints in <run>/checkpoints/
      action_noise_sigma: null # off-policy algorithms: exploration noise, fraction of [-1, 1]
      action_noise: normal     # or ornstein_uhlenbeck

The package splits by consumer: :mod:`.spec` (the dataclasses), :mod:`.build` (spec -> live
objects, used by training), :mod:`.serialize` (spec <-> dict/YAML, used by the CLI).
"""

from .build import (
    Role,
    build_env,
    build_loader,
    build_perturbation,
    build_provider,
    build_repository,
    build_sampler,
    resolve_target,
)
from .serialize import anchor_files, from_dict, load_yaml, parse_yaml, save_yaml, to_dict
from .spec import (
    CallableSpec,
    ExperimentSpec,
    ParameterSpec,
    SplitSpec,
    TrainSpec,
    WandbSpec,
    WeatherFileSpec,
    WeatherLoaderSpec,
    WeatherSpec,
)

__all__ = [
    "CallableSpec",
    "ExperimentSpec",
    "ParameterSpec",
    "Role",
    "SplitSpec",
    "TrainSpec",
    "WandbSpec",
    "WeatherFileSpec",
    "WeatherLoaderSpec",
    "WeatherSpec",
    "anchor_files",
    "build_env",
    "build_loader",
    "build_perturbation",
    "build_provider",
    "build_repository",
    "build_sampler",
    "from_dict",
    "load_yaml",
    "parse_yaml",
    "resolve_target",
    "save_yaml",
    "to_dict",
]
