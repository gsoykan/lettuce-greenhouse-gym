# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `ExperimentSpec.wrappers`: Gymnasium wrappers named as callable specs and applied, innermost
  first, around the env for training and evaluation alike (`apply_wrappers`). `run_episode` accepts
  a wrapped env: resets and steps go through the wrapper, the plant is read from the bare env.
- `LettuceGreenhouseEnv.parameters_used` and `ParameterObservation(..., used=True)`: the coefficient
  set that drove the last transition, the "told after the fact" disclosure under per-step schemes.

## [0.1.2] - 2026-09-10

### Fixed

- A provider's per-transition draw now happens **before** the observation that precedes the
  transition, at reset for transition 0 and after each transition for the next. Previously the draw
  happened inside `step()` after the action was chosen, so a wrapper appending the coefficients to
  the observation told the policy the *previous* draw under per-step randomisation. `info["params"]`
  now reports the coefficients the next transition will use; the new `info["params_used"]` keeps the
  set that drove the last one, so both disclosure conventions can be built. Per-season providers are
  unaffected; per-step RNG streams change.

### Added

- `EnvConfig.terminal_at_harvest`: report the end of the season as `truncated` instead of
  `terminated`, for formulations that bootstrap through the horizon. `run_episode` stops on either.
- `LettuceGreenhouseEnv.observation_layout`: the observation's blocks as `(name, slice)` pairs.

## [0.1.1] - 2026-09-09

### Changed

- PyYAML is a core dependency instead of part of the `train` extra. Experiment specs are YAML and
  the command line parses `--set` values and `--config` files as YAML, so on a core-only install
  `lettuce-gym baselines --set env.episode_days=1` failed with a hint to install the training extra.

## [0.1.0] - 2026-09-09

First release of the van Henten lettuce greenhouse as a Gymnasium environment.

### Added

- `LettuceGreenhouse-v0`: the four-state van Henten model (1994, 2003) as a CasADi RK4 integrator,
  the benchmark economic reward with soft comfort penalties, and the 2014 Bleiswijk weather trace.
  Every default reproduces the published RL/MPC benchmark (40 days from 9 February 2014, 30-minute
  steps, delta actions).
- `EnvConfig` with actuator, initial-state and rate-limit overrides, absolute or delta actions, a
  configurable observation layout (previous control, season clock as `progress` or `index`, weather
  window); constructor injection for the dynamics model, reward, parameter provider, weather
  sampler and weather repository. `PenaltyBound` may carry a daytime range switched on by outdoor
  radiation, honoured by the reward and both MPCs. `ParameterObservation` is the wrapper that
  appends chosen coefficients to the observation, rescaled by a caller-chosen range.
- The dynamics-model contract in two layers: `DynamicsModel` asks only for `build_integrator(dt)`
  and `build_measurement()` returning plain callables `F(x, u, v, c)` and `g(x)`, so numeric and
  learned models implement it directly; `SymbolicDynamicsModel`, the base of `VanHentenLettuce`,
  takes `rhs` and `measurement` as CasADi expressions and compiles them. The env validates a
  model's output widths at construction; `NominalMPC` requires the symbolic layer and says so.
- Parameters as a hook, not a policy: `ParameterProvider.sample(rng)` at reset and
  `step(rng, step_index, current)` before every transition, so domain randomisation, per-step
  noise, drift or faults are a subclass away. `RandomizedParameterProvider` ships absolute ranges
  and `.relative(base, half_width, names, per_step)`; the drawn set is visible in `info["params"]`.
- Public control API for model-based methods: `state`, `control`, `parameters`, `integrator`,
  `measurement`, `weather_forecast`, `encode_control`, `decode_action`; and for branching an
  episode, `snapshot()` / `restore(EnvState)` and `observe(x, u_prev, step_index)`, so a planner can
  roll a policy out from the current moment and come back, or evaluate it at a hypothetical state.
- `EnvConfig.absolute_rate_limit`: in absolute action mode, clip a level to within `du_max` of the
  previous control.
- Weather through one protocol: a `WeatherLoader` is any callable returning a `WeatherSeries`,
  registered by name in a `WeatherRepository`. Packaged: the Bleiswijk 2014 trace and a synthetic
  generator. `lettuce_greenhouse_gym.knmi` fetches KNMI hourly station-years (CC BY 4.0) into the
  package's CSV format with an attribution note (CLI `lettuce-gym weather fetch-knmi` / `list`);
  `weather.write_csv` writes that format from physical channels. Samplers draw a source and a start
  day per episode or cycle through them; `split_start_days` holds out days of one trace.
  `docs/custom_weather.md` is the guide, with `examples/weather_txt.py` and
  `examples/weather_epw.py` as complete loaders. `WeatherPerturbation` adds two identity-default
  hook points on an episode's weather: `realise` (what the plant experiences) and `forecast` (what
  controllers are told); the env checks shape and physical bounds of what comes back.
- Reference controllers in `baselines/`: `AllOff`, `ConstantControl`, `GrowerHeuristic` (the
  grower regime of van Henten's validation experiments) and `NominalMPC` (multiple shooting,
  slack-variable comfort bands, IPOPT), with `run_episode` and `EpisodeLog`. `MPCSettings` selects
  the planning coefficients (nominal, the episode's true ones, or any supplied set) and the planning
  comfort bands independently of the reward; every solve is recorded as a `SolveStats` and
  `Controller.step_info()` puts it under `info["controller"]` in the log. `ScenarioMPC` hedges over
  coefficient sequences drawn from a `ParameterProvider`, one shared control plan against the
  expected reward; with one fixed scenario it reproduces `NominalMPC`. `EpisodeLog.to_records` /
  `write_csv` flatten a season into one row per step.
- `experiment/`: `ExperimentSpec` as a serialisable experiment record (YAML in and out) and
  builders that turn it into a live environment. Weather specs name packaged traces, CSV files
  (`files`) or arbitrary loaders (`loaders`, as `package.module:function` or
  `script.py:function` with keyword arguments), several training sources, and separate evaluation
  sources and days, and a `WeatherPerturbation` (`perturbation`); `parameters` selects `ranges` or
  `relative`, once per episode or `per_step`, or any `ParameterProvider` through `provider`. Callable
  specs (`target: package.module:function` or `script.py:function`, plus `kwargs`) are the one form
  for loaders, perturbations, samplers (`weather.sampler`, `eval_sampler`) and providers.
- `[train]` extra: Stable-Baselines3 PPO, SAC, DDPG and TD3 with vectorised environments, observation
  normalisation, reward-component logging to TensorBoard and Weights & Biases, evaluation through
  the same runner as the reference controllers (periodic evaluation logs the reward components
  beside the mean return), run provenance in `run.json`, and the `lettuce-gym` command line
  (`train`, `eval`, `baselines`, `weather`); periodic evaluation on held-out weather with
  best-model tracking, checkpoints, multi-seed runs, and plain-data spellings for SB3 objects in a
  spec (`learning_rate: lin_<x>`, `activation_fn: relu`, `action_noise_sigma` with `action_noise`
  Gaussian or Ornstein-Uhlenbeck).
  `examples/benchmark.yaml`, `benchmark_sac.yaml` (the published RL baseline's setup) and
  `benchmark_sac_knmi.yaml` (its train-on-KNMI, evaluate-on-Bleiswijk protocol).
- `[plot]` extra: `plot_episode`, a season figure of observables, controls, weather and reward.
- Numerical parity tests against the published benchmark's dynamics, measurement and reward
  (1e-15, 0, 1e-17), and a closed-loop cross-check reproducing the benchmark MPC's reported return
  to 1e-11. The package is type-checked with mypy in CI, backing its `Typing :: Typed` classifier.

### Notes

- The benchmark's reference code starts its season on 8 February 2014 through an off-by-one in
  file-offset arithmetic; this package follows the paper's stated 9 February (`start_day=40`).
  `env.reset(options={"start_day": 39.0})` reproduces the code's window.
