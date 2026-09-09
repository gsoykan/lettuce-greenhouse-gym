# lettuce-greenhouse-gym

A [Gymnasium](https://gymnasium.farama.org/) environment for the van Henten lettuce greenhouse:
the reduced four-state climate/crop model that has become the standard benchmark for comparing
reinforcement learning, model predictive control, and hybrid controllers in greenhouse climate
management.

The package reimplements the model from the published equations, with three runtime dependencies
and type-checked in CI. It ships the dynamics as a [CasADi](https://web.casadi.org/) integrator, so
the same function the env steps can be embedded in an MPC, the standard economic reward with soft
comfort penalties, the measured 2014 Bleiswijk weather trace the benchmark is defined on, and the
configuration needed for control, RL, and learned-dynamics studies.

```python
import gymnasium as gym
import lettuce_greenhouse_gym  # registers "LettuceGreenhouse-v0"

env = gym.make("LettuceGreenhouse-v0")      # every default = the published benchmark
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Installation

Requires Python 3.11 or newer.

```bash
pip install git+https://github.com/gsoykan/lettuce-greenhouse-gym.git@v0.1.0   # from GitHub
uv add git+https://github.com/gsoykan/lettuce-greenhouse-gym --tag v0.1.0        # or, in a uv project
```

A PyPI release (`pip install lettuce-greenhouse-gym`) is planned; until then install from GitHub.

Runtime dependencies: `gymnasium`, `numpy`, `casadi`. The 6 MB weather trace is packaged with the
wheel; nothing is downloaded at runtime.
Optional extras: `[train]` (Stable-Baselines3, torch, PyYAML, TensorBoard, wandb), `[plot]`
(matplotlib), `[all]`.

## The model in one screen

Four states, three controls, four weather disturbances, four grower-facing observables. Vector
order everywhere in the package is fixed by the `STATE`, `CONTROL`, `EXOGENOUS`, and `OBSERVABLE`
groups, which also carry the units and physical bounds.

| vector | components (in order) | units |
|---|---|---|
| state `x` | `dry_weight`, `indoor_co2`, `indoor_temp`, `indoor_vapor` | kg/m², kg/m³, °C, kg/m³ |
| control `u` | `co2_supply`, `ventilation`, `heating` | mg/m²/s, mm/s, W/m² |
| weather `v` | `rad`, `out_co2`, `out_temp`, `out_vapor` | W/m², kg/m³, °C, kg/m³ |
| observables `y = g(x)` | `dry_weight`, `co2_ppm`, `indoor_temp`, `rh` | kg/m², ppm, °C, % |

Benchmark actuator limits are `co2_supply ∈ [0, 1.2]`, `ventilation ∈ [0, 7.5]`,
`heating ∈ [0, 150]`, with per-step rate limits `du_max = [0.12, 0.75, 15]`; the season starts
from `x0 = [0.0035, 0.001, 15, 0.008]` and `u0 = [0, 0, 50]`. All of these are overridable (see
*Configuring experiments*).

```python
from lettuce_greenhouse_gym import STATE, CONTROL
print(CONTROL.names, CONTROL.bounds())      # names and (lo, hi) arrays
x = STATE.unpack(env.unwrapped.state)       # typed view: x.dry_weight, x.indoor_temp, ...
```

Dynamics `dx/dt = f(x, u, v; c)` are the crop-growth, CO₂, energy, and water-vapour balances of
van Henten (1994, 2003), integrated with fixed-step RK4 over one control interval and clamped to
the state box. `c` is the 23-entry coefficient vector, passed to the integrator at runtime rather
than compiled in. That enables domain randomisation, parameter-sensitivity sweeps, and
sim-to-real studies where the controller's model and the plant disagree, all without rebuilding
the integrator.

## The benchmark scenario

`EnvConfig()` and `gym.make("LettuceGreenhouse-v0")` with no arguments give:

| setting | benchmark default |
|---|---|
| season | 40 days from 9 February 2014 (Bleiswijk trace, start day 40), 30-minute control step → 1920 steps |
| action space | `Box(-1, 1, (3,))`, **delta** mode: `u_next = clip(u_prev + a · du_max, lo, hi)` |
| observation | 12 floats: observables `y` (4), previous control (3), season progress (1), current weather (4) |
| reward | per-step profit `price · Δdry_weight − energy_cost − co2_cost` minus hinge penalties outside the comfort bands (CO₂ 500–1600 ppm, temperature 10–20 °C, RH ≤ 80 %) |
| episode end | `terminated=True` at harvest (the season is a terminal state, not a time limit); `truncated` is never set and no `TimeLimit` wrapper is registered |
| `info` | reward breakdown (`revenue`, `energy_cost`, `co2_cost`, `penalty_*`, …), `params`/`param_names` (the coefficient vector in effect), `start_day` |

## Configuring experiments

Everything that is not physics lives in `EnvConfig`, plain data you can serialise or sweep:

```python
from lettuce_greenhouse_gym import (
    ActionMode, ControlOverride, EnvConfig, InitialControl, InitialState, benchmark_reward_config,
)

config = EnvConfig(
    dt=900.0,                                  # 15-minute steps; n_steps follows automatically
    episode_days=60.0,
    action_mode=ActionMode.ABSOLUTE,           # action *is* the control level, rescaled to [lo, hi]
    include_previous_control=False,
    weather_window=13,                         # now + 12 steps of forecast in the observation
    reward=benchmark_reward_config(energy_cost=0.25),           # an energy-price scenario
    control_overrides=(ControlOverride("heating", hi=200.0, du_max=20.0),),
    initial_state=(InitialState("dry_weight", 0.005),),
    initial_control=(InitialControl("heating", 80.0),),
)
env = gym.make("LettuceGreenhouse-v0", config=config)
```

Comfort bands and penalty weights are `PenaltyBound`s inside `RewardConfig`; a band may carry a
second, daytime range switched on by outdoor radiation (`day_lo`, `day_hi`, `day_radiation`), which
the reward and both MPCs honour; the economic prices
are a `ParameterSet` with validated overrides (an unknown name or out-of-range price raises).

### Swap points

The environment is assembled by constructor injection. Replace one collaborator and keep the rest:

| argument | default | replace it to … |
|---|---|---|
| `model: DynamicsModel` | `VanHentenLettuce()` | run a different, hybrid or learned dynamics model behind the same env (see *Swapping the dynamics model*) |
| `reward: Reward` | `EconomicReward(config.reward)` | change the objective; any callable `(RewardContext) -> (float, dict)` works |
| `parameter_provider` | `FixedParameterProvider(nominal)` | randomise coefficients per episode (`RandomizedParameterProvider(base, {"leak": (1e-5, 2e-5)})`) |
| `weather_sampler` | `FixedWeatherSampler(BENCHMARK_SCENARIO)` | draw start days per episode (`RandomWeatherSampler`, `CyclingWeatherSampler`) |
| `weather_perturbation` | `None` | shape what the plant experiences or what controllers are told (see *Weather data*) |
| `weather_repository` | `default_repository()` | register your own traces by name; any callable returning a `WeatherSeries` (see *Weather data*) |

Two action conventions are configured rather than injected: `action_mode` is `delta` (the
benchmark's rate-limited change) or `absolute` (a level), and `absolute_rate_limit=True` clips a
level to within `du_max` of the previous control, for formulations that constrain the rate of a
level-valued input.

```python
from lettuce_greenhouse_gym import (
    LettuceGreenhouseEnv, MODEL_COEFFS, ParameterSet, RandomizedParameterProvider,
    RandomWeatherSampler, BLEISWIJK_2014,
)

env = LettuceGreenhouseEnv(
    EnvConfig(),
    parameter_provider=RandomizedParameterProvider(
        ParameterSet.from_defaults(MODEL_COEFFS), {"leak": (0.5e-5, 1.0e-5)}
    ),
    weather_sampler=RandomWeatherSampler(BLEISWIJK_2014, start_days=[20.0, 40.0, 60.0, 80.0]),
)
env.reset(seed=0)                               # one seed drives both weather and parameter draws
env.reset(seed=0, options={"start_day": 100.0}) # or pin the weather for evaluation
```

`RewardContext` carries the full transition (`x_prev, x_next, u, v, y, c, t, dt`), so a custom
reward can depend on weather or on the drawn parameters without changes to the library.

Parameters can change at two points, and the env owns only those points: a provider's `sample(rng)`
is called at `reset`, its `step(rng, step_index, current)` before every transition, and whatever set
comes back drives the next transition and appears in `info["params"]`. Subclass `ParameterProvider`
for drift, faults, or anything else; two reference schemes ship: absolute per-name ranges
(`RandomizedParameterProvider(base, {"leak": (1e-5, 2e-5)})`) and relative scaling of every
coefficient by `1 + U(−h, h)` (`RandomizedParameterProvider.relative(base, 0.05)`), each either once
per episode or, with `per_step=True`, redrawn every step. In a spec the latter is
`parameters: {relative: 0.05, per_step: true}`; any other scheme is your own provider named by
`parameters: {provider: {target: my_study:PoolProvider, kwargs: {...}}}`, called with the nominal
`ParameterSet` and your keyword arguments, so it reaches the CLI and training like the built-in ones.
Whether a *policy* may see the drawn coefficients is a research question, so the env keeps them in
`info["params"]`; `ParameterObservation(env, names, ranges)` is the wrapper that appends them to the
observation, rescaled by a range of your choosing.

### For model-based control

The env exposes exactly what an MPC or planner needs, and guarantees it is the same function the
env itself steps:

```python
e = env.unwrapped
F = e.integrator                       # F(x, u, v, c) -> x_next; a casadi.Function for the default model
g = e.measurement                      # g(x) -> y
x, u_prev, c = e.state, e.control, e.parameters.to_array()
v_horizon = e.weather_forecast(48)     # (4, 48): weather from now, one column per control step
```

Building an MPC amounts to unrolling `F` symbolically over `v_horizon`. Parameter randomisation
stays visible through `info["params"]`, so a sim-to-real or domain-randomisation study can read the
"true" coefficients per episode.

An episode can also be branched. `snapshot()` returns an `EnvState`, plain data with the state,
previous control, step and coefficients; `restore(state)` puts the episode back there, weather and
RNG untouched, and returns the observation. A controller can therefore roll a policy out from the
current moment under several parameter draws and come back, or place the plant at a chosen state
and time to generate data. `observe(x, u_prev, step_index)` builds the observation the env would
emit at a state it is only considering, using the same function `step` uses, so a policy can be
evaluated along a candidate trajectory without moving the plant.

### Swapping the dynamics model

The model contract has two layers, so that a learned model is not asked for an ODE it does not have:

- **`DynamicsModel`** is what the environment needs: the variable groups, a `ParameterSet` of
  coefficients, `build_integrator(dt)` returning a callable `F(x, u, v, c) -> x_next`, and
  `build_measurement()` returning `g(x) -> y`. Any Python callable with those shapes works, so a
  numeric or learned model (a fitted network, a table, a hybrid of the two) subclasses this
  directly. The env checks the output widths once at construction.
- **`SymbolicDynamicsModel`** adds `rhs` and `measurement` as CasADi expressions and compiles them
  into `casadi.Function`s (RK4, then a clamp to the state bounds). `VanHentenLettuce` is one. This
  is the layer a gradient-based planner needs: `NominalMPC` differentiates through `F`, and it
  refuses a numeric model with a message saying so. A learned model that should also serve an
  optimiser is written on this layer, with its fitted weights as CasADi operations.

Both layers predict the environment's own four states, since the reward and the comfort bands index
them by name. A latent world model therefore uses the env as a data source and an evaluator rather
than running behind this seam.

### Weather

- `bleiswijk_2014()`: the packaged trace (see *Data* below), 5-minute samples, 10 Jan – 29 Nov 2014.
- `WeatherSeries.from_csv(path, epoch_day)`: any trace in the same headerless `time, Io, To, RH, Vo, CO2ppm` format; RH and ppm are converted to densities with the package's own `units`. `weather.write_csv` is the inverse.
- `WeatherSeries.from_channels(name, dt, epoch_day, rad=..., out_co2=..., out_temp=..., out_vapor=...)`: from arrays; the last step of any custom loader (see *Weather data*).
- `WeatherSeries.synthetic(...)`: a smooth diurnal/seasonal signal for smoke tests.
- `split_start_days(series, n_steps, dt, rng)`: held-out train/evaluation start days from one trace.

`start_day` is a day of the year, not an offset into a file, so traces beginning on different dates
line up.

## Reference controllers

`lettuce_greenhouse_gym.baselines` ships reference controllers, a grower rule and two model
predictive controllers, that give every study a reproducible number to beat, and a `run_episode`
helper that logs a full season:

```python
from lettuce_greenhouse_gym import LettuceGreenhouseEnv
from lettuce_greenhouse_gym.baselines import GrowerHeuristic, GrowerRules, run_episode

log = run_episode(LettuceGreenhouseEnv(), GrowerHeuristic(), seed=0)
log.total_return, log.x.shape, log.u.shape        # (-4.35, (1921, 4), (1920, 3))
run_episode(env, GrowerHeuristic(GrowerRules(temp_night=12.0)), seed=0)   # your own setpoints
```

Controllers return physical controls; `env.encode_control(u)` turns them into actions, so the same
controller runs in both action modes. Season returns on the benchmark scenario (these are pinned by
the test suite, so a change means the benchmark changed):

| controller | reads | return |
|---|---|---|
| `AllOff` | nothing | −64.53 |
| `ConstantControl([0, 0, 50])`, hold the initial heating | nothing | −73.57 |
| `GrowerHeuristic` | observables + current outdoor weather | −4.35 |
| `NominalMPC` (horizon 12 steps) | true state, weather forecast, model (privileged) | +3.62 |
| `ScenarioMPC` | as `NominalMPC`, plus a belief about the coefficients (privileged) | see below |

`GrowerHeuristic` is a grower's climate computer: proportional heating below a setpoint,
ventilation above it or above a humidity limit, CO₂ dosing in daylight while the vents are nearly
shut. Its default setpoints (14 °C day, 10 °C night, CO₂ to 750 ppm) are the regime the model's
validation crops were grown under (van Henten, 1994). It reads no privileged information, so it is a
fair comparison for an observation-only RL policy.

`NominalMPC` is a receding-horizon controller built entirely on the public control API: it unrolls
`env.integrator` over `env.weather_forecast(H)` with CasADi and IPOPT (bundled with CasADi, no extra
dependency), maximising the env's own economic reward with the comfort bands as slack variables. It
plans with the nominal coefficients by default, so under parameter randomisation it is a
plant-model-mismatch baseline; `MPCSettings(use_true_parameters=True)` makes it an oracle, and
`MPCSettings(parameters=...)` plans with any other coefficient set, a mis-specified or identified
model. `MPCSettings(bounds=...)` lets the planner use comfort bands that differ from the reward it is
scored on, for instance a tightened humidity band. A season takes about 100 s (≈50 ms per solve at
the default horizon); `MPCSettings(horizon=..., ipopt_options={...})` exposes the horizon and the
solver. Every solve is recorded in `mpc.solves` as a `SolveStats` (success, iterations, wall time,
IPOPT status), and `run_episode` stores the current one under `info["controller"]`, so compute cost
is measured alongside return.

`ScenarioMPC` hedges instead of committing: before every solve it draws several coefficient
sequences from a `ParameterProvider`, its belief about the plant, runs one copy of the dynamics per
scenario and optimises a single control sequence against their average reward. With one scenario
from a fixed provider it reproduces `NominalMPC` exactly, which the tests pin. The provider is yours:
the env's own gives a well-specified controller, another one a mis-specified one, and a per-step
scheme is redrawn along the horizon as it would be on the plant. Its RNG is its own, seeded in
`ScenarioMPCSettings`, so what the controller imagines is never correlated with what happens. Cost
grows with the number of scenarios, roughly linearly in solve time.

`EpisodeLog.to_records(dt)` flattens a season into one row per step, states, observables, controls,
weather, reward, every scalar in `info` and a controller's solve statistics, and `write_csv(path)`
writes it with a header, so analysis scripts read one file instead of stacking arrays.

## Training and evaluating RL agents

Install the optional extras (`pip install "lettuce-greenhouse-gym[train]"`, or `[all]` to include
plotting). The training code ships inside the package and imports Stable-Baselines3 lazily, so a
plain install stays light and a missing extra produces a one-line install hint.

### The experiment spec

Everything about a run is one plain-data object, `ExperimentSpec`: the `EnvConfig`, the weather
choice, per-episode parameter ranges, and the training settings. Its defaults are the benchmark, it
round-trips through YAML, and every run directory gets a `spec.yaml` as its record.

```python
from lettuce_greenhouse_gym.experiment import (
    ExperimentSpec, ParameterSpec, SplitSpec, TrainSpec, WeatherSpec, build_env,
)
from lettuce_greenhouse_gym.train.sb3 import train, load, evaluate, return_components

spec = ExperimentSpec(
    weather=WeatherSpec(start_day=None, split=SplitSpec(train_fraction=0.8, seed=0)),  # held-out seasons
    parameters=ParameterSpec(ranges={"leak": (0.5e-5, 1.0e-5)}),                       # domain randomisation
    train=TrainSpec(algo="ppo", total_timesteps=1_000_000, n_envs=8, net_arch=(128, 128)),  # or sac, ddpg, td3
)
result = train(spec)                              # runs/<name>/{spec.yaml,model.zip,vecnormalize.pkl,monitor/}
spec, model, vecnormalize = load(result.run_dir)
log = evaluate(model, spec, vecnormalize=vecnormalize, seed=0)   # an EpisodeLog, like the baselines
print(return_components(log))
```

`TrainSpec` also takes `eval_every` (evaluate on the *eval* weather, the held-out days under a
split, and keep the best model in `<run>/best/`), `checkpoint_every` (`<run>/checkpoints/`, with the
normalisation statistics), and `train_seeds(spec, [0, 1, 2])` runs one seed after another as
`<run_name>_seed<k>`; the CLI exposes them as `--eval-every`, `--checkpoint-every`, `--seeds`.
`build_env(spec, "train" | "eval")` gives the bare env for your own loop. The full YAML schema is in
the `lettuce_greenhouse_gym.experiment` docstring; `examples/benchmark.yaml` is the canonical dump of
the defaults.

### The command line

```bash
lettuce-gym baselines                                  # reference-controller table (add mpc: ~100 s)
lettuce-gym train --algo ppo --timesteps 1000000 --n-envs 8 --run-name ppo_benchmark \
    --set 'train.algo_kwargs={n_steps: 4096, batch_size: 1024, n_epochs: 10, gamma: 0.95}'
lettuce-gym eval runs/ppo_benchmark --plot season.png
lettuce-gym train --config my_run.yaml --set env.episode_days=60 --print-config   # show the merged spec
```

Precedence is defaults < `--config` YAML < repeatable `--set key.path=value` < named flags.
`--set` values are parsed as YAML, so lists and mappings work inline.

### What a run produces

`spec.yaml`, the experiment record, and `run.json`, its provenance: package and dependency versions,
interpreter, platform, command line, working directory, the git commit of the working directory if it
is a checkout, and start and finish times. `monitor/*.monitor.csv` (per-worker episode returns), a
TensorBoard directory when TensorBoard is installed, and `episode/revenue`, `episode/energy_cost`,
`episode/co2_cost`, `episode/penalty`: season totals of the reward's parts, averaged over the
episodes of each rollout, next to SB3's `rollout/*` and `train/*`. Periodic evaluation logs the same
components as `eval/*` beside `eval/mean_reward`. With `--wandb PROJECT` (or
`TrainSpec.wandb=WandbSpec(project=...)`) the same rows go to Weights & Biases together with the spec
as the run config; offline runs upload with a plain `wandb sync`.

Reference results on the benchmark season, evaluated deterministically. PPO uses the literature's
PPO hyperparameters (`n_steps` 4096, batch 1024, 10 epochs, γ 0.95, [128, 128]) with 8 workers and 1M
steps, about 6 minutes on a laptop CPU; SAC is `examples/benchmark_sac.yaml`, about 20 minutes:

| controller | information | return |
|---|---|---|
| `AllOff` | none | −64.53 |
| `GrowerHeuristic` | observation + current weather | −4.35 |
| PPO, 1M steps, 8 workers | observation only | +2.36 |
| SAC, published-baseline configuration (13-step forecast, 192k steps) | observation only | +2.89 |
| SAC, same configuration, trained and evaluated under ±5 % per-step parameter noise (10 seeds) | observation + 13-step forecast | 2.35 ± 0.03 |
| `NominalMPC`, horizon 12, evaluated under the same noise (3 seeds) | true state, forecast, nominal model | 2.13 ± 0.02 |
| `NominalMPC`, horizon 12 | true state, perfect forecast, model | +3.62 |

Single seeds unless a seed count is given; the paper reports its RL results over its own seeds.
Treat these as references for a working pipeline, not as results.

### The published RL baseline

The RL agents reported for this benchmark (van Laatum et al., 2026) observe exactly what this
package's default layout provides: the four observables, the previously applied control, the season
clock, and the weather. Their forecast variant sees the current step plus twelve more; that is
`weather_window=13`, 60 entries. Their clock is the raw step index where ours defaults to season
progress in [0, 1]; the two are equivalent once observations are normalised, and
`EnvConfig(timestep_encoding=TimestepEncoding.INDEX)` gives the raw index for a bit-for-bit match.

The reported agents are SAC on the 13-step forecast observation, one environment, 192k steps, with
normalised observations. `examples/benchmark_sac.yaml` is that configuration:

```bash
lettuce-gym train --config examples/benchmark_sac.yaml --run-name sac_benchmark
```

Three of its settings are objects rather than numbers; the spec spells them as plain data:
`learning_rate: lin_5e-3` (a linear schedule to zero), `policy_kwargs: {activation_fn: relu}`, and
`action_noise_sigma: 0.05` (exploration noise on the action, for the off-policy algorithms; the
default is Gaussian, `action_noise: ornstein_uhlenbeck` selects the alternative). Expect
seed-to-seed variation at this budget; the paper reports its own numbers over its own seeds.

The paper's stochastic experiments perturb every model coefficient by a uniform relative factor,
redrawn at each step; a perturbation level δ there corresponds to
`parameters: {relative: δ/2, per_step: true}` here.

### Custom policies and algorithms

`TrainSpec.policy` takes any SB3 policy name, and `algo_kwargs` is passed to the constructor
unchanged, so a custom feature extractor goes through `algo_kwargs={"policy_kwargs": {...}}`;
the `net_arch` shorthand merges into it rather than replacing it. For an algorithm SB3 does not
ship, build the vectorised env with `make_vec_env(spec)` and hand it to your own trainer; anything
that implements `predict(obs, deterministic)` works with `PolicyController` and `evaluate`.

## Plotting

`pip install "lettuce-greenhouse-gym[plot]"` adds `plot_episode`, a 3×3 season figure: observables
against their comfort bands, controls against their actuator limits, weather, and the cumulative
reward components. It takes the `EpisodeLog` that `run_episode` and `evaluate` return, so reference
controllers and trained policies plot identically.

```python
from lettuce_greenhouse_gym.plot import plot_episode
fig = plot_episode(log, spec.env, title="ppo_benchmark")
fig.savefig("season.png", dpi=120)
```

The CLI exposes it as `lettuce-gym eval RUN --plot FILE` and `lettuce-gym baselines --plot DIR`.

## Weather data

The package ships one trace: outdoor weather measured at the WUR Glas facility in Bleiswijk in 2014
(see *Data* below), on which the benchmark season is defined. Anything else plugs in through one
protocol.

### Bringing your own weather

A weather source is a callable that returns a `WeatherSeries`: four channels on a uniform grid, in
model units, plus the grid step `dt` and `epoch_day`, the day of the year of the first sample.
Whatever you start from, a text export from your own station, an EnergyPlus EPW file, a NetCDF
reanalysis, a database, a generator, the loader is one function that parses, converts units,
resamples onto a grid, and ends in `WeatherSeries.from_channels`, which does the humidity and CO₂
conversions for you. It reaches the environment in one of three ways:

- **In Python**: `WeatherRepository({"site": partial(load_epw, "site.epw")})` passed as the env's
  `weather_repository`, with a sampler naming `"site"`.
- **In a spec**, so the CLI, training with several worker processes, and a saved run can all
  rebuild it:

  ```yaml
  weather:
    source: site
    start_day: 40.0
    loaders:
      site: {target: examples/weather_epw.py:load_epw, kwargs: {path: site.epw}}
  ```

  `target` is `package.module:function` or `path/to/script.py:function`; relative paths are taken
  from the YAML's directory. A spec with loaders runs the code it names, so treat a spec from
  someone else as you would treat their script.
- **As a file, once**: `lettuce_greenhouse_gym.weather.write_csv` writes the package's CSV format
  (`time, Io, To, RH, Vo, CO2ppm`), which a spec then names under `weather.files` with its
  `epoch_day`, no code needed.

[`docs/custom_weather.md`](docs/custom_weather.md) walks through all of this on a text file and
covers the choices and pitfalls for any format. `examples/weather_txt.py` and
`examples/weather_epw.py` are complete loaders you can run as they are.

### More years from KNMI

For generalisation studies, and for the benchmark literature's training protocol (train on several
years, evaluate on Bleiswijk), fetch years from KNMI, the Dutch meteorological institute, whose
hourly station observations are open data (CC BY 4.0, attribution to KNMI):

```bash
lettuce-gym weather fetch-knmi --station 344 --years 2001-2010 --out weather/   # 344 = Rotterdam Airport
```

Each year becomes `weather/knmi_<station>_<year>.csv` in the package's CSV format (5-minute rows,
linearly interpolated from the hourly observations; outdoor CO₂ is a constant 400 ppm since KNMI does
not measure it, `--co2` changes it) plus a `SOURCE.md` with the attribution and the conversions
applied. Files start at 00:00 on 1 January, so their `epoch_day` is 1.0.

### Several traces, and training weather that differs from evaluation weather

A spec can train on several sources, drawing one per episode, while evaluating on another trace:

```yaml
weather:
  source: [knmi_344_2009, knmi_344_2010]     # one is drawn per episode
  start_day: 40.0                            # the same season start in every year
  files:
    knmi_344_2009: {path: weather/knmi_344_2009.csv, epoch_day: 1.0}   # relative to this YAML
    knmi_344_2010: {path: weather/knmi_344_2010.csv, epoch_day: 1.0}
  eval_source: bleiswijk_2014                # evaluation stays on the benchmark trace
```

Sources may be packaged names, `files` keys or `loaders` keys, mixed freely. `eval_start_days`
overrides the evaluation days; `start_days` gives training a list of season starts; `split` holds
out start days of a single trace. When seasons should pair particular traces with particular days,
or follow any other rule, `weather.sampler` (and `eval_sampler`) names your own `WeatherSampler` in
the same `target` and `kwargs` form as a loader, replacing the start-day modes for that role.
`lettuce-gym weather list` prints the sources a spec can name. Run directories record absolute
paths, so a saved `spec.yaml` reproduces on the machine that wrote it.

### Perturbing weather

A trace is what was measured. Two things a study may want to change about it are hook points on the
env, both identity by default: what the **plant experiences** and what a **controller is told**.
Subclass `WeatherPerturbation` and override `realise(rng, weather, dt)`, called once per episode
with the trace from the season start to its end at the control step, or `forecast(rng, step_index,
weather, dt)`, called on every forecast the env hands out, the observation's weather window and
`weather_forecast(n)` alike, with the realised weather from now on. A stochastic realisation around
the measured trace goes in the first; forecast error growing with lead time goes in the second, and
leaves the plant untouched. Both receive the env's RNG, so a seed still fixes the episode, and both
must return the same shape within the physical bounds, or the env refuses. In a spec:
`weather: {perturbation: {target: my_study:ForecastNoise, kwargs: {sigma: 0.1}}}`.

## Where this departs from van Henten's papers

Defaults follow the RL/MPC benchmark that grew around the model, because that is what makes results
comparable. Where the benchmark and the original papers differ, the code says so and lets you
choose:

- **Leakage** `leak` defaults to `0.75e-5 m/s` (benchmark); van Henten (1994, 2003) give `0.75e-4`.
  Override it with `ParameterSet.override(leak=0.75e-4)` or a provider.
- **Prices** are in euro per kWh / kg (benchmark); van Henten's are in guilders. Two of the four
  economic coefficients therefore differ from the 2003 paper (`model/parameters/economic.py`).
- **Comfort bands** and the soft-penalty reward are the benchmark's; van Henten's problem has hard
  constraints (6.5–20 °C, RH ≤ 70 %). Set your own `PenaltyBound`s or supply another `Reward`.
- **Season** is the benchmark's 40 days from 9 February 2014 (day of year 40) at 30-minute steps,
  as stated in van Laatum et al. (2026). The reference implementation's file-offset arithmetic
  actually starts one day earlier, on 8 February; `env.reset(options={"start_day": 39.0})`
  reproduces that window if you need to match published numbers exactly.
- **Relative humidity** in `y` uses the Magnus formula; van Henten (2003) eqn 12 uses a different
  saturation curve (coefficient `c_v,4`). They agree to within ~1 %.

## Data

The packaged weather trace `outdoorWeatherWurGlas2014.csv` was measured at the WUR Glas
reference greenhouse facility in Bleiswijk, the Netherlands, in 2014, and is shipped unmodified.
Please cite the reference below in any work that uses it:

> Kempkes, F. L. K., Janse, J., & Hemming, S. (2014). Greenhouse concept with high insulating
> double glass with coatings and new climate control strategies; from design to results from tomato
> experiments. *Acta Horticulturae*, (1037), 83–92. <https://doi.org/10.17660/ActaHortic.2014.1037.6>

Format details and the epoch convention are in `src/lettuce_greenhouse_gym/data/SOURCE.md`.

## References

### Van Henten's works — the model

- van Henten, E. J. (1994). Validation of a dynamic lettuce growth model for greenhouse climate
  control. *Agricultural Systems*, 45(1), 55–72. <https://doi.org/10.1016/S0308-521X(94)90280-1>
- van Henten, E. J. (1994). *Greenhouse climate management: an optimal control approach.* PhD thesis,
  Wageningen Agricultural University. <https://doi.org/10.18174/205106>
- van Henten, E. J. (2003). Sensitivity analysis of an optimal control problem in greenhouse climate
  management. *Biosystems Engineering*, 85(3), 355–364. <https://doi.org/10.1016/S1537-5110(03)00068-0>

Equation numbers quoted in the source refer to the 2003 paper.

### The benchmark — reward, comfort bands, season, and prices

- van Laatum, B., Msaad, S., van Henten, E. J., McAllister, R. D., & Boersma, S. (2026). Stochastic
  model predictive control with reinforcement learning for greenhouse production systems under
  parametric uncertainty. *Control Engineering Practice*, 169, 106787.
  <https://doi.org/10.1016/j.conengprac.2026.106787>
- Mallick, S., Airaldi, F., Dabiri, A., Sun, C., & De Schutter, B. (2025). Reinforcement
  learning-based model predictive control for greenhouse climate control. *Smart Agricultural
  Technology*, 10, 100751. <https://doi.org/10.1016/j.atech.2024.100751>
- Morcego, B., Yin, W., Boersma, S., van Henten, E., Puig, V., & Sun, C. (2023). Reinforcement
  learning versus model predictive control on greenhouse climate control. *Computers and Electronics
  in Agriculture*, 215, 108372. <https://doi.org/10.1016/j.compag.2023.108372>

## Citing this package

A machine-readable `CITATION.cff` is in the repository root (GitHub shows a *Cite this repository*
button from it). Please also cite the van Henten papers above, which define the model.

```bibtex
@software{soykan2026lettucegreenhousegym,
  author  = {Soykan, G{\"u}rkan},
  title   = {lettuce-greenhouse-gym: a Gymnasium environment for the van Henten lettuce greenhouse model},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/gsoykan/lettuce-greenhouse-gym},
}
```

## Development

The reasoning behind the main structural choices is in [`docs/design.md`](docs/design.md).

```bash
uv sync --all-extras             # extras are needed for the train/plot tests; they skip without them
uv run pytest -q -m "not slow"   # incl. numerical parity gates for f, g, and the reward (~15 s)
uv run pytest -q                 # everything, incl. the pinned 40-day MPC season (~2 min)
uv run ruff check . && uv run ruff format --check .
uv run mypy                      # the package is typed; CI checks it
uv run pre-commit run -a
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). The weather trace carries its own attribution
requirement (see *Data*).
