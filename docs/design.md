# Design notes

Short arguments for choices a reader of the source might otherwise wonder about. The code says
what it does; this says why.

## One description of the observation

`observation_space` is built once and the observation vector is rebuilt every step. Written
independently they drift: change `weather_window` and one says 12 while the other produces 8, and
nothing complains, because both are valid arrays. So the layout is described once, in
`_build_observation_layout`, and both readers concatenate it in the same order. Adding a block means
editing that one function.

## Delta actions by default

In delta mode an action is a change: zero holds the current setting, plus or minus one moves by one
full `du_max`, and the result is clipped to the actuator range. This is what real actuators do, a
boiler cannot jump from 0 to 150 W/m² in one step, and it is what the benchmark uses. Absolute mode
is simpler and convenient when comparing against an MPC that optimises levels directly;
`absolute_rate_limit` adds the rate constraint back for formulations that want both.

## Variable groups as the single source of truth

Names, order, units and physical bounds of states, controls, weather and observables live in the
variable groups. Everything else derives from them: Gymnasium spaces, index enums for the ODE,
typed views, the CSV column order. An environment that wrote its own bounds would create a second
copy that silently drifts from the definitions.

## Hooks with identity defaults

Where a study might intervene, the environment offers a point, not a policy: a `ParameterProvider`
is asked for coefficients at reset and before every step; a `WeatherPerturbation` may reshape what
the plant experiences and what a controller is told; a `Reward` sees the whole transition. Every
default is the identity or the nominal value, so `EnvConfig()` reproduces the benchmark and a study
adds exactly the mechanism it wants to test. Whether a controller may know the drawn coefficients
is itself a research question, so they travel in `info["params"]` and a wrapper decides.

## Two model layers

`DynamicsModel` is what the environment needs: a one-step integrator and a measurement map as plain
callables. `SymbolicDynamicsModel` adds the CasADi expressions a gradient-based planner needs and
compiles them. The split exists so a learned or numeric model is not asked for an ODE it does not
have, and so the MPC can say clearly which layer it requires. Both predict the environment's own
four physical states, since the reward and comfort bands index them by name; a latent world model
uses the environment as a data source and an evaluator instead.

## What the loader refuses

A `WeatherSeries` validates its channels against physical bounds on construction, `from_csv`
rejects a non-uniform time column, the repository rejects a loader that returns anything else, and
an episode that would run off the trace raises at reset. Each refusal replaces a plausible-looking
season on impossible weather with an error at the right place.
