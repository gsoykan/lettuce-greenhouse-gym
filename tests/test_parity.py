"""Numeric parity gate: the model must reproduce the reference implementation.

``data/vanhenten_parity.npz`` was generated once from the reference van Henten implementation
(see ``tools/generate_parity_fixture.py``); these tests keep the package honest without ever
importing that project. Samples are stored row-major ``(n, dim)`` while CasADi consumes columns,
hence the transposes.
"""

import pathlib

import numpy as np
import pytest

from lettuce_greenhouse_gym.model.parameters import MODEL_COEFFS, ParameterSet
from lettuce_greenhouse_gym.model.vanhenten import VanHentenLettuce
from lettuce_greenhouse_gym.variables.observables import OBSERVABLE

# How the fixture was generated (one-off, by a dev script kept outside the repo):
#   source     reference van Henten implementation (CasADi RK4)
#   seed       numpy default_rng(20240724)
#   nominal    300 samples of (x, u, v) drawn uniformly in-bounds: x in [X_MIN, X_MAX],
#              u in [0, U_MAX], v in [rad 0..1000 W/m2, out_co2 4e-4..9e-4 kg/m3,
#              out_temp -10..35 degC, out_vapor 0..0.02 kg/m3]; ~10% of rows forced to
#              radiation == 0 to cover the night-time photosynthesis branch.
#              Stored with the reference x_next at dt = 900 s and 1800 s, plus y.
#   perturbed  100 further samples whose 23 coefficients are scaled by U(0.75, 1.25),
#              with the reference x_next at dt = 900 s; exercises the runtime-c path.
# Regenerate only against a reference implementation; `c_nominal` is asserted below to still
# match this package's own MODEL_COEFFS, so a drifted fixture fails loudly rather than silently.
#
# resolve relative to THIS FILE, not the cwd — pytest may be run from anywhere
FIXTURE = np.load(pathlib.Path(__file__).parent / "data" / "vanhenten_parity.npz")

# Absolute tolerance with rtol=0: states span very different magnitudes (dry weight ~1e-3,
# temperature ~30), so a relative tolerance would be brutal on one end and meaningless on the
# other. Observed error is ~7e-15; 1e-12 leaves headroom for platform float differences.
TOL = 1e-12


def test_c_nominal_matches_registry():
    """If a coefficient is ever edited, this fails first — with an obvious message."""
    expected = ParameterSet.from_defaults(MODEL_COEFFS).to_array()
    np.testing.assert_array_equal(FIXTURE["c_nominal"], expected)


@pytest.mark.parametrize("dt", [900, 1800], ids=["dt900", "dt1800"])
def test_integrator_matches_reference(dt):
    model = VanHentenLettuce()
    F = model.build_integrator(float(dt))
    c = model.constants.to_array()
    ours = np.array(F(x=FIXTURE["x"].T, u=FIXTURE["u"].T, v=FIXTURE["v"].T, c=c)["x_next"])
    np.testing.assert_allclose(ours, FIXTURE[f"x_next_dt{dt}"].T, atol=TOL, rtol=0)


def test_measurement_matches_reference():
    g = VanHentenLettuce().build_measurement()
    ours = np.array(g(x=FIXTURE["x"].T)["y"])
    np.testing.assert_allclose(ours, FIXTURE["y"].T, atol=TOL, rtol=0)


def test_integrator_matches_reference_with_perturbed_parameters():
    """Coefficients must flow in as an argument, not be baked into F."""
    model = VanHentenLettuce()
    F = model.build_integrator(900.0)
    ours = np.array(
        F(
            x=FIXTURE["x_perturbed"].T,
            u=FIXTURE["u_perturbed"].T,
            v=FIXTURE["v_perturbed"].T,
            c=FIXTURE["c_perturbed"].T,
        )["x_next"]
    )
    np.testing.assert_allclose(ours, FIXTURE["x_next_perturbed_dt900"].T, atol=TOL, rtol=0)


def test_measurement_outputs_within_observable_bounds():
    """g must never produce a value outside the declared observable bounds."""
    g = VanHentenLettuce().build_measurement()
    y = np.array(g(x=FIXTURE["x"].T)["y"])  # (4, 300)
    lo, hi = OBSERVABLE.bounds()
    assert np.all(y >= lo[:, None] - 1e-9)
    assert np.all(y <= hi[:, None] + 1e-9)
