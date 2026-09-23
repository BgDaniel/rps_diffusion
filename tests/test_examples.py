"""Tests for the winding diagnostics and the example scenario definitions."""

from __future__ import annotations

import numpy as np
import pytest

from rps_diffusion import Domain, RPSSimulator, winding, winding_number
from rps_diffusion.examples import GROUPS


@pytest.mark.parametrize("n", [1, 2, -1])
def test_winding_initial_condition_has_requested_winding(n: int) -> None:
    sim = RPSSimulator(3.0, sigma=0.0, Nx=64, domain=Domain.annulus(64))
    res = sim.run(winding(64, winding_number=n), t_max=0.0, progress=False)
    assert winding_number(res, 0, (0.5, 0.5), 0.37) == pytest.approx(n, abs=1e-6)


def test_winding_number_rejects_circle_outside_domain() -> None:
    sim = RPSSimulator(3.0, sigma=0.0, Nx=32, domain=Domain.annulus(32))
    res = sim.run(winding(32), t_max=0.0, progress=False)
    with pytest.raises(ValueError):
        winding_number(res, 0, (0.5, 0.5), 0.1)  # inside the hole


@pytest.mark.parametrize(
    ("group", "scenario"), [(g, s) for g in GROUPS for s in g.scenarios], ids=lambda x: getattr(x, "name", "")
)
def test_scenarios_are_well_formed(group, scenario) -> None:
    """Every scenario builds a valid domain and initial condition and passes the CFL check."""
    dom = scenario.domain(scenario.Nx, scenario.L) if scenario.domain else None
    rho0 = scenario.initial(scenario.Nx, scenario.L)
    assert rho0.shape == (3, scenario.Nx, scenario.Nx)
    np.testing.assert_allclose(rho0.sum(axis=0), 1.0)
    RPSSimulator(scenario.rates, sigma=scenario.sigma, L=scenario.L, dt=scenario.dt, Nx=scenario.Nx, domain=dom)


def test_scenario_names_are_unique() -> None:
    keys = [(g.key, s.name) for g in GROUPS for s in g.scenarios]
    assert len(keys) == len(set(keys))
