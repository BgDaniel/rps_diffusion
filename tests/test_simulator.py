"""Tests for RPSSimulator and the frequency analysis."""

from __future__ import annotations

import numpy as np
import pytest

from rps_diffusion import (
    Domain,
    LambdaField,
    RPSSimulator,
    blobs,
    concentrated,
    dominant_frequencies,
    frequency_spectrum,
)
from rps_diffusion.domain import grid


def test_cfl_violation_raises() -> None:
    lam = np.ones((64, 64))
    with pytest.raises(ValueError, match="CFL"):
        RPSSimulator(lam, sigma=1.0, dt=0.01)
    RPSSimulator(lam, sigma=0.05, dt=0.005)  # fine


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        RPSSimulator(np.ones((32, 32)), sigma=0.1, Nx=64)
    with pytest.raises(ValueError):
        RPSSimulator(np.ones((32, 32)), sigma=0.1, domain=Domain.disk(64))


def test_pure_diffusion_cosine_decay_through_simulator() -> None:
    """With λ = 0 a cosine perturbation decays at γ_mn = σ²π²(m²+n²)/(2L²)."""
    Nx, L, sigma, m, n = 48, 1.0, 0.1, 1, 1
    X, Y = grid(Nx, L)
    mode = np.cos(m * np.pi * X / L) * np.cos(n * np.pi * Y / L)
    a = 0.1
    rho0 = np.stack([1 / 3 + a * mode, 1 / 3 - a * mode, np.full_like(mode, 1 / 3)])
    res = RPSSimulator(np.zeros((Nx, Nx)), sigma=sigma, L=L, dt=0.005).run(rho0, 2.0, 50, progress=False)
    gamma = sigma**2 * np.pi**2 * (m**2 + n**2) / (2 * L**2)
    for t, snap in zip(res.t, res.snapshots):
        np.testing.assert_allclose(snap[0], 1 / 3 + a * mode * np.exp(-gamma * t), atol=2e-4)


@pytest.mark.parametrize("domain", [None, Domain.disk(40), Domain.annulus(40), Domain.l_shape(40)])
def test_mass_conserved_without_reaction(domain: Domain | None) -> None:
    rho0 = blobs(40)
    sim = RPSSimulator(np.zeros((40, 40)), sigma=0.05, dt=0.01, domain=domain)
    res = sim.run(rho0, 2.0, save_every=50, progress=False)
    np.testing.assert_allclose(res.fractions - res.fractions[0], 0.0, atol=1e-10)
    assert np.isnan(res.snapshots[-1][:, ~sim.mask]).all()
    np.testing.assert_allclose(np.nansum(res.snapshots[-1], axis=0)[sim.mask], 1.0, atol=1e-6)


def test_variance_decays_under_diffusion() -> None:
    sim = RPSSimulator(np.zeros((32, 32)), sigma=0.1, dt=0.005, domain=Domain.disk(32))
    res = sim.run(blobs(32), 3.0, save_every=100, progress=False)
    assert np.all(np.diff(res.variances, axis=0) <= 1e-12)


@pytest.mark.parametrize("domain", [None, Domain.disk(48)])
def test_mean_fraction_oscillates_at_omega0(domain: Domain | None) -> None:
    Nx, lam = 48, 3.0
    field = LambdaField(Nx).add_background(lam).build()
    sim = RPSSimulator(field, sigma=0.05, dt=0.005, domain=domain)
    res = sim.run(concentrated(Nx, radius=0.3, u0=(0.36, 0.32, 0.32)), t_max=60.0, save_every=10, progress=False)
    f0 = lam / np.sqrt(3) / (2 * np.pi)
    assert dominant_frequencies(res, n=1)[0] == pytest.approx(f0, rel=0.02)
    freqs, power = frequency_spectrum(res, species=1)
    assert freqs.shape == power.shape
    assert freqs[np.argmax(power[1:]) + 1] == pytest.approx(f0, abs=2 * (freqs[1] - freqs[0]))
