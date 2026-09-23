"""Tests for RPSSimulator, the reaction term and the frequency analysis."""

from __future__ import annotations

import numpy as np
import pytest

from rps_diffusion import (
    Domain,
    RPSSimulator,
    blobs,
    concentrated,
    dominant_frequencies,
    fixed_point,
    frequency_spectrum,
    omega0,
)
from rps_diffusion.domain import grid
from rps_diffusion.simulator import as_rates, reaction


def test_rates_parsing() -> None:
    assert as_rates(2.0) == (2.0, 2.0, 2.0)
    assert as_rates((1, 2, 3)) == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        as_rates((1, 2))
    with pytest.raises(ValueError):
        as_rates((1, -2, 3))


@pytest.mark.parametrize("rates", [2.0, (1.0, 2.0, 4.0), (0.5, 3.0, 1.5)])
def test_fixed_point_and_linear_frequency(rates) -> None:
    """ρ* is a zero of f, and the Jacobian there has eigenvalues ±iω₀."""
    rs = np.array(fixed_point(rates))
    assert rs.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(reaction(rs, rates), 0.0, atol=1e-14)
    h = 1e-7
    J = np.column_stack([(reaction(rs + h * e, rates) - reaction(rs - h * e, rates)) / (2 * h) for e in np.eye(3)])
    ev = np.linalg.eigvals(J)
    assert np.max(np.abs(ev.real)) < 1e-6
    assert np.max(np.abs(ev.imag)) == pytest.approx(omega0(rates), rel=1e-6)


def test_equal_rates_reduce_to_lambda_over_sqrt3() -> None:
    assert omega0(3.0) == pytest.approx(3.0 / np.sqrt(3))
    assert fixed_point(3.0) == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_reaction_conserves_total() -> None:
    rho = blobs(16)
    np.testing.assert_allclose(reaction(rho, (1.0, 2.5, 4.0)).sum(axis=0), 0.0, atol=1e-14)


def test_cfl_violation_raises() -> None:
    with pytest.raises(ValueError, match="CFL"):
        RPSSimulator(1.0, sigma=1.0, dt=0.01, Nx=64)
    RPSSimulator(1.0, sigma=0.05, dt=0.005, Nx=64)  # fine


def test_grid_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        RPSSimulator(1.0, sigma=0.1, Nx=32, domain=Domain.disk(64))
    with pytest.raises(ValueError):
        RPSSimulator(1.0, sigma=0.1, Nx=32, domain=np.ones((64, 64), dtype=bool))


def test_pure_diffusion_cosine_decay_through_simulator() -> None:
    """With λ = 0 a cosine perturbation decays at γ_mn = σ²π²(m²+n²)/(2L²)."""
    Nx, L, sigma, m, n = 48, 1.0, 0.1, 1, 1
    X, Y = grid(Nx, L)
    mode = np.cos(m * np.pi * X / L) * np.cos(n * np.pi * Y / L)
    a = 0.1
    rho0 = np.stack([1 / 3 + a * mode, 1 / 3 - a * mode, np.full_like(mode, 1 / 3)])
    res = RPSSimulator(0.0, sigma=sigma, L=L, dt=0.005, Nx=Nx).run(rho0, 2.0, 50, progress=False)
    gamma = sigma**2 * np.pi**2 * (m**2 + n**2) / (2 * L**2)
    for t, snap in zip(res.t, res.snapshots):
        np.testing.assert_allclose(snap[0], 1 / 3 + a * mode * np.exp(-gamma * t), atol=2e-4)


@pytest.mark.parametrize("domain", [None, Domain.disk(40), Domain.annulus(40), Domain.l_shape(40)])
def test_mass_conserved_without_reaction(domain: Domain | None) -> None:
    rho0 = blobs(40)
    sim = RPSSimulator(0.0, sigma=0.05, dt=0.01, Nx=40, domain=domain)
    res = sim.run(rho0, 2.0, save_every=50, progress=False)
    np.testing.assert_allclose(res.fractions - res.fractions[0], 0.0, atol=1e-10)
    assert np.isnan(res.snapshots[-1][:, ~sim.mask]).all()
    np.testing.assert_allclose(np.nansum(res.snapshots[-1], axis=0)[sim.mask], 1.0, atol=1e-6)


def test_variance_decays_under_diffusion() -> None:
    sim = RPSSimulator(0.0, sigma=0.1, dt=0.005, domain=Domain.disk(32))
    res = sim.run(blobs(32), 3.0, save_every=100, progress=False)
    assert np.all(np.diff(res.variances, axis=0) <= 1e-12)


@pytest.mark.parametrize(
    ("rates", "domain"),
    [(3.0, None), (3.0, Domain.disk(48)), ((1.0, 2.0, 4.0), None), ((1.0, 2.0, 4.0), Domain.annulus(48))],
)
def test_mean_fraction_oscillates_at_omega0(rates, domain: Domain | None) -> None:
    Nx = 48
    rs = np.array(fixed_point(rates))
    rho0 = concentrated(Nx, radius=0.3, u0=tuple(rs + [0.03, -0.015, -0.015]), background=tuple(rs))
    sim = RPSSimulator(rates, sigma=0.05, dt=0.01, Nx=Nx, domain=domain)
    res = sim.run(rho0, t_max=60.0, save_every=5, progress=False)
    f0 = omega0(rates) / (2 * np.pi)
    assert dominant_frequencies(res, n=1)[0] == pytest.approx(f0, rel=0.02)
    freqs, power = frequency_spectrum(res, species=1)
    assert freqs.shape == power.shape
    assert freqs[np.argmax(power[1:]) + 1] == pytest.approx(f0, abs=2 * (freqs[1] - freqs[0]))
    # The mean fractions oscillate around the fixed point ρ*.
    np.testing.assert_allclose(res.fractions.mean(axis=0), rs, atol=5e-3)


@pytest.mark.filterwarnings("ignore:.*inflate the reaction cycles")
def test_heun_preserves_neutral_cycle_better_than_euler() -> None:
    """For equal rates the RPS ODE conserves ρ_S ρ_R ρ_P; Heun drifts far less than Euler."""
    Nx, lam = 8, 3.0
    rho0 = concentrated(Nx, radius=1.0, u0=(0.5, 0.3, 0.2))
    invariant0 = rho0.prod(axis=0).mean()
    drift = {}
    for method in ("euler", "heun"):
        sim = RPSSimulator(lam, sigma=0.0, dt=0.02, Nx=Nx, method=method)
        res = sim.run(rho0, 20.0, save_every=1000, progress=False)
        drift[method] = abs(res.snapshots[-1].prod(axis=0).mean() - invariant0)
    assert drift["heun"] < 0.05 * drift["euler"]


@pytest.mark.parametrize("style", ["surface", "rgb"])
def test_make_video_gif_fallback(tmp_path, monkeypatch, style: str) -> None:
    import rps_diffusion.visualize as vis

    monkeypatch.setattr(vis, "_ffmpeg_path", lambda: None)
    sim = RPSSimulator((1.0, 2.0, 3.0), sigma=0.05, dt=0.01, domain=Domain.disk(24))
    res = sim.run(blobs(24), 1.0, save_every=10, progress=False)
    out = vis.make_video(res, tmp_path / "v.mp4", max_frames=5, style=style)
    assert out.suffix == ".gif" and out.stat().st_size > 0
