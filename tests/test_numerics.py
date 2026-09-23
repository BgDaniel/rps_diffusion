"""Tests of the Neumann Laplacian against analytic solutions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j0

from rps_diffusion._numerics import laplacian_neumann, max_stable_dt, neighbour_masks
from rps_diffusion.domain import Domain, grid

J1_FIRST_ZERO = 3.8317059702075125  # first positive zero of J0' = -J1


def _evolve_heat(u: np.ndarray, D: float, dx: float, t: float, dt: float, faces=None) -> np.ndarray:
    for _ in range(int(round(t / dt))):
        u = u + dt * D * laplacian_neumann(u, dx, faces)
    return u


@pytest.mark.parametrize(("m", "n"), [(1, 0), (1, 2), (3, 1)])
def test_cosine_mode_decays_exponentially(m: int, n: int) -> None:
    """cos(mπx/L) cos(nπy/L) satisfies Neumann BC and decays like exp(-γ_mn t)."""
    Nx, L, sigma = 64, 1.0, 0.1
    D = sigma**2 / 2
    X, Y = grid(Nx, L)
    u0 = np.cos(m * np.pi * X / L) * np.cos(n * np.pi * Y / L)
    t, dt = 2.0, 0.2 * max_stable_dt(L / Nx, D)
    u = _evolve_heat(u0, D, L / Nx, t, dt)
    gamma = sigma**2 * np.pi**2 * (m**2 + n**2) / (2 * L**2)
    np.testing.assert_allclose(u, u0 * np.exp(-gamma * t), atol=2e-3 * np.exp(-gamma * t))


def test_cosine_is_discrete_eigenvector() -> None:
    """On the cell-centred grid the cosine modes are exact eigenvectors (DCT-II)."""
    Nx, L, m = 32, 2.0, 3
    dx = L / Nx
    X, _ = grid(Nx, L)
    u = np.cos(m * np.pi * X / L)
    eig = -4 / dx**2 * np.sin(m * np.pi / (2 * Nx)) ** 2
    np.testing.assert_allclose(laplacian_neumann(u, dx), eig * u, atol=1e-10)


def test_full_mask_matches_ghost_cells() -> None:
    rng = np.random.default_rng(0)
    u = rng.random((3, 20, 20))
    faces = neighbour_masks(np.ones((20, 20), dtype=bool))
    np.testing.assert_allclose(laplacian_neumann(u, 0.1, faces), laplacian_neumann(u, 0.1))


@pytest.mark.parametrize(
    "domain",
    [Domain.disk(48), Domain.annulus(48), Domain.l_shape(48), Domain(48).cut_disk(0.5, 0.5, 0.2)],
)
def test_masked_laplacian_conserves_mass_and_constants(domain: Domain) -> None:
    rng = np.random.default_rng(1)
    faces = neighbour_masks(domain.mask)
    u = rng.random((48, 48))
    lap = laplacian_neumann(u, domain.dx, faces)
    assert abs(lap[domain.mask].sum()) < 1e-8 * np.abs(lap).sum()
    assert np.all(lap[~domain.mask] == 0)
    np.testing.assert_allclose(laplacian_neumann(np.ones((48, 48)), domain.dx, faces), 0.0)


def test_disk_bessel_mode_decays_at_neumann_rate() -> None:
    """J0(j'₁ r / R) is the first radial Neumann eigenmode of the disk."""
    Nx, L = 160, 1.0
    R = L / 2
    dom = Domain.disk(Nx, L)
    X, Y = grid(Nx, L)
    r = np.hypot(X - L / 2, Y - L / 2)
    u0 = np.where(dom.mask, j0(J1_FIRST_ZERO * r / R), 0.0)
    D, t = 5e-3, 3.0
    dt = 0.2 * max_stable_dt(dom.dx, D)
    u = _evolve_heat(u0, D, dom.dx, t, dt, neighbour_masks(dom.mask))
    ratio = (u[dom.mask] @ u0[dom.mask]) / (u0[dom.mask] @ u0[dom.mask])
    expected = np.exp(-D * (J1_FIRST_ZERO / R) ** 2 * t)
    # Staircase boundary is first-order accurate; allow a few percent in the rate.
    assert abs(np.log(ratio) / np.log(expected) - 1) < 0.05


def test_max_stable_dt() -> None:
    assert max_stable_dt(0.1, 0.5, ndim=1) == pytest.approx(0.01)
    assert max_stable_dt(0.1, 0.5, ndim=2) == pytest.approx(0.005)
    assert max_stable_dt(0.1, 0.0) == float("inf")
