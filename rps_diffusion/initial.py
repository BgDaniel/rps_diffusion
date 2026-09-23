"""Initial-condition factories.

Every factory returns an array of shape ``(3, Nx, Nx)`` ordered as
``(ρ_S, ρ_R, ρ_P)`` with values in ``[0, 1]`` summing to 1 pointwise. The
array covers the whole bounding box; :class:`~rps_diffusion.RPSSimulator`
ignores cells outside the chosen :class:`~rps_diffusion.Domain`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .domain import grid

__all__ = ["homogeneous", "random_perturbation", "stripes", "blobs", "concentrated", "normalise"]

S, R, P = 0, 1, 2


def normalise(rho: np.ndarray) -> np.ndarray:
    """Clip to ``[0, 1]`` and rescale so the three species sum to 1 pointwise.

    Cells where all three densities vanish are reset to ``(1/3, 1/3, 1/3)``.

    Parameters
    ----------
    rho : np.ndarray
        Array of shape ``(3, ...)``.

    Returns
    -------
    np.ndarray
        Normalised copy of ``rho``.
    """
    out = np.clip(rho, 0.0, 1.0)
    total = out.sum(axis=0, keepdims=True)
    return np.divide(out, total, out=np.full_like(out, 1.0 / 3.0), where=total > 0.0)


def homogeneous(Nx: int, u0: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)) -> np.ndarray:
    """Spatially constant state.

    Parameters
    ----------
    Nx : int
        Grid cells per axis.
    u0 : tuple of float, default (1/3, 1/3, 1/3)
        Fractions ``(u_S, u_R, u_P)``; rescaled to sum to 1.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, Nx, Nx)``.
    """
    u = np.asarray(u0, dtype=float).reshape(3, 1, 1)
    return normalise(np.broadcast_to(u, (3, Nx, Nx)).copy())


def random_perturbation(
    Nx: int,
    noise: float = 0.05,
    seed: int = 0,
    background: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> np.ndarray:
    """Gaussian noise around a coexistence state.

    Parameters
    ----------
    Nx : int
        Grid cells per axis.
    noise : float, default 0.05
        Standard deviation of the independent per-cell, per-species noise.
    seed : int, default 0
        Seed of the random generator.
    background : tuple of float, default (1/3, 1/3, 1/3)
        State to perturb, e.g. the fixed point ``fixed_point(rates)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, Nx, Nx)``.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(background, dtype=float).reshape(3, 1, 1)
    return normalise(base / base.sum() + noise * rng.standard_normal((3, Nx, Nx)))


def stripes(Nx: int, axis: Literal["x", "y"] = "x") -> np.ndarray:
    """Three equal pure-species stripes ``S | R | P``.

    Parameters
    ----------
    Nx : int
        Grid cells per axis.
    axis : {'x', 'y'}, default 'x'
        Axis along which the stripes follow each other.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, Nx, Nx)``.
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    idx = np.minimum(3 * np.arange(Nx) // Nx, 2)
    band = np.eye(3)[:, idx]  # (3, Nx): one-hot species per column
    rho = np.broadcast_to(band[:, None, :], (3, Nx, Nx)).copy()
    return rho if axis == "x" else rho.transpose(0, 2, 1).copy()


def blobs(Nx: int, L: float = 1.0, n_blobs: int = 6, seed: int = 0, width: float = 0.08) -> np.ndarray:
    """Random Gaussian blobs of each species on a uniform background.

    Parameters
    ----------
    Nx : int
        Grid cells per axis.
    L : float, default 1.0
        Side length of the bounding square.
    n_blobs : int, default 6
        Number of blobs *per species*.
    seed : int, default 0
        Seed of the random generator.
    width : float, default 0.08
        Blob standard deviation as a fraction of ``L``.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, Nx, Nx)``.
    """
    rng = np.random.default_rng(seed)
    X, Y = grid(Nx, L)
    s = width * L
    rho = np.full((3, Nx, Nx), 0.05)
    for k in range(3):
        for cx, cy in rng.uniform(0.0, L, size=(n_blobs, 2)):
            rho[k] += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s**2))
    return normalise(rho)


def concentrated(
    Nx: int,
    L: float = 1.0,
    centre: tuple[float, float] = (0.5, 0.5),
    radius: float = 0.15,
    u0: tuple[float, float, float] = (0.35, 0.32, 0.33),
    background: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> np.ndarray:
    """Given fractions inside a disk, a coexistence state outside.

    A small localised perturbation of the fixed point keeps the dynamics in
    the linear regime, so the mean fractions oscillate cleanly at
    ``ω₀ = √(λ_S λ_R λ_P / Σλ)`` (``λ / √3`` for equal rates).

    Parameters
    ----------
    Nx : int
        Grid cells per axis.
    L : float, default 1.0
        Side length of the bounding square.
    centre : tuple of float, default (0.5, 0.5)
        Disk centre in units of ``L``.
    radius : float, default 0.15
        Disk radius in units of ``L``.
    u0 : tuple of float, default (0.35, 0.32, 0.33)
        Fractions inside the disk; rescaled to sum to 1.
    background : tuple of float, default (1/3, 1/3, 1/3)
        Fractions outside the disk, e.g. the fixed point ``fixed_point(rates)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, Nx, Nx)``.
    """
    X, Y = grid(Nx, L)
    inside = (X - centre[0] * L) ** 2 + (Y - centre[1] * L) ** 2 <= (radius * L) ** 2
    rho = homogeneous(Nx, background)
    u = np.asarray(u0, dtype=float)
    rho[:, inside] = (u / u.sum())[:, None]
    return rho
