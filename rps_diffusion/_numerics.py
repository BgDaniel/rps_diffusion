"""Private numerical kernels: Neumann Laplacian on masked domains and CFL checks."""

from __future__ import annotations

import numpy as np

__all__ = ["NeighbourMasks", "neighbour_masks", "laplacian_neumann", "max_stable_dt"]

NeighbourMasks = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def neighbour_masks(mask: np.ndarray) -> NeighbourMasks:
    """Precompute which cell faces are open (both adjacent cells inside the domain).

    Parameters
    ----------
    mask : np.ndarray
        Boolean array of shape ``(Ny, Nx)``; ``True`` for cells inside the domain.

    Returns
    -------
    tuple of np.ndarray
        Float arrays ``(up, down, left, right)`` of shape ``(Ny, Nx)``. Entry
        ``[j, i]`` of ``right`` is 1 if cell ``(j, i)`` and its neighbour
        ``(j, i+1)`` are both inside the domain, else 0. Closed faces carry no
        flux, which is the discrete no-flux (Neumann) boundary condition.
    """
    m = np.asarray(mask, dtype=bool)
    p = np.pad(m, 1, mode="constant", constant_values=False)
    up = (m & p[2:, 1:-1]).astype(float)
    down = (m & p[:-2, 1:-1]).astype(float)
    right = (m & p[1:-1, 2:]).astype(float)
    left = (m & p[1:-1, :-2]).astype(float)
    return up, down, left, right


def laplacian_neumann(
    u: np.ndarray,
    dx: float,
    faces: NeighbourMasks | None = None,
) -> np.ndarray:
    """Five-point finite-volume Laplacian with no-flux boundary conditions.

    For every cell the Laplacian is the sum of the fluxes through its four
    faces, ``sum_j (u_j - u_i) / dx**2``, where only open faces (both cells
    inside the domain) contribute. On the full square this is identical to the
    classical ghost-cell scheme with ``u_ghost = u_boundary``; on an arbitrary
    masked domain it enforces zero normal flux on the staircase boundary and
    conserves ``sum(u)`` over the domain exactly.

    Parameters
    ----------
    u : np.ndarray
        Field of shape ``(..., Ny, Nx)``. Leading axes are treated as a batch.
    dx : float
        Grid spacing (identical in both directions).
    faces : NeighbourMasks, optional
        Output of :func:`neighbour_masks`. ``None`` means the full rectangle.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``u``. Values outside the domain are 0.
    """
    p = np.pad(u, [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    c = p[..., 1:-1, 1:-1]
    d_up = p[..., 2:, 1:-1] - c
    d_down = p[..., :-2, 1:-1] - c
    d_right = p[..., 1:-1, 2:] - c
    d_left = p[..., 1:-1, :-2] - c
    if faces is None:
        # Edge padding makes boundary differences vanish: ghost-cell Neumann BC.
        return (d_up + d_down + d_left + d_right) / dx**2
    up, down, left, right = faces
    return (up * d_up + down * d_down + left * d_left + right * d_right) / dx**2


def max_stable_dt(dx: float, D: float, ndim: int = 2) -> float:
    """Largest stable explicit-Euler time step for the diffusion equation.

    Parameters
    ----------
    dx : float
        Grid spacing.
    D : float
        Diffusion coefficient (``sigma**2 / 2`` for the RPS model).
    ndim : int, default 2
        Spatial dimension. The 1-D bound is ``dx**2 / (2 D)``; in ``ndim``
        dimensions the five-point stencil requires ``dx**2 / (2 ndim D)``.

    Returns
    -------
    float
        The stability limit; ``inf`` if ``D == 0``.
    """
    if D <= 0.0:
        return float("inf")
    return dx**2 / (2.0 * ndim * D)
