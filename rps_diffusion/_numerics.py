"""Private numerical kernels: Neumann Laplacian on masked domains and CFL checks."""

from __future__ import annotations

import numpy as np

__all__ = ["Faces", "neighbour_masks", "laplacian_neumann", "max_stable_dt"]

#: Open-face indicators ``(fy, fx)`` of shapes ``(Ny-1, Nx)`` and ``(Ny, Nx-1)``.
Faces = tuple[np.ndarray, np.ndarray]


def neighbour_masks(mask: np.ndarray) -> Faces:
    """Precompute which interior cell faces are open (both adjacent cells inside).

    Parameters
    ----------
    mask : np.ndarray
        Boolean array of shape ``(Ny, Nx)``; ``True`` for cells inside the domain.

    Returns
    -------
    fy, fx : np.ndarray
        Float arrays of shape ``(Ny-1, Nx)`` and ``(Ny, Nx-1)``. ``fy[j, i]``
        is 1 if cells ``(j, i)`` and ``(j+1, i)`` are both inside the domain,
        ``fx[j, i]`` likewise for ``(j, i)`` and ``(j, i+1)``. Closed faces
        carry no flux, which is the discrete no-flux (Neumann) condition.
    """
    m = np.asarray(mask, dtype=bool)
    fy = (m[1:, :] & m[:-1, :]).astype(float)
    fx = (m[:, 1:] & m[:, :-1]).astype(float)
    return fy, fx


def laplacian_neumann(
    u: np.ndarray,
    dx: float,
    faces: Faces | None = None,
    out: np.ndarray | None = None,
    work: tuple[np.ndarray, np.ndarray] | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Five-point finite-volume Laplacian with no-flux boundary conditions.

    The Laplacian of a cell is the sum of the fluxes ``(u_j - u_i) / dx²``
    through its open faces. Faces on the edge of the bounding box, and faces
    between an inside and an outside cell, are closed. On the full square
    this is identical to the ghost-cell scheme with
    ``u_ghost = u_boundary``. On any masked domain it enforces zero normal
    flux on the staircase boundary and conserves ``sum(u)`` over the domain
    exactly.

    Parameters
    ----------
    u : np.ndarray
        Field of shape ``(..., Ny, Nx)``. Leading axes are treated as a batch.
    dx : float
        Grid spacing (identical in both directions).
    faces : Faces, optional
        Output of :func:`neighbour_masks`. ``None`` means the full rectangle.
    out : np.ndarray, optional
        Preallocated result array with the shape of ``u``.
    work : tuple of np.ndarray, optional
        Preallocated flux buffers of shapes ``(..., Ny-1, Nx)`` and
        ``(..., Ny, Nx-1)``, which avoid temporary allocations in hot loops.
    scale : float, default 1.0
        Factor applied to the result (e.g. the diffusion coefficient).

    Returns
    -------
    np.ndarray
        ``scale * ∇²u``, with the shape of ``u``. Values outside the domain
        are 0.
    """
    if out is None:
        out = np.empty_like(u, dtype=float)
    if work is None:
        work = (np.empty(u.shape[:-2] + (u.shape[-2] - 1, u.shape[-1])),
                np.empty(u.shape[:-1] + (u.shape[-1] - 1,)))
    gy, gx = work

    out.fill(0.0)
    np.subtract(u[..., 1:, :], u[..., :-1, :], out=gy)
    np.subtract(u[..., :, 1:], u[..., :, :-1], out=gx)
    if faces is not None:
        gy *= faces[0]
        gx *= faces[1]
    out[..., :-1, :] += gy
    out[..., 1:, :] -= gy
    out[..., :, :-1] += gx
    out[..., :, 1:] -= gx
    out *= scale / dx**2
    return out


def max_stable_dt(dx: float, D: float, ndim: int = 2) -> float:
    """Largest stable explicit time step for the diffusion equation.

    Holds for both explicit Euler and Heun (RK2), whose stability regions
    cover the same interval ``[-2, 0]`` of the negative real axis.

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
