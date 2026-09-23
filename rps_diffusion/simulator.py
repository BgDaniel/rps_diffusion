"""Explicit finite-difference solver for the cyclic RPS reaction-diffusion PDE.

    ∂_t ρ_i = (σ²/2) ∇² ρ_i + λ(x) f_i(ρ),   i ∈ {S, R, P}

    f_S = ρ_S (ρ_P − ρ_R),  f_R = ρ_R (ρ_S − ρ_P),  f_P = ρ_P (ρ_R − ρ_S)

with no-flux boundary conditions on the boundary of an arbitrarily shaped
domain Ω ⊂ [0, L]².
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from ._numerics import laplacian_neumann, max_stable_dt, neighbour_masks
from .domain import Domain
from .initial import normalise

__all__ = ["SimResult", "RPSSimulator", "reaction"]

SPECIES: tuple[str, str, str] = ("S", "R", "P")


def reaction(rho: np.ndarray) -> np.ndarray:
    """Cyclic RPS reaction term ``f(ρ)`` without the rate λ.

    Parameters
    ----------
    rho : np.ndarray
        Array of shape ``(3, ...)`` ordered ``(ρ_S, ρ_R, ρ_P)``.

    Returns
    -------
    np.ndarray
        ``(f_S, f_R, f_P)`` with the same shape as ``rho``.
    """
    s, r, p = rho
    return np.stack([s * (p - r), r * (s - p), p * (r - s)])


@dataclass
class SimResult:
    """Output of :meth:`RPSSimulator.run`.

    Attributes
    ----------
    t : np.ndarray
        Saved times, shape ``(T,)``.
    snapshots : list of np.ndarray
        ``T`` density fields of shape ``(3, Nx, Nx)``, stored as float32 to
        save memory. Cells outside the domain are ``NaN``.
    fractions : np.ndarray
        Spatial mean of each species over the domain, shape ``(T, 3)``.
    variances : np.ndarray
        Spatial variance of each species over the domain, shape ``(T, 3)``.
    lambda_field : np.ndarray
        The λ(x, y) array used, shape ``(Nx, Nx)``.
    sigma : float
        Noise amplitude σ (diffusion coefficient ``D = σ²/2``).
    L : float
        Side length of the bounding square.
    Nx : int
        Grid cells per axis.
    mask : np.ndarray
        Boolean domain mask, shape ``(Nx, Nx)``.
    dt : float
        Integration time step.
    """

    t: np.ndarray
    snapshots: list[np.ndarray]
    fractions: np.ndarray
    lambda_field: np.ndarray
    sigma: float
    L: float
    Nx: int
    mask: np.ndarray = field(default_factory=lambda: np.ones((0, 0), dtype=bool))
    variances: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    dt: float = float("nan")

    @property
    def lambda_mean(self) -> float:
        """Mean of λ over the domain."""
        return float(self.lambda_field[self.mask].mean())

    @property
    def omega0(self) -> float:
        """Linearised angular frequency ``λ_mean / √3``."""
        return self.lambda_mean / np.sqrt(3.0)


class RPSSimulator:
    """Explicit-Euler finite-difference solver for the RPS reaction-diffusion PDE.

    Parameters
    ----------
    lambda_field : np.ndarray
        Interaction rate λ(x, y), shape ``(Nx, Nx)`` (see
        :class:`~rps_diffusion.LambdaField`).
    sigma : float
        Noise amplitude σ; the diffusion coefficient is ``D = σ²/2``.
    L : float, default 1.0
        Side length of the bounding square ``[0, L]²``.
    dt : float, default 0.005
        Time step.
    Nx : int, optional
        Grid cells per axis. Inferred from ``lambda_field`` when omitted;
        if given it must match.
    domain : Domain or np.ndarray, optional
        Shape of the domain, as a :class:`~rps_diffusion.Domain` or a boolean
        ``(Nx, Nx)`` mask. ``None`` means the full square. No-flux boundary
        conditions hold on the boundary of whatever shape is given.

    Raises
    ------
    ValueError
        If shapes are inconsistent, the domain is empty, or ``dt`` violates
        the diffusive CFL condition ``dt <= dx² / (4 D)`` (the 2-D form of
        ``dx² / (2 D)``; ``dx = L / Nx``).
    """

    def __init__(
        self,
        lambda_field: np.ndarray,
        sigma: float,
        L: float = 1.0,
        dt: float = 0.005,
        Nx: int | None = None,
        domain: Domain | np.ndarray | None = None,
    ) -> None:
        lam = np.asarray(lambda_field, dtype=float)
        if lam.ndim != 2 or lam.shape[0] != lam.shape[1]:
            raise ValueError("lambda_field must be a square 2-D array")
        if Nx is None:
            Nx = lam.shape[0]
        if lam.shape != (Nx, Nx):
            raise ValueError(f"lambda_field has shape {lam.shape}, expected {(Nx, Nx)}")
        if np.any(lam < 0):
            raise ValueError("lambda_field must be non-negative")
        if sigma < 0 or L <= 0 or dt <= 0:
            raise ValueError("require sigma >= 0, L > 0, dt > 0")

        if domain is None:
            mask = np.ones((Nx, Nx), dtype=bool)
        elif isinstance(domain, Domain):
            if domain.Nx != Nx or not np.isclose(domain.L, L):
                raise ValueError("Domain grid (Nx, L) does not match the simulator")
            domain.validate()
            mask = domain.mask
        else:
            mask = np.asarray(domain, dtype=bool)
            if mask.shape != (Nx, Nx):
                raise ValueError(f"domain mask has shape {mask.shape}, expected {(Nx, Nx)}")
            Domain.from_mask(mask, L).validate()

        self.Nx: int = int(Nx)
        self.L: float = float(L)
        self.dx: float = self.L / self.Nx
        self.sigma: float = float(sigma)
        self.D: float = 0.5 * self.sigma**2
        self.dt: float = float(dt)
        self.mask: np.ndarray = mask
        self.lambda_field: np.ndarray = np.where(mask, lam, 0.0)
        self._faces = None if mask.all() else neighbour_masks(mask)

        dt_max = max_stable_dt(self.dx, self.D, ndim=2)
        if self.dt > dt_max:
            raise ValueError(
                f"dt = {self.dt:g} violates the CFL condition dt <= dx²/(4D) = {dt_max:g} "
                f"(dx = {self.dx:g}, D = σ²/2 = {self.D:g})"
            )
        lam_max = float(self.lambda_field.max())
        if lam_max * self.dt > 0.05:
            warnings.warn(
                f"λ_max·dt = {lam_max * self.dt:.3g} is large; explicit Euler will visibly "
                "inflate the reaction cycles. Consider a smaller dt.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    def rhs(self, rho: np.ndarray) -> np.ndarray:
        """Right-hand side ``D ∇²ρ + λ f(ρ)`` of the PDE.

        Parameters
        ----------
        rho : np.ndarray
            State of shape ``(3, Nx, Nx)``.

        Returns
        -------
        np.ndarray
            Time derivative, shape ``(3, Nx, Nx)``.
        """
        out = self.lambda_field * reaction(rho)
        if self.D > 0:
            out += self.D * laplacian_neumann(rho, self.dx, self._faces)
        return out

    def step(self, rho: np.ndarray) -> np.ndarray:
        """Advance one explicit-Euler step, then clip and renormalise.

        Parameters
        ----------
        rho : np.ndarray
            State of shape ``(3, Nx, Nx)``.

        Returns
        -------
        np.ndarray
            New state with ``ρ_S + ρ_R + ρ_P = 1`` in every cell.
        """
        return normalise(rho + self.dt * self.rhs(rho))

    def _record(self, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inside = rho[:, self.mask]
        snap = np.where(self.mask, rho, np.nan).astype(np.float32)
        return snap, inside.mean(axis=1), inside.var(axis=1)

    def run(
        self,
        rho0: np.ndarray,
        t_max: float,
        save_every: int = 10,
        progress: bool = True,
    ) -> SimResult:
        """Integrate the PDE from ``rho0`` up to ``t_max``.

        Parameters
        ----------
        rho0 : np.ndarray
            Initial condition, shape ``(3, Nx, Nx)`` (see
            :mod:`rps_diffusion.initial`). Values outside the domain are
            ignored; the rest is clipped and normalised.
        t_max : float
            Final time.
        save_every : int, default 10
            Record a snapshot every ``save_every`` steps (plus ``t = 0``).
        progress : bool, default True
            Show a tqdm progress bar.

        Returns
        -------
        SimResult
            Saved times, snapshots and domain-averaged statistics.
        """
        rho0 = np.asarray(rho0, dtype=float)
        if rho0.shape != (3, self.Nx, self.Nx):
            raise ValueError(f"rho0 has shape {rho0.shape}, expected {(3, self.Nx, self.Nx)}")
        if save_every < 1:
            raise ValueError("save_every must be >= 1")

        rho = normalise(rho0)
        rho[:, ~self.mask] = 1.0 / 3.0  # inert filler outside Ω (λ = 0, no flux)
        n_steps = int(round(t_max / self.dt))

        times: list[float] = []
        snaps: list[np.ndarray] = []
        means: list[np.ndarray] = []
        varis: list[np.ndarray] = []

        def record(n: int) -> None:
            snap, mean, var = self._record(rho)
            times.append(n * self.dt)
            snaps.append(snap)
            means.append(mean)
            varis.append(var)

        record(0)
        for n in tqdm(range(1, n_steps + 1), disable=not progress, desc="RPS", unit="step"):
            rho = self.step(rho)
            if n % save_every == 0:
                record(n)

        return SimResult(
            t=np.asarray(times),
            snapshots=snaps,
            fractions=np.asarray(means),
            lambda_field=self.lambda_field.copy(),
            sigma=self.sigma,
            L=self.L,
            Nx=self.Nx,
            mask=self.mask.copy(),
            variances=np.asarray(varis),
            dt=self.dt,
        )
