"""Explicit finite-difference solver for the cyclic RPS reaction-diffusion PDE.

    ∂_t ρ_i = (σ²/2) ∇² ρ_i + f_i(ρ),   i ∈ {S, R, P}

    f_S = ρ_S (λ_S ρ_P − λ_R ρ_R)
    f_R = ρ_R (λ_R ρ_S − λ_P ρ_P)
    f_P = ρ_P (λ_P ρ_R − λ_S ρ_S)

with three constant interaction rates: λ_S (Scissors beats Paper), λ_R
(Rock beats Scissors) and λ_P (Paper beats Rock). No-flux boundary
conditions hold on the boundary of an arbitrarily shaped domain Ω ⊂ [0, L]².
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from ._numerics import laplacian_neumann, max_stable_dt, neighbour_masks
from .domain import Domain
from .initial import normalise

__all__ = ["SimResult", "RPSSimulator", "Rates", "as_rates", "fixed_point", "omega0", "reaction"]

SPECIES: tuple[str, str, str] = ("S", "R", "P")

Rates = tuple[float, float, float]
Method = Literal["heun", "euler"]

#: Largest λ_max·dt before a warning, per method. Per step, explicit Euler
#: inflates a neutral cycle by about (ω dt)²/2, Heun only by about (ω dt)⁴/8.
_REACTION_DT_WARN: dict[str, float] = {"euler": 0.05, "heun": 0.25}


def as_rates(rates: float | tuple[float, float, float]) -> Rates:
    """Normalise a rate specification to ``(λ_S, λ_R, λ_P)``.

    Parameters
    ----------
    rates : float or tuple of float
        A single value (all three rates equal) or ``(λ_S, λ_R, λ_P)``.

    Returns
    -------
    tuple of float
        ``(λ_S, λ_R, λ_P)``, all non-negative.

    Raises
    ------
    ValueError
        If there are not exactly three rates or any rate is negative.
    """
    arr = np.full(3, float(rates)) if np.ndim(rates) == 0 else np.asarray(rates, dtype=float)
    if arr.shape != (3,):
        raise ValueError("rates must be a scalar or a triple (lambda_S, lambda_R, lambda_P)")
    if np.any(arr < 0):
        raise ValueError("interaction rates must be non-negative")
    return float(arr[0]), float(arr[1]), float(arr[2])


def fixed_point(rates: float | tuple[float, float, float]) -> Rates:
    """Interior coexistence fixed point ``ρ*`` of the reaction.

    Setting ``f = 0`` with all species present gives
    ``ρ* = (λ_P, λ_S, λ_R) / (λ_S + λ_R + λ_P)``: each species' equilibrium
    share is set by the rate at which its *prey* is beaten by the third
    species ("survival of the weakest").

    Parameters
    ----------
    rates : float or tuple of float
        ``(λ_S, λ_R, λ_P)`` or a single common rate.

    Returns
    -------
    tuple of float
        ``(ρ*_S, ρ*_R, ρ*_P)``.
    """
    lS, lR, lP = as_rates(rates)
    total = lS + lR + lP
    if total == 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return lP / total, lS / total, lR / total


def omega0(rates: float | tuple[float, float, float]) -> float:
    """Angular frequency of the linearised oscillation around ``ρ*``.

    ``ω₀ = √(λ_S λ_R λ_P / (λ_S + λ_R + λ_P))``, which reduces to
    ``λ / √3`` for equal rates.

    Parameters
    ----------
    rates : float or tuple of float
        ``(λ_S, λ_R, λ_P)`` or a single common rate.

    Returns
    -------
    float
        ``ω₀`` (0 if all rates vanish).
    """
    lS, lR, lP = as_rates(rates)
    total = lS + lR + lP
    return float(np.sqrt(lS * lR * lP / total)) if total > 0 else 0.0


def reaction(rho: np.ndarray, rates: float | tuple[float, float, float] = 1.0) -> np.ndarray:
    """Cyclic RPS reaction term ``f(ρ)``.

    Parameters
    ----------
    rho : np.ndarray
        Array of shape ``(3, ...)`` ordered ``(ρ_S, ρ_R, ρ_P)``.
    rates : float or tuple of float, default 1.0
        ``(λ_S, λ_R, λ_P)`` or a single common rate.

    Returns
    -------
    np.ndarray
        ``(f_S, f_R, f_P)`` with the same shape as ``rho``.
    """
    rho = np.asarray(rho, dtype=float)
    flat = rho.reshape(3, -1)  # a single state (3,) becomes (3, 1)
    out = np.empty_like(flat)
    _reaction_into(flat, as_rates(rates), out, np.empty_like(flat[0]))
    return out.reshape(rho.shape)


def _reaction_into(rho: np.ndarray, rates: Rates, out: np.ndarray, tmp: np.ndarray) -> None:
    lS, lR, lP = rates
    s, r, p = rho
    # f_S = ρ_S (λ_S ρ_P − λ_R ρ_R)
    np.multiply(p, lS, out=out[0])
    np.multiply(r, lR, out=tmp)
    out[0] -= tmp
    out[0] *= s
    # f_R = ρ_R (λ_R ρ_S − λ_P ρ_P)
    np.multiply(s, lR, out=out[1])
    np.multiply(p, lP, out=tmp)
    out[1] -= tmp
    out[1] *= r
    # f_P = ρ_P (λ_P ρ_R − λ_S ρ_S)
    np.multiply(r, lP, out=out[2])
    np.multiply(s, lS, out=tmp)
    out[2] -= tmp
    out[2] *= p


def _normalise_inplace(rho: np.ndarray, total: np.ndarray) -> None:
    np.clip(rho, 0.0, 1.0, out=rho)
    np.add(rho[0], rho[1], out=total)
    total += rho[2]
    if not total.all():  # a cell lost every species: reset it to the centre
        empty = total == 0.0
        rho[:, empty] = 1.0 / 3.0
        total[empty] = 1.0
    rho /= total


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
    rates : tuple of float
        Interaction rates ``(λ_S, λ_R, λ_P)``.
    sigma : float
        Noise amplitude σ (diffusion coefficient ``D = σ²/2``).
    L : float
        Side length of the bounding square.
    Nx : int
        Grid cells per axis.
    mask : np.ndarray
        Boolean domain mask, shape ``(Nx, Nx)``.
    variances : np.ndarray
        Spatial variance of each species over the domain, shape ``(T, 3)``.
    dt : float
        Integration time step.
    """

    t: np.ndarray
    snapshots: list[np.ndarray]
    fractions: np.ndarray
    rates: Rates
    sigma: float
    L: float
    Nx: int
    mask: np.ndarray = field(default_factory=lambda: np.ones((0, 0), dtype=bool))
    variances: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    dt: float = float("nan")

    @property
    def omega0(self) -> float:
        """Linearised angular frequency ``√(λ_S λ_R λ_P / Σλ)``."""
        return omega0(self.rates)

    @property
    def fixed_point(self) -> Rates:
        """Interior fixed point ``ρ* = (λ_P, λ_S, λ_R) / Σλ``."""
        return fixed_point(self.rates)


class RPSSimulator:
    """Explicit finite-difference solver (Heun or Euler) for the RPS reaction-diffusion PDE.

    Parameters
    ----------
    rates : float or tuple of float
        Interaction rates ``(λ_S, λ_R, λ_P)``: Scissors beats Paper at rate
        ``λ_S``, Rock beats Scissors at rate ``λ_R`` and Paper beats Rock at
        rate ``λ_P``. A single number sets all three equal. The rates are
        constant in space and time.
    sigma : float
        Noise amplitude σ; the diffusion coefficient is ``D = σ²/2``.
    L : float, default 1.0
        Side length of the bounding square ``[0, L]²``.
    dt : float, default 0.005
        Time step.
    Nx : int, optional
        Grid cells per axis. Taken from ``domain`` when that is given,
        otherwise 64.
    domain : Domain or np.ndarray, optional
        Shape of the domain, as a :class:`~rps_diffusion.Domain` or a boolean
        ``(Nx, Nx)`` mask. ``None`` means the full square. No-flux boundary
        conditions hold on the boundary of whatever shape is given.
    method : {'heun', 'euler'}, default 'heun'
        Time integrator. ``'heun'`` (explicit trapezoidal RK2) is
        second-order accurate and permits a much larger ``dt`` than forward
        ``'euler'`` for the same accuracy of the reaction cycles. Both have
        the same diffusive stability limit.

    Raises
    ------
    ValueError
        If shapes are inconsistent, the domain is empty, or ``dt`` violates
        the diffusive CFL condition ``dt <= dx² / (4 D)`` (the 2-D form of
        ``dx² / (2 D)``; ``dx = L / Nx``).
    """

    def __init__(
        self,
        rates: float | tuple[float, float, float],
        sigma: float,
        L: float = 1.0,
        dt: float = 0.005,
        Nx: int | None = None,
        domain: Domain | np.ndarray | None = None,
        method: Method = "heun",
    ) -> None:
        self.rates: Rates = as_rates(rates)
        if sigma < 0 or L <= 0 or dt <= 0:
            raise ValueError("require sigma >= 0, L > 0, dt > 0")
        if method not in _REACTION_DT_WARN:
            raise ValueError(f"unknown method {method!r}; use 'heun' or 'euler'")

        if domain is None:
            Nx = 64 if Nx is None else Nx
            mask = np.ones((Nx, Nx), dtype=bool)
        elif isinstance(domain, Domain):
            Nx = domain.Nx if Nx is None else Nx
            if domain.Nx != Nx or not np.isclose(domain.L, L):
                raise ValueError("Domain grid (Nx, L) does not match the simulator")
            domain.validate()
            mask = domain.mask
        else:
            mask = np.asarray(domain, dtype=bool)
            Nx = mask.shape[0] if Nx is None else Nx
            if mask.shape != (Nx, Nx):
                raise ValueError(f"domain mask has shape {mask.shape}, expected {(Nx, Nx)}")
            Domain.from_mask(mask, L).validate()

        self.Nx: int = int(Nx)
        self.L: float = float(L)
        self.dx: float = self.L / self.Nx
        self.sigma: float = float(sigma)
        self.D: float = 0.5 * self.sigma**2
        self.dt: float = float(dt)
        self.method: Method = method
        self.mask: np.ndarray = mask
        self._inside = mask.astype(float)  # switches the reaction off outside Ω
        self._faces = None if mask.all() else neighbour_masks(mask)

        dt_max = max_stable_dt(self.dx, self.D, ndim=2)
        if self.dt > dt_max:
            raise ValueError(
                f"dt = {self.dt:g} violates the CFL condition dt <= dx²/(4D) = {dt_max:g} "
                f"(dx = {self.dx:g}, D = σ²/2 = {self.D:g})"
            )
        lam_max = max(self.rates)
        if lam_max * self.dt > _REACTION_DT_WARN[method]:
            warnings.warn(
                f"λ_max·dt = {lam_max * self.dt:.3g} is large for method {method!r}; the "
                "integrator will visibly inflate the reaction cycles. Consider a smaller dt.",
                stacklevel=2,
            )

        # Preallocated buffers for the in-place stepping kernel.
        shape = (3, self.Nx, self.Nx)
        self._k1 = np.empty(shape)
        self._k2 = np.empty(shape)
        self._tmp = np.empty(shape)
        self._lap = np.empty(shape)
        self._cell = np.empty((self.Nx, self.Nx))
        self._total = np.empty((self.Nx, self.Nx))
        self._work = (np.empty((3, self.Nx - 1, self.Nx)), np.empty((3, self.Nx, self.Nx - 1)))

    # ------------------------------------------------------------------
    def rhs(self, rho: np.ndarray) -> np.ndarray:
        """Right-hand side ``D ∇²ρ + f(ρ)`` of the PDE.

        Parameters
        ----------
        rho : np.ndarray
            State of shape ``(3, Nx, Nx)``.

        Returns
        -------
        np.ndarray
            Time derivative, shape ``(3, Nx, Nx)``.
        """
        out = np.empty((3, self.Nx, self.Nx))
        self._rhs_into(np.asarray(rho, dtype=float), out)
        return out

    def _rhs_into(self, rho: np.ndarray, out: np.ndarray) -> None:
        _reaction_into(rho, self.rates, out, self._cell)
        if self._faces is not None:
            out *= self._inside
        if self.D > 0:
            laplacian_neumann(rho, self.dx, self._faces, out=self._lap, work=self._work, scale=self.D)
            out += self._lap

    def _step_inplace(self, rho: np.ndarray) -> None:
        dt, k1 = self.dt, self._k1
        self._rhs_into(rho, k1)
        if self.method == "euler":
            k1 *= dt
            rho += k1
        else:  # Heun: predictor with Euler, corrector with the trapezoidal rule
            k2, tmp = self._k2, self._tmp
            np.multiply(k1, dt, out=tmp)
            tmp += rho
            self._rhs_into(tmp, k2)
            k1 += k2
            k1 *= 0.5 * dt
            rho += k1
        _normalise_inplace(rho, self._total)

    def step(self, rho: np.ndarray) -> np.ndarray:
        """Advance one time step, then clip and renormalise.

        Parameters
        ----------
        rho : np.ndarray
            State of shape ``(3, Nx, Nx)``.

        Returns
        -------
        np.ndarray
            New state with ``ρ_S + ρ_R + ρ_P = 1`` in every cell.
        """
        new = np.array(rho, dtype=float)
        self._step_inplace(new)
        return new

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
        rho[:, ~self.mask] = 1.0 / 3.0  # inert filler outside Ω (no reaction, no flux)
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
            self._step_inplace(rho)
            if n % save_every == 0:
                record(n)

        return SimResult(
            t=np.asarray(times),
            snapshots=snaps,
            fractions=np.asarray(means),
            rates=self.rates,
            sigma=self.sigma,
            L=self.L,
            Nx=self.Nx,
            mask=self.mask.copy(),
            variances=np.asarray(varis),
            dt=self.dt,
        )
