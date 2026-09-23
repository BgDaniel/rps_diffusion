"""Spatial geometry: simulation domains of arbitrary shape and the λ(x, y) rate field.

All objects live on a cell-centred grid covering the bounding box ``[0, L]²``
with ``Nx × Nx`` cells. Cell ``[j, i]`` has centre ``((i + ½) dx, (j + ½) dx)``
with ``dx = L / Nx``; the first array axis is *y*, the second is *x*.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from matplotlib.path import Path
from scipy import ndimage

__all__ = ["grid", "Domain", "LambdaField"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def grid(Nx: int, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates of the ``Nx × Nx`` grid on ``[0, L]²``.

    Parameters
    ----------
    Nx : int
        Number of cells per axis.
    L : float
        Side length of the bounding square.

    Returns
    -------
    X, Y : np.ndarray
        Arrays of shape ``(Nx, Nx)``; ``X[j, i]`` and ``Y[j, i]`` are the
        coordinates of cell ``[j, i]``.
    """
    c = (np.arange(Nx) + 0.5) * (L / Nx)
    X, Y = np.meshgrid(c, c, indexing="xy")
    return X, Y


def _rect_mask(X: np.ndarray, Y: np.ndarray, x0: float, y0: float, w: float, h: float) -> np.ndarray:
    return (X >= x0) & (X <= x0 + w) & (Y >= y0) & (Y <= y0 + h)


def _disk_mask(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r**2


def _annulus_mask(
    X: np.ndarray, Y: np.ndarray, cx: float, cy: float, r_inner: float, r_outer: float
) -> np.ndarray:
    if r_inner > r_outer:
        raise ValueError("r_inner must not exceed r_outer")
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    return (r2 >= r_inner**2) & (r2 <= r_outer**2)


def _ellipse_mask(
    X: np.ndarray, Y: np.ndarray, cx: float, cy: float, a: float, b: float, angle: float
) -> np.ndarray:
    ca, sa = np.cos(angle), np.sin(angle)
    xr = (X - cx) * ca + (Y - cy) * sa
    yr = -(X - cx) * sa + (Y - cy) * ca
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def _polygon_mask(X: np.ndarray, Y: np.ndarray, vertices: Sequence[tuple[float, float]]) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
        raise ValueError("vertices must be a sequence of at least three (x, y) pairs")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return Path(verts).contains_points(pts).reshape(X.shape)


class _Grid:
    """Shared grid bookkeeping for :class:`Domain` and :class:`LambdaField`."""

    def __init__(self, Nx: int, L: float) -> None:
        if Nx < 3:
            raise ValueError("Nx must be at least 3")
        if L <= 0:
            raise ValueError("L must be positive")
        self.Nx: int = int(Nx)
        self.L: float = float(L)
        self.dx: float = self.L / self.Nx
        self.X, self.Y = grid(self.Nx, self.L)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
class Domain(_Grid):
    """Simulation domain of arbitrary shape inside the bounding box ``[0, L]²``.

    The domain is a boolean mask over the grid. Cells outside the mask are
    not simulated, and no flux crosses the domain boundary (Neumann
    condition), whatever its shape. Shapes are built by chaining ``add_*``
    (union) and ``cut_*`` (set difference) calls, applied in order.

    Parameters
    ----------
    Nx : int
        Number of cells per axis of the bounding box.
    L : float
        Side length of the bounding box.
    full : bool, default True
        Start from the full square (``True``) or from the empty set
        (``False``, useful before a sequence of ``add_*`` calls).

    Examples
    --------
    An annulus with a notch, built by chaining:

    >>> dom = (Domain(128, 1.0, full=False)
    ...        .add_disk(0.5, 0.5, 0.45)
    ...        .cut_disk(0.5, 0.5, 0.2)
    ...        .cut_rectangle(0.45, 0.0, 0.1, 0.5))

    Ready-made shapes:

    >>> Domain.disk(128, 1.0)            # inscribed disk
    >>> Domain.l_shape(128, 1.0)         # L-shaped room
    """

    def __init__(self, Nx: int, L: float = 1.0, full: bool = True) -> None:
        super().__init__(Nx, L)
        self._mask = np.full((self.Nx, self.Nx), bool(full))

    # ---- construction: union -------------------------------------------------
    def _apply(self, region: np.ndarray, add: bool) -> Domain:
        if add:
            self._mask |= region
        else:
            self._mask &= ~region
        return self

    def add_rectangle(self, x0: float, y0: float, w: float, h: float) -> Domain:
        """Add the axis-aligned rectangle with corner ``(x0, y0)``, width ``w``, height ``h``."""
        return self._apply(_rect_mask(self.X, self.Y, x0, y0, w, h), True)

    def add_square(self, x0: float, y0: float, w: float) -> Domain:
        """Add the axis-aligned square with corner ``(x0, y0)`` and side ``w``."""
        return self.add_rectangle(x0, y0, w, w)

    def add_disk(self, cx: float, cy: float, r: float) -> Domain:
        """Add the filled disk with centre ``(cx, cy)`` and radius ``r``."""
        return self._apply(_disk_mask(self.X, self.Y, cx, cy, r), True)

    def add_annulus(self, cx: float, cy: float, r_inner: float, r_outer: float) -> Domain:
        """Add the ring ``r_inner <= |x - c| <= r_outer``."""
        return self._apply(_annulus_mask(self.X, self.Y, cx, cy, r_inner, r_outer), True)

    def add_ellipse(self, cx: float, cy: float, a: float, b: float, angle: float = 0.0) -> Domain:
        """Add an ellipse with semi-axes ``a``, ``b``, rotated by ``angle`` radians."""
        return self._apply(_ellipse_mask(self.X, self.Y, cx, cy, a, b, angle), True)

    def add_polygon(self, vertices: Sequence[tuple[float, float]]) -> Domain:
        """Add the (closed, possibly non-convex) polygon with the given vertices."""
        return self._apply(_polygon_mask(self.X, self.Y, vertices), True)

    def add_function(self, inside: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Domain:
        """Add all cells whose centres satisfy ``inside(X, Y)`` (vectorised predicate)."""
        return self._apply(np.asarray(inside(self.X, self.Y), dtype=bool), True)

    # ---- construction: difference -------------------------------------------
    def cut_rectangle(self, x0: float, y0: float, w: float, h: float) -> Domain:
        """Remove the axis-aligned rectangle with corner ``(x0, y0)``, width ``w``, height ``h``."""
        return self._apply(_rect_mask(self.X, self.Y, x0, y0, w, h), False)

    def cut_square(self, x0: float, y0: float, w: float) -> Domain:
        """Remove the axis-aligned square with corner ``(x0, y0)`` and side ``w``."""
        return self.cut_rectangle(x0, y0, w, w)

    def cut_disk(self, cx: float, cy: float, r: float) -> Domain:
        """Remove the disk with centre ``(cx, cy)`` and radius ``r`` (e.g. an obstacle)."""
        return self._apply(_disk_mask(self.X, self.Y, cx, cy, r), False)

    def cut_ellipse(self, cx: float, cy: float, a: float, b: float, angle: float = 0.0) -> Domain:
        """Remove an ellipse with semi-axes ``a``, ``b``, rotated by ``angle`` radians."""
        return self._apply(_ellipse_mask(self.X, self.Y, cx, cy, a, b, angle), False)

    def cut_polygon(self, vertices: Sequence[tuple[float, float]]) -> Domain:
        """Remove the polygon with the given vertices."""
        return self._apply(_polygon_mask(self.X, self.Y, vertices), False)

    def cut_function(self, inside: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Domain:
        """Remove all cells whose centres satisfy ``inside(X, Y)``."""
        return self._apply(np.asarray(inside(self.X, self.Y), dtype=bool), False)

    # ---- ready-made shapes ---------------------------------------------------
    @classmethod
    def square(cls, Nx: int, L: float = 1.0) -> Domain:
        """The full square ``[0, L]²``."""
        return cls(Nx, L, full=True)

    @classmethod
    def disk(cls, Nx: int, L: float = 1.0, margin: float = 0.0) -> Domain:
        """The disk inscribed in the bounding box, shrunk by ``margin``."""
        return cls(Nx, L, full=False).add_disk(L / 2, L / 2, L / 2 - margin)

    @classmethod
    def annulus(cls, Nx: int, L: float = 1.0, inner_fraction: float = 0.4) -> Domain:
        """Ring centred in the box; inner radius is ``inner_fraction`` of the outer one."""
        R = L / 2
        return cls(Nx, L, full=False).add_annulus(L / 2, L / 2, inner_fraction * R, R)

    @classmethod
    def l_shape(cls, Nx: int, L: float = 1.0, notch: float = 0.5) -> Domain:
        """Square with the upper-right ``notch × notch`` fraction removed."""
        return cls(Nx, L).cut_square(L * (1 - notch), L * (1 - notch), L * notch)

    @classmethod
    def polygon(cls, Nx: int, L: float, vertices: Sequence[tuple[float, float]]) -> Domain:
        """Domain given by a single polygon."""
        return cls(Nx, L, full=False).add_polygon(vertices)

    @classmethod
    def from_mask(cls, mask: np.ndarray, L: float = 1.0) -> Domain:
        """Wrap an existing boolean ``(Nx, Nx)`` mask (e.g. loaded from an image)."""
        m = np.asarray(mask, dtype=bool)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("mask must be a square 2-D array")
        dom = cls(m.shape[0], L, full=False)
        dom._mask = m.copy()
        return dom

    # ---- queries -------------------------------------------------------------
    @property
    def mask(self) -> np.ndarray:
        """Boolean ``(Nx, Nx)`` array, ``True`` inside the domain (a copy)."""
        return self._mask.copy()

    @property
    def n_cells(self) -> int:
        """Number of grid cells inside the domain."""
        return int(self._mask.sum())

    @property
    def area(self) -> float:
        """Area of the discretised domain."""
        return self.n_cells * self.dx**2

    @property
    def is_full(self) -> bool:
        """``True`` if the domain is the full bounding square."""
        return bool(self._mask.all())

    def n_components(self) -> int:
        """Number of 4-connected components (disconnected parts do not exchange mass)."""
        return int(ndimage.label(self._mask)[1])

    def validate(self) -> None:
        """Raise if the domain is empty; warn if it is disconnected.

        Raises
        ------
        ValueError
            If the domain contains no cells.
        """
        if self.n_cells == 0:
            raise ValueError("Domain is empty")
        k = self.n_components()
        if k > 1:
            warnings.warn(
                f"Domain has {k} disconnected components; they evolve independently.",
                stacklevel=2,
            )

    def __repr__(self) -> str:
        return f"Domain(Nx={self.Nx}, L={self.L}, cells={self.n_cells}, area={self.area:.4g})"


# ---------------------------------------------------------------------------
# LambdaField
# ---------------------------------------------------------------------------
class LambdaField(_Grid):
    """Builder for the spatially varying interaction rate λ(x, y).

    Regions are painted in the order they are added: a later region
    overwrites earlier values where it applies. All ``add_*`` methods return
    ``self`` so they can be chained.

    Parameters
    ----------
    Nx : int
        Number of grid cells per axis.
    L : float
        Side length of the bounding square ``[0, L]²``.

    Examples
    --------
    >>> lam = (LambdaField(64, 1.0)
    ...        .add_background(0.5)
    ...        .add_disk(2.0, 0.3, 0.3, 0.15)
    ...        .add_disk(5.0, 0.7, 0.7, 0.15)
    ...        .build())
    """

    def __init__(self, Nx: int, L: float = 1.0) -> None:
        super().__init__(Nx, L)
        self._field = np.zeros((self.Nx, self.Nx), dtype=float)

    def _paint(self, region: np.ndarray, lam: float) -> LambdaField:
        if lam < 0:
            raise ValueError("lam must be non-negative")
        self._field[region] = float(lam)
        return self

    def add_background(self, lam: float) -> LambdaField:
        """Set the whole field to ``lam``."""
        return self._paint(np.ones_like(self._field, dtype=bool), lam)

    def add_square(self, lam: float, x0: float, y0: float, w: float) -> LambdaField:
        """Set ``lam`` on the axis-aligned square with corner ``(x0, y0)`` and side ``w``."""
        return self._paint(_rect_mask(self.X, self.Y, x0, y0, w, w), lam)

    def add_rectangle(self, lam: float, x0: float, y0: float, w: float, h: float) -> LambdaField:
        """Set ``lam`` on the axis-aligned rectangle with corner ``(x0, y0)``."""
        return self._paint(_rect_mask(self.X, self.Y, x0, y0, w, h), lam)

    def add_disk(self, lam: float, cx: float, cy: float, r: float) -> LambdaField:
        """Set ``lam`` on the filled disk with centre ``(cx, cy)`` and radius ``r``."""
        return self._paint(_disk_mask(self.X, self.Y, cx, cy, r), lam)

    def add_annulus(
        self, lam: float, cx: float, cy: float, r_inner: float, r_outer: float
    ) -> LambdaField:
        """Set ``lam`` on the ring between ``r_inner`` and ``r_outer`` around ``(cx, cy)``."""
        return self._paint(_annulus_mask(self.X, self.Y, cx, cy, r_inner, r_outer), lam)

    def add_polygon(self, lam: float, vertices: Sequence[tuple[float, float]]) -> LambdaField:
        """Set ``lam`` inside the polygon with the given vertices."""
        return self._paint(_polygon_mask(self.X, self.Y, vertices), lam)

    def add_function(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> LambdaField:
        """Overwrite the field with ``func(X, Y)`` (smooth λ profiles, gradients, ...)."""
        values = np.broadcast_to(np.asarray(func(self.X, self.Y), dtype=float), self._field.shape)
        if np.any(values < 0):
            raise ValueError("lambda must be non-negative")
        self._field = values.copy()
        return self

    def build(self) -> np.ndarray:
        """Return the ``(Nx, Nx)`` λ array (a copy)."""
        return self._field.copy()
