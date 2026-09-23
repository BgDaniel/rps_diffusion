"""Videos and plots of RPS simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes

from .simulator import SPECIES, SimResult

__all__ = ["COLORS", "density_rgb", "make_video", "plot_fractions", "plot_snapshot"]

#: Line colours matching the RGB image (R = Scissors, G = Paper, B = Rock).
COLORS: dict[str, str] = {"S": "#d62728", "R": "#1f5fd6", "P": "#2ca02c"}
_OUTSIDE = np.array([0.93, 0.93, 0.93])


def density_rgb(rho: np.ndarray, contrast: float = 1.0) -> np.ndarray:
    """Map a density field to an RGB image (R = Scissors, G = Paper, B = Rock).

    Parameters
    ----------
    rho : np.ndarray
        Densities of shape ``(3, Nx, Nx)`` ordered ``(S, R, P)``; ``NaN``
        marks cells outside the domain (drawn light grey).
    contrast : float, default 1.0
        Deviations from ``1/3`` are multiplied by this factor before display,
        making small perturbations visible.

    Returns
    -------
    np.ndarray
        Image of shape ``(Nx, Nx, 3)`` with values in ``[0, 1]``.
    """
    s, r, p = rho
    img = np.stack([s, p, r], axis=-1)
    img = np.clip(1 / 3 + contrast * (img - 1 / 3), 0.0, 1.0)
    outside = np.isnan(img).any(axis=-1)
    img[outside] = _OUTSIDE
    return img


def _imshow_kw(L: float) -> dict:
    return {"origin": "lower", "extent": (0, L, 0, L), "interpolation": "nearest"}


def plot_snapshot(result: SimResult, index: int = -1, ax: Axes | None = None, contrast: float = 1.0) -> Axes:
    """Show one density snapshot as an RGB image.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    index : int, default -1
        Snapshot index.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created if omitted.
    contrast : float, default 1.0
        See :func:`density_rgb`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(density_rgb(result.snapshots[index], contrast), **_imshow_kw(result.L))
    ax.set_title(f"t = {result.t[index]:.2f}")
    return ax


def plot_fractions(result: SimResult, ax: Axes | None = None, show_omega0: bool = True) -> Axes:
    """Plot the domain-averaged fractions ``u_S(t)``, ``u_R(t)``, ``u_P(t)``.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created if omitted.
    show_omega0 : bool, default True
        Draw a horizontal bracket of length ``T₀ = 2π√3 / λ_mean``, the
        period of the linearised dynamics around ``(1/3, 1/3, 1/3)``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for k, name in enumerate(SPECIES):
        ax.plot(result.t, result.fractions[:, k], color=COLORS[name], lw=1.4, label=f"$u_{name}$")
    ax.axhline(1 / 3, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("t")
    ax.set_ylabel("mean fraction")
    ax.set_xlim(result.t[0], result.t[-1])

    lam = result.lambda_mean
    if show_omega0 and lam > 0:
        T0 = 2 * np.pi * np.sqrt(3) / lam
        lo, hi = np.nanmin(result.fractions), np.nanmax(result.fractions)
        pad = 0.1 * max(hi - lo, 1e-4)
        y = hi + pad
        t0 = result.t[0] + 0.05 * (result.t[-1] - result.t[0])
        ax.annotate("", xy=(t0, y), xytext=(t0 + T0, y),
                    arrowprops={"arrowstyle": "|-|", "color": "k", "lw": 1.0, "shrinkA": 0, "shrinkB": 0})
        ax.text(t0 + T0 / 2, y + 0.3 * pad,
                rf"$T_0 = 2\pi\sqrt{{3}}/\bar\lambda = {T0:.3g}$", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(lo - pad, hi + 3 * pad)
    ax.legend(loc="lower right", fontsize=9, ncol=3)
    return ax


def make_video(
    result: SimResult,
    path: str | Path,
    fps: int = 20,
    stride: int = 1,
    contrast: float = 1.0,
    dpi: int = 90,
) -> Path:
    """Render the simulation as an MP4 (GIF if ffmpeg is unavailable).

    Each frame has four panels: the RGB density image (R = Scissors,
    G = Paper, B = Rock), the static λ(x, y) field, the mean fractions up to
    the current time, and the spatial variance ``Var[ρ_i]`` over the domain.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    path : str or Path
        Output file. The suffix is replaced by ``.gif`` when falling back.
    fps : int, default 20
        Frames per second.
    stride : int, default 1
        Use every ``stride``-th snapshot as a frame.
    contrast : float, default 1.0
        Contrast of the density image (see :func:`density_rgb`).
    dpi : int, default 90
        Resolution of the rendered frames.

    Returns
    -------
    Path
        The file actually written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if animation.writers.is_available("ffmpeg"):
        writer: animation.AbstractMovieWriter = animation.FFMpegWriter(fps=fps, bitrate=2400)
        path = path.with_suffix(".mp4")
    else:
        writer = animation.PillowWriter(fps=fps)
        path = path.with_suffix(".gif")

    frames = list(range(0, len(result.t), max(1, stride)))
    t, fr, var = result.t, result.fractions, result.variances
    L = result.L

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
    (ax_img, ax_lam), (ax_frac, ax_var) = axes

    im = ax_img.imshow(density_rgb(result.snapshots[0], contrast), **_imshow_kw(L))
    ax_img.set_title("densities  (red = Scissors, green = Paper, blue = Rock)", fontsize=10)
    title = ax_img.text(0.02, 0.97, "", transform=ax_img.transAxes, va="top", fontsize=9,
                        bbox={"fc": "white", "alpha": 0.7, "ec": "none"})

    lam = np.ma.masked_where(~result.mask, result.lambda_field)
    lim = ax_lam.imshow(lam, cmap="viridis", **_imshow_kw(L))
    fig.colorbar(lim, ax=ax_lam, fraction=0.046, pad=0.04, label=r"$\lambda$")
    ax_lam.set_title(r"interaction rate $\lambda(x,y)$")

    lines_f, lines_v = [], []
    for k, name in enumerate(SPECIES):
        (lf,) = ax_frac.plot([], [], color=COLORS[name], lw=1.3, label=f"$u_{name}$")
        (lv,) = ax_var.plot([], [], color=COLORS[name], lw=1.3, label=f"Var $\\rho_{name}$")
        lines_f.append(lf)
        lines_v.append(lv)

    def _limits(ax: Axes, data: np.ndarray, log: bool = False) -> None:
        ax.set_xlim(t[0], t[-1] if t[-1] > t[0] else t[0] + 1)
        if log:
            pos = data[data > 0]
            if pos.size:
                ax.set_yscale("log")
                ax.set_ylim(pos.min() / 2, pos.max() * 2)
                return
        lo, hi = np.nanmin(data), np.nanmax(data)
        pad = 0.08 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)

    _limits(ax_frac, fr)
    ax_frac.axhline(1 / 3, color="0.6", lw=0.8, ls=":")
    ax_frac.set(xlabel="t", ylabel="mean fraction", title="domain-averaged fractions")
    ax_frac.legend(fontsize=8, loc="upper right", ncol=3)
    pos = var[var > 0]
    _limits(ax_var, var, log=bool(pos.size and pos.max() / pos.min() > 100))
    ax_var.set(xlabel="t", ylabel="spatial variance", title="spatial variance")
    ax_var.legend(fontsize=8, loc="upper right", ncol=3)
    fig.tight_layout()

    def update(i: int) -> list:
        im.set_data(density_rgb(result.snapshots[i], contrast))
        title.set_text(f"t = {t[i]:.2f}")
        for k in range(3):
            lines_f[k].set_data(t[: i + 1], fr[: i + 1, k])
            lines_v[k].set_data(t[: i + 1], var[: i + 1, k])
        return [im, title, *lines_f, *lines_v]

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(str(path), writer=writer, dpi=dpi)
    plt.close(fig)
    return path
