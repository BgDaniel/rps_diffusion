"""Videos and plots of RPS simulations."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from .domain import grid
from .simulator import SPECIES, SimResult

__all__ = [
    "COLORS",
    "density_rgb",
    "describe_parameters",
    "make_video",
    "plot_fractions",
    "plot_snapshot",
    "plot_surfaces",
]

#: Line colours matching the RGB image (R = Scissors, G = Paper, B = Rock).
COLORS: dict[str, str] = {"S": "#d62728", "R": "#1f5fd6", "P": "#2ca02c"}
_NAMES: dict[str, str] = {"S": "Scissors", "R": "Rock", "P": "Paper"}
_CMAPS: dict[str, str] = {"S": "Reds", "R": "Blues", "P": "Greens"}
_OUTSIDE = np.array([0.93, 0.93, 0.93])
_SURFACE_CELLS = 48  # surfaces are drawn with at most this many cells per axis


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


# ---------------------------------------------------------------------------
# 3-D surfaces
# ---------------------------------------------------------------------------
def _surface_grid(result: SimResult) -> tuple[slice, np.ndarray, np.ndarray]:
    step = max(1, int(np.ceil(result.Nx / _SURFACE_CELLS)))
    sl = slice(None, None, step)
    X, Y = grid(result.Nx, result.L)
    return sl, X[sl, sl], Y[sl, sl]


def _surface_zlim(result: SimResult, frames: list[int]) -> tuple[float, float]:
    lo = min(float(np.nanmin(result.snapshots[i])) for i in frames)
    hi = max(float(np.nanmax(result.snapshots[i])) for i in frames)
    pad = 0.05 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _setup_surface_axes(ax: Axes, name: str, L: float, zlim: tuple[float, float]) -> None:
    ax.set(xlim=(0, L), ylim=(0, L), zlim=zlim)
    ax.set_xlabel("x", fontsize=8, labelpad=-6)
    ax.set_ylabel("y", fontsize=8, labelpad=-6)
    ax.set_title(rf"$\rho_{name}$  ({_NAMES[name]})", fontsize=11, color=COLORS[name])
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=32, azim=-58)


def _draw_surface(ax: Axes, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, name: str,
                  zlim: tuple[float, float], animated: bool = False):
    span = zlim[1] - zlim[0]
    # Masked (outside-domain) cells produce no polygons.
    return ax.plot_surface(X, Y, np.ma.masked_invalid(Z), cmap=_CMAPS[name], vmin=zlim[0] - 0.3 * span,
                           vmax=zlim[1], linewidth=0, antialiased=False, rstride=1, cstride=1,
                           animated=animated)


def plot_surfaces(
    result: SimResult,
    index: int = -1,
    fig: Figure | None = None,
    axes: list[Axes] | None = None,
) -> Figure:
    """Show one snapshot as three 3-D surface plots, one per species.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    index : int, default -1
        Snapshot index.
    fig : matplotlib.figure.Figure, optional
        Target figure; a new one is created if omitted.
    axes : list of Axes, optional
        Three existing 3-D axes to draw into (``projection='3d'``). If
        omitted, a row of three is added to ``fig``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn into.
    """
    if axes is None:
        if fig is None:
            fig = plt.figure(figsize=(15, 4.8))
        axes = [fig.add_subplot(1, 3, k + 1, projection="3d") for k in range(3)]
        fig.suptitle(f"t = {result.t[index]:.2f}    ·    {describe_parameters(result)}", fontsize=10)
    fig = axes[0].figure
    i = index % len(result.t)
    sl, X, Y = _surface_grid(result)
    zlim = _surface_zlim(result, [i])
    for ax, k, name in zip(axes, range(3), SPECIES):
        _setup_surface_axes(ax, name, result.L, zlim)
        _draw_surface(ax, X, Y, result.snapshots[i][k][sl, sl], name, zlim)
    return fig


def describe_parameters(result: SimResult) -> str:
    """One-line summary of the model parameters, for figure titles.

    Parameters
    ----------
    result : SimResult
        Simulation output.

    Returns
    -------
    str
        Noise σ, the three interaction rates, the fixed point ρ*, ω₀ and the
        grid, in matplotlib mathtext.
    """
    lS, lR, lP = result.rates
    rs = result.fixed_point
    dom = "full square" if result.mask.all() else f"shaped domain ({result.mask.mean():.0%} of box)"
    return (rf"$\sigma$ = {result.sigma:g}   ·   $\lambda_S$ = {lS:g},  $\lambda_R$ = {lR:g},  "
            rf"$\lambda_P$ = {lP:g}   ·   $\rho^*$ = ({rs[0]:.3f}, {rs[1]:.3f}, {rs[2]:.3f}),  "
            rf"$\omega_0$ = {result.omega0:.3g}   ·   "
            f"L = {result.L:g}, Nx = {result.Nx}, dt = {result.dt:g}, {dom}")


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def plot_fractions(result: SimResult, ax: Axes | None = None, show_omega0: bool = True) -> Axes:
    """Plot the domain-averaged fractions ``u_S(t)``, ``u_R(t)``, ``u_P(t)``.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created if omitted.
    show_omega0 : bool, default True
        Draw a horizontal bracket of length ``T₀ = 2π / ω₀`` with
        ``ω₀ = √(λ_S λ_R λ_P / Σλ)``, the period of the linearised dynamics
        around the fixed point ``ρ*`` (dotted lines).

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for k, name in enumerate(SPECIES):
        ax.plot(result.t, result.fractions[:, k], color=COLORS[name], lw=1.4, label=f"$u_{name}$")
    for k, name in enumerate(SPECIES):
        ax.axhline(result.fixed_point[k], color=COLORS[name], lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("t")
    ax.set_ylabel("mean fraction")
    ax.set_xlim(result.t[0], result.t[-1])

    w0 = result.omega0
    if show_omega0 and w0 > 0:
        T0 = 2 * np.pi / w0
        lo, hi = np.nanmin(result.fractions), np.nanmax(result.fractions)
        pad = 0.1 * max(hi - lo, 1e-4)
        y = hi + pad
        t0 = result.t[0] + 0.05 * (result.t[-1] - result.t[0])
        ax.annotate("", xy=(t0, y), xytext=(t0 + T0, y),
                    arrowprops={"arrowstyle": "|-|", "color": "k", "lw": 1.0, "shrinkA": 0, "shrinkB": 0})
        ax.text(t0 + T0 / 2, y + 0.3 * pad,
                rf"$T_0 = 2\pi/\omega_0 = {T0:.3g}$", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(lo - pad, hi + 3 * pad)
    ax.legend(loc="lower right", fontsize=9, ncol=3)
    return ax


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def _ffmpeg_path() -> str | None:
    configured = mpl.rcParams.get("animation.ffmpeg_path", "ffmpeg")
    return shutil.which(configured) or shutil.which("ffmpeg")


def _setup_series_axes(ax_frac: Axes, ax_var: Axes, result: SimResult) -> tuple[list, list]:
    t, fr, var = result.t, result.fractions, result.variances
    t_end = t[-1] if t[-1] > t[0] else t[0] + 1.0
    lines_f, lines_v = [], []
    for name in SPECIES:
        (lf,) = ax_frac.plot([], [], color=COLORS[name], lw=1.3, label=f"$u_{name}$", animated=True)
        (lv,) = ax_var.plot([], [], color=COLORS[name], lw=1.3, label=rf"Var $\rho_{name}$", animated=True)
        lines_f.append(lf)
        lines_v.append(lv)

    lo, hi = float(np.nanmin(fr)), float(np.nanmax(fr))
    pad = 0.08 * max(hi - lo, 1e-6)
    ax_frac.set(xlim=(t[0], t_end), ylim=(lo - pad, hi + pad), xlabel="t", ylabel="mean fraction",
                title="domain-averaged fractions")
    for k, name in enumerate(SPECIES):
        ax_frac.axhline(result.fixed_point[k], color=COLORS[name], lw=0.8, ls=":", alpha=0.7)
    ax_frac.legend(fontsize=8, loc="upper right", ncol=3)

    pos = var[var > 0]
    if pos.size and pos.max() / pos.min() > 100:
        ax_var.set_yscale("log")
        ax_var.set_ylim(pos.min() / 2, pos.max() * 2)
    else:
        vlo, vhi = float(np.nanmin(var)), float(np.nanmax(var))
        vpad = 0.08 * max(vhi - vlo, 1e-12)
        ax_var.set_ylim(vlo - vpad, vhi + vpad)
    ax_var.set(xlim=(t[0], t_end), xlabel="t", ylabel="spatial variance", title="spatial variance")
    ax_var.legend(fontsize=8, loc="upper right", ncol=3)
    return lines_f, lines_v


def _render_frames(
    result: SimResult, frames: list[int], contrast: float, dpi: int, style: str
) -> Iterator[np.ndarray]:
    """Yield RGB video frames, redrawing only the animated artists (blitting)."""
    t, fr, var, L = result.t, result.fractions, result.variances, result.L

    if style == "surface":
        fig = Figure(figsize=(15, 9), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        gs = fig.add_gridspec(2, 6, height_ratios=(1.25, 1))
        ax_surf = [fig.add_subplot(gs[0, 2 * k: 2 * k + 2], projection="3d") for k in range(3)]
        ax_frac, ax_var = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])
        sl, X, Y = _surface_grid(result)
        zlim = _surface_zlim(result, frames)
        for ax, name in zip(ax_surf, SPECIES):
            _setup_surface_axes(ax, name, L, zlim)
        stamp = fig.text(0.5, 0.99, "", ha="center", va="top", fontsize=13, animated=True)
        fig.text(0.5, 0.955, describe_parameters(result), ha="center", va="top", fontsize=10, color="0.25")
    elif style == "rgb":
        fig = Figure(figsize=(16, 5.6), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax_img, ax_frac, ax_var = fig.subplots(1, 3)
        im = ax_img.imshow(density_rgb(result.snapshots[frames[0]], contrast), animated=True, **_imshow_kw(L))
        ax_img.set_title("densities  (red = Scissors, green = Paper, blue = Rock)", fontsize=10)
        stamp = ax_img.text(0.02, 0.97, "", transform=ax_img.transAxes, va="top", fontsize=9, animated=True,
                            bbox={"fc": "white", "alpha": 0.7, "ec": "none"})
        fig.suptitle(describe_parameters(result), fontsize=10, color="0.25")
    else:
        raise ValueError("style must be 'surface' or 'rgb'")

    lines_f, lines_v = _setup_series_axes(ax_frac, ax_var, result)
    if style == "surface":
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.89, wspace=0.6, hspace=0.15)
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    canvas.draw()  # static parts only: animated artists are skipped
    background = canvas.copy_from_bbox(fig.bbox)
    surfaces: list = []
    for i in frames:
        canvas.restore_region(background)
        stamp.set_text(f"t = {t[i]:.2f}")
        if style == "surface":
            for surf in surfaces:
                surf.remove()
            surfaces = []
            for ax, k, name in zip(ax_surf, range(3), SPECIES):
                surf = _draw_surface(ax, X, Y, result.snapshots[i][k][sl, sl], name, zlim, animated=True)
                surf.do_3d_projection()  # uses the view matrix from the initial full draw
                ax.draw_artist(surf)
                surfaces.append(surf)
            fig.draw_artist(stamp)
        else:
            im.set_data(density_rgb(result.snapshots[i], contrast))
            ax_img.draw_artist(im)
            ax_img.draw_artist(stamp)
        for k in range(3):
            lines_f[k].set_data(t[: i + 1], fr[: i + 1, k])
            lines_v[k].set_data(t[: i + 1], var[: i + 1, k])
            ax_frac.draw_artist(lines_f[k])
            ax_var.draw_artist(lines_v[k])
        yield np.asarray(canvas.buffer_rgba())[..., :3].copy()


def make_video(
    result: SimResult,
    path: str | Path,
    fps: int = 20,
    stride: int = 1,
    contrast: float = 1.0,
    dpi: int = 80,
    max_frames: int | None = 300,
    style: Literal["surface", "rgb"] = "surface",
) -> Path:
    """Render the simulation as an MP4 (GIF if ffmpeg is unavailable).

    With ``style='surface'`` (default) each frame shows three 3-D surface
    plots of ``ρ_S``, ``ρ_R`` and ``ρ_P`` on a common z-scale, above the
    mean fractions up to the current time and the spatial variance
    ``Var[ρ_i]`` over the domain. With ``style='rgb'`` the three surfaces
    are replaced by one RGB density image (R = Scissors, G = Paper,
    B = Rock). The model parameters are printed at the top of every frame. Static parts of the figure are
    drawn once; each frame only redraws the changing artists (blitting) on
    an off-screen Agg canvas.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    path : str or Path
        Output file. The suffix is set to ``.mp4`` or ``.gif``.
    fps : int, default 20
        Frames per second.
    stride : int, default 1
        Use every ``stride``-th snapshot as a frame.
    contrast : float, default 1.0
        Contrast of the RGB density image (``style='rgb'`` only; see
        :func:`density_rgb`).
    dpi : int, default 80
        Resolution of the rendered frames.
    max_frames : int, optional
        Increase ``stride`` so that at most this many frames are rendered.
        ``None`` renders every ``stride``-th snapshot.
    style : {'surface', 'rgb'}, default 'surface'
        How the densities are shown.

    Returns
    -------
    Path
        The file actually written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(result.t)
    stride = max(1, stride)
    if max_frames is not None and n > max_frames * stride:
        stride = int(np.ceil(n / max_frames))
    frames = list(range(0, n, stride))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    rendered = _render_frames(result, frames, contrast, dpi, style)

    ffmpeg = _ffmpeg_path()
    if ffmpeg is not None:
        path = path.with_suffix(".mp4")
        first = next(rendered)
        h, w = first.shape[:2]
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path),
        ]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
            assert proc.stdin is not None
            proc.stdin.write(first.tobytes())
            for frame in rendered:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")
        return path

    # GIF fallback: quantise all frames against one shared palette, which is much
    # faster than per-frame median-cut and avoids palette flicker.
    path = path.with_suffix(".gif")
    arrays = list(rendered)
    sample = np.concatenate([arrays[0], arrays[len(arrays) // 2], arrays[-1]], axis=0)
    palette = Image.fromarray(sample).quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    images = [Image.fromarray(a).quantize(palette=palette, dither=Image.Dither.NONE) for a in arrays]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0,
                   optimize=False)  # skips per-frame diffing, ~5x faster encoding
    return path
