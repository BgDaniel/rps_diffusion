"""Videos and plots of RPS simulations."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

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


def _ffmpeg_path() -> str | None:
    configured = mpl.rcParams.get("animation.ffmpeg_path", "ffmpeg")
    return shutil.which(configured) or shutil.which("ffmpeg")


def _render_frames(result: SimResult, frames: list[int], contrast: float, dpi: int) -> Iterator[np.ndarray]:
    """Yield RGB frames of the 2×2 panel, redrawing only the animated artists."""
    t, fr, var, L = result.t, result.fractions, result.variances, result.L
    fig = Figure(figsize=(10, 8.4), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    (ax_img, ax_lam), (ax_frac, ax_var) = fig.subplots(2, 2)

    im = ax_img.imshow(density_rgb(result.snapshots[frames[0]], contrast), animated=True, **_imshow_kw(L))
    ax_img.set_title("densities  (red = Scissors, green = Paper, blue = Rock)", fontsize=10)
    stamp = ax_img.text(0.02, 0.97, "", transform=ax_img.transAxes, va="top", fontsize=9, animated=True,
                        bbox={"fc": "white", "alpha": 0.7, "ec": "none"})

    lam = np.ma.masked_where(~result.mask, result.lambda_field)
    lim = ax_lam.imshow(lam, cmap="viridis", **_imshow_kw(L))
    fig.colorbar(lim, ax=ax_lam, fraction=0.046, pad=0.04, label=r"$\lambda$")
    ax_lam.set_title(r"interaction rate $\lambda(x,y)$")

    t_end = t[-1] if t[-1] > t[0] else t[0] + 1.0
    lines_f, lines_v = [], []
    for k, name in enumerate(SPECIES):
        (lf,) = ax_frac.plot([], [], color=COLORS[name], lw=1.3, label=f"$u_{name}$", animated=True)
        (lv,) = ax_var.plot([], [], color=COLORS[name], lw=1.3, label=rf"Var $\rho_{name}$", animated=True)
        lines_f.append(lf)
        lines_v.append(lv)

    lo, hi = float(np.nanmin(fr)), float(np.nanmax(fr))
    pad = 0.08 * max(hi - lo, 1e-6)
    ax_frac.set(xlim=(t[0], t_end), ylim=(lo - pad, hi + pad), xlabel="t", ylabel="mean fraction",
                title="domain-averaged fractions")
    ax_frac.axhline(1 / 3, color="0.6", lw=0.8, ls=":")
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
    fig.tight_layout()

    canvas.draw()  # static parts only: animated artists are skipped
    background = canvas.copy_from_bbox(fig.bbox)
    for i in frames:
        canvas.restore_region(background)
        im.set_data(density_rgb(result.snapshots[i], contrast))
        stamp.set_text(f"t = {t[i]:.2f}")
        for k in range(3):
            lines_f[k].set_data(t[: i + 1], fr[: i + 1, k])
            lines_v[k].set_data(t[: i + 1], var[: i + 1, k])
        ax_img.draw_artist(im)
        ax_img.draw_artist(stamp)
        for line in lines_f:
            ax_frac.draw_artist(line)
        for line in lines_v:
            ax_var.draw_artist(line)
        yield np.asarray(canvas.buffer_rgba())[..., :3].copy()


def make_video(
    result: SimResult,
    path: str | Path,
    fps: int = 20,
    stride: int = 1,
    contrast: float = 1.0,
    dpi: int = 80,
    max_frames: int | None = 300,
) -> Path:
    """Render the simulation as an MP4 (GIF if ffmpeg is unavailable).

    Each frame has four panels: the RGB density image (R = Scissors,
    G = Paper, B = Rock), the static λ(x, y) field, the mean fractions up to
    the current time, and the spatial variance ``Var[ρ_i]`` over the domain.
    The static parts of the figure are drawn once; each frame only redraws
    the changing artists (blitting) on an off-screen Agg canvas.

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
        Contrast of the density image (see :func:`density_rgb`).
    dpi : int, default 80
        Resolution of the rendered frames (the figure is 10 × 8.4 inches).
    max_frames : int, optional
        Increase ``stride`` so that at most this many frames are rendered.
        ``None`` renders every ``stride``-th snapshot.

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
    rendered = _render_frames(result, frames, contrast, dpi)

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
