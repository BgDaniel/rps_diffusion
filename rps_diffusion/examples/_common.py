"""Shared helpers for the example scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rps_diffusion.analysis import dominant_frequencies, frequency_spectrum  # noqa: E402
from rps_diffusion.simulator import SimResult  # noqa: E402
from rps_diffusion.visualize import (  # noqa: E402
    describe_parameters,
    make_video,
    plot_fractions,
    plot_surfaces,
)

OUTPUT_DIR = Path("examples") / "output"


def save_outputs(result: SimResult, name: str, title: str, out_dir: Path = OUTPUT_DIR) -> None:
    """Write the summary figure and the video for one scenario.

    Every scenario uses the same summary layout. The top row shows the mean
    fractions (with the fixed point ρ* and the T₀ bracket) and the power
    spectrum with f₀ and its overtones. The bottom row shows the three
    density surfaces at the final time. The title carries all model
    parameters (σ, the three rates, ρ*, ω₀, grid).

    Parameters
    ----------
    result : SimResult
        Simulation output.
    name : str
        File name stem.
    title : str
        Scenario title shown above the parameter line.
    out_dir : Path
        Output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 6, height_ratios=(1, 1.15))
    ax_frac = fig.add_subplot(gs[0, 0:3])
    ax_spec = fig.add_subplot(gs[0, 3:6])
    ax_surf = [fig.add_subplot(gs[1, 2 * k: 2 * k + 2], projection="3d") for k in range(3)]

    plot_fractions(result, ax=ax_frac)
    ax_frac.set_title("domain-averaged fractions")
    frequency_spectrum(result, 0, ax=ax_spec)
    ax_spec.set_title("power spectrum of $u_S$")
    plot_surfaces(result, -1, axes=ax_surf)
    ax_surf[1].text2D(0.5, 1.12, f"densities at final time t = {result.t[-1]:g}",
                      transform=ax_surf[1].transAxes, ha="center", fontsize=11)

    fig.suptitle(f"{title}\n{describe_parameters(result)}", fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.04, top=0.9, wspace=0.6, hspace=0.25)
    plot_path = out_dir / f"{name}.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    video_path = make_video(result, out_dir / f"{name}.mp4", max_frames=150)
    print(f"  plot  -> {plot_path}\n  video -> {video_path}")


def report_frequencies(result: SimResult, expected: dict[str, float] | None = None, n: int = 3) -> None:
    """Print the measured spectral peaks next to the predicted frequencies.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    expected : dict of str to float, optional
        Label -> predicted frequency (cycles per unit time). Defaults to
        ``f0 = ω₀ / 2π`` with ``ω₀ = √(λ_S λ_R λ_P / Σλ)``.
    n : int
        Number of peaks to report.
    """
    if expected is None:
        expected = {"f0": result.omega0 / (2 * np.pi)}
    found = dominant_frequencies(result, n=n)
    print("  measured peaks f   :", ", ".join(f"{f:.4f}" for f in found))
    for label, f in expected.items():
        print(f"  predicted {label:<9}: {f:.4f}")
