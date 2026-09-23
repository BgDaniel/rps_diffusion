"""Shared helpers for the example scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from rps_diffusion.analysis import dominant_frequencies, frequency_spectrum  # noqa: E402
from rps_diffusion.simulator import SimResult  # noqa: E402
from rps_diffusion.visualize import make_video, plot_fractions  # noqa: E402

OUTPUT_DIR = Path("examples") / "output"


def save_outputs(
    result: SimResult,
    name: str,
    out_dir: Path = OUTPUT_DIR,
    stride: int = 1,
    contrast: float = 1.0,
    spectrum: bool = True,
) -> None:
    """Write the video, the fraction plot and (optionally) the spectrum.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    name : str
        File name stem.
    out_dir : Path
        Output directory.
    stride : int
        Snapshot stride for the video.
    contrast : float
        Contrast of the density image.
    spectrum : bool
        Also plot the power spectrum next to the fractions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = 2 if spectrum else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4), squeeze=False)
    plot_fractions(result, ax=axes[0, 0])
    axes[0, 0].set_title("domain-averaged fractions")
    if spectrum:
        frequency_spectrum(result, 0, ax=axes[0, 1])
        axes[0, 1].set_title("power spectrum of $u_S$")
    fig.tight_layout()
    plot_path = out_dir / f"{name}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    video_path = make_video(result, out_dir / f"{name}.mp4", stride=stride, contrast=contrast)
    print(f"  plot  -> {plot_path}\n  video -> {video_path}")


def report_frequencies(result: SimResult, expected: dict[str, float], n: int = 3) -> None:
    """Print the measured spectral peaks next to the predicted frequencies.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    expected : dict of str to float
        Label -> predicted frequency (cycles per unit time).
    n : int
        Number of peaks to report.
    """
    found = dominant_frequencies(result, n=n)
    print("  measured peaks f   :", ", ".join(f"{f:.4f}" for f in found))
    for label, f in expected.items():
        print(f"  predicted {label:<9}: {f:.4f}")
