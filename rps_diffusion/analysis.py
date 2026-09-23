"""Frequency analysis of the domain-averaged species fractions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.signal import find_peaks

from .simulator import SPECIES, SimResult

__all__ = ["frequency_spectrum", "dominant_frequencies", "plot_spectrum", "winding_number"]


def _sample_spacing(result: SimResult) -> float:
    dts = np.diff(result.t)
    if len(dts) < 3:
        raise ValueError("need at least four saved time points for a spectrum")
    if not np.allclose(dts, dts[0], rtol=1e-6):
        raise ValueError("saved times are not uniformly spaced")
    return float(dts[0])


def frequency_spectrum(
    result: SimResult,
    species: int = 0,
    ax: Axes | None = None,
    discard: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum of a mean-fraction time series.

    The series is de-meaned and Hann-windowed before the FFT. Frequencies are
    ordinary frequencies (cycles per unit time); the linearised prediction is
    ``f₀ = ω₀ / 2π`` with ``ω₀ = √(λ_S λ_R λ_P / Σλ)``.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    species : int, default 0
        0 = S, 1 = R, 2 = P.
    ax : matplotlib.axes.Axes, optional
        If given, the spectrum is plotted there with ``f₀``, ``2f₀`` and
        ``3f₀`` marked by vertical lines.
    discard : float, default 0.0
        Initial time span to drop (transients) before the FFT.

    Returns
    -------
    freqs : np.ndarray
        Non-negative frequencies.
    power : np.ndarray
        Power ``|FFT|²`` at each frequency.
    """
    h = _sample_spacing(result)
    keep = result.t >= result.t[0] + discard
    x = result.fractions[keep, species]
    x = (x - x.mean()) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=h)
    if ax is not None:
        _draw_spectrum(ax, freqs, power, result, SPECIES[species])
    return freqs, power


def _draw_spectrum(ax: Axes, freqs: np.ndarray, power: np.ndarray, result: SimResult, label: str) -> None:
    ax.semilogy(freqs[1:], power[1:] + 1e-300, lw=1.2, label=f"species {label}")
    f0 = result.omega0 / (2 * np.pi)
    if f0 > 0:
        for k, style in zip((1, 2, 3), ("-", "--", ":")):
            if k * f0 <= freqs[-1]:
                ax.axvline(k * f0, color="0.3", ls=style, lw=0.9,
                           label=r"$f_0=\omega_0/2\pi$" if k == 1 else f"{k}$f_0$")
    ax.set_xlabel("frequency  [1 / time]")
    ax.set_ylabel("power")
    # Show 6 f0 or 1.5x the strongest peaks, whichever is wider (nonlinear runs peak elsewhere).
    peaks, _ = find_peaks(power[1:])
    top = freqs[peaks[np.argsort(power[peaks + 1])[-3:]] + 1] if peaks.size else np.array([0.0])
    ax.set_xlim(0, min(freqs[-1], max(6 * f0, 1.5 * top.max())))
    ax.set_ylim(power[1:].max() * 1e-8, power[1:].max() * 5)
    ax.legend(fontsize=8)


def plot_spectrum(result: SimResult, species: int = 0, discard: float = 0.0) -> plt.Figure:
    """Create a figure with the power spectrum and marked ``f₀`` overtones.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    species : int, default 0
        0 = S, 1 = R, 2 = P.
    discard : float, default 0.0
        Initial time span to drop before the FFT.

    Returns
    -------
    matplotlib.figure.Figure
        The new figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    frequency_spectrum(result, species, ax=ax, discard=discard)
    ax.set_title("Spectrum of the mean fraction")
    fig.tight_layout()
    return fig


def dominant_frequencies(
    result: SimResult,
    n: int = 3,
    species: int | None = None,
    discard: float = 0.0,
) -> list[float]:
    """Frequencies of the ``n`` strongest spectral peaks (DC excluded).

    Peak positions are refined by parabolic interpolation of the log-power
    around each local maximum.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    n : int, default 3
        Number of peaks to return.
    species : int, optional
        Species to analyse; ``None`` sums the spectra of all three.
    discard : float, default 0.0
        Initial time span to drop before the FFT.

    Returns
    -------
    list of float
        Peak frequencies sorted by decreasing power (fewer if fewer peaks exist).
    """
    idx = range(3) if species is None else (species,)
    freqs, power = frequency_spectrum(result, 0, discard=discard)
    power = sum(frequency_spectrum(result, k, discard=discard)[1] for k in idx)
    peaks, _ = find_peaks(power[1:])
    peaks = peaks + 1
    peaks = peaks[np.argsort(power[peaks])[::-1]][:n]
    df = freqs[1] - freqs[0]
    out: list[float] = []
    for p in peaks:
        if 0 < p < len(power) - 1:
            a, b, c = np.log(power[p - 1 : p + 2] + 1e-300)
            denom = a - 2 * b + c
            shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
            out.append(float(freqs[p] + shift * df))
        else:
            out.append(float(freqs[p]))
    return out


def winding_number(
    result: SimResult,
    index: int,
    centre: tuple[float, float],
    radius: float,
    n_points: int = 720,
) -> float:
    """Winding number of the reaction phase along a circle.

    Each cell's deviation from the fixed point, ``ρ − ρ*``, is mapped to the
    complex number ``z = d_S + d_R e^{2πi/3} + d_P e^{4πi/3}``. Its argument
    is the phase of the local S → R → P cycle. The function returns how many
    times this phase winds around 0 along the circle.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    index : int
        Snapshot index.
    centre : tuple of float
        Circle centre in absolute coordinates.
    radius : float
        Circle radius; the whole circle should lie inside the domain.
    n_points : int, default 720
        Sampling points along the circle.

    Returns
    -------
    float
        The winding number (close to an integer when the phase is well
        defined everywhere on the circle).
    """
    th = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    x = centre[0] + radius * np.cos(th)
    y = centre[1] + radius * np.sin(th)
    dx = result.L / result.Nx
    i = np.clip((x / dx).astype(int), 0, result.Nx - 1)
    j = np.clip((y / dx).astype(int), 0, result.Nx - 1)
    d = result.snapshots[index][:, j, i] - np.asarray(result.fixed_point)[:, None]
    if np.isnan(d).any():
        raise ValueError("the circle leaves the domain")
    z = d[0] + d[1] * np.exp(2j * np.pi / 3) + d[2] * np.exp(4j * np.pi / 3)
    dphi = np.angle(np.roll(z, -1) / z)  # phase increments in (-π, π]
    return float(dphi.sum() / (2 * np.pi))
