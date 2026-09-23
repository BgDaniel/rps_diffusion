"""Scenario framework shared by all example groups."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rps_diffusion.analysis import dominant_frequencies, frequency_spectrum, winding_number  # noqa: E402
from rps_diffusion.domain import Domain, grid  # noqa: E402
from rps_diffusion.simulator import RPSSimulator, SimResult  # noqa: E402
from rps_diffusion.visualize import (  # noqa: E402
    _surface_zlim,
    describe_parameters,
    make_video,
    plot_fractions,
    plot_surfaces,
)

OUTPUT_DIR = Path("examples") / "output"

#: Initial condition factory: ``(Nx, L) -> (3, Nx, Nx)`` array.
ICFactory = Callable[[int, float], np.ndarray]
#: Domain factory: ``(Nx, L) -> Domain``.
DomainFactory = Callable[[int, float], Domain]


@dataclass(frozen=True)
class Scenario:
    """Declarative description of one example run.

    Attributes
    ----------
    name : str
        File name stem (unique within its group).
    title : str
        One-line title (matplotlib mathtext allowed).
    description : str
        What to look for; shown in the gallery.
    rates : float or tuple of float
        Interaction rates ``(λ_S, λ_R, λ_P)`` or one common rate.
    sigma : float
        Noise amplitude σ.
    initial : ICFactory
        Initial condition ``(Nx, L) -> rho0``.
    t_max : float
        Simulated time.
    domain : DomainFactory, optional
        Domain shape; ``None`` is the full square.
    Nx : int
        Grid cells per axis.
    L : float
        Side length of the bounding box.
    dt : float
        Time step.
    save_every : int
        Steps between snapshots.
    loops : list of (label, centre, radius)
        Circles (in units of ``L``) along which the winding number of the
        reaction phase is measured at the start and the end.
    """

    name: str
    title: str
    description: str
    rates: float | tuple[float, float, float]
    sigma: float
    initial: ICFactory
    t_max: float
    domain: DomainFactory | None = None
    Nx: int = 96
    L: float = 1.0
    dt: float = 0.01
    save_every: int = 10
    loops: tuple[tuple[str, tuple[float, float], float], ...] = ()


@dataclass
class Group:
    """A topological family of scenarios.

    Attributes
    ----------
    key : str
        Module / directory name.
    title : str
        Human-readable title.
    description : str
        What the scenarios have in common.
    scenarios : list of Scenario
        The runs in this group.
    """

    key: str
    title: str
    description: str
    scenarios: list[Scenario] = field(default_factory=list)


def simulate(scenario: Scenario, progress: bool = True) -> SimResult:
    """Run the simulation described by ``scenario``.

    Parameters
    ----------
    scenario : Scenario
        The run to perform.
    progress : bool, default True
        Show a progress bar.

    Returns
    -------
    SimResult
        Simulation output.
    """
    dom = scenario.domain(scenario.Nx, scenario.L) if scenario.domain else None
    sim = RPSSimulator(scenario.rates, sigma=scenario.sigma, L=scenario.L, dt=scenario.dt,
                       Nx=scenario.Nx, domain=dom)
    return sim.run(scenario.initial(scenario.Nx, scenario.L), scenario.t_max,
                   save_every=scenario.save_every, progress=progress)


def save_summary(result: SimResult, path: Path, title: str) -> None:
    """Write the summary figure: fractions + spectrum, initial and final surfaces.

    Parameters
    ----------
    result : SimResult
        Simulation output.
    path : Path
        PNG file to write.
    title : str
        Scenario title shown above the parameter line.
    """
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(3, 6, height_ratios=(1, 1.1, 1.1))
    ax_frac = fig.add_subplot(gs[0, 0:3])
    ax_spec = fig.add_subplot(gs[0, 3:6])
    rows = [[fig.add_subplot(gs[r, 2 * k: 2 * k + 2], projection="3d") for k in range(3)] for r in (1, 2)]

    plot_fractions(result, ax=ax_frac)
    ax_frac.set_title("domain-averaged fractions")
    frequency_spectrum(result, 0, ax=ax_spec)
    ax_spec.set_title("power spectrum of $u_S$")

    zlim = _surface_zlim(result, [0, len(result.t) - 1])
    for axes, index, label in ((rows[0], 0, "initial distributions"), (rows[1], -1, "final distributions")):
        plot_surfaces(result, index, axes=axes, zlim=zlim)
        axes[1].text2D(0.5, 1.13, f"{label},  t = {result.t[index]:g}", transform=axes[1].transAxes,
                       ha="center", fontsize=12, fontweight="bold")

    fig.suptitle(f"{title}\n{describe_parameters(result)}", fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.03, top=0.92, wspace=0.6, hspace=0.25)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run_scenario(group: Group, scenario: Scenario, out_dir: Path = OUTPUT_DIR) -> dict:
    """Simulate one scenario and write its summary figure and video.

    Parameters
    ----------
    group : Group
        The group the scenario belongs to (sets the output sub-directory).
    scenario : Scenario
        The run to perform.
    out_dir : Path
        Root output directory.

    Returns
    -------
    dict
        Paths and measured quantities, used for the gallery.
    """
    print(f"[{group.key}] {scenario.name}")
    res = simulate(scenario)
    target = out_dir / group.key
    target.mkdir(parents=True, exist_ok=True)
    png = target / f"{scenario.name}.png"
    save_summary(res, png, scenario.title)
    video = make_video(res, target / f"{scenario.name}.mp4", max_frames=120, dpi=65)

    f0 = res.omega0 / (2 * np.pi)
    peaks = dominant_frequencies(res, n=3)
    late = res.t >= res.t[-1] / 2
    info = {
        "png": png,
        "video": video,
        "f0": f0,
        "peaks": peaks,
        "mean_late": res.fractions[late].mean(axis=0),
        "range": (float(res.fractions.min()), float(res.fractions.max())),
        "fixed_point": res.fixed_point,
    }
    windings = [
        (label, winding_number(res, 0, (c[0] * res.L, c[1] * res.L), r * res.L),
         winding_number(res, -1, (c[0] * res.L, c[1] * res.L), r * res.L))
        for label, c, r in scenario.loops
    ]
    info["windings"] = windings
    print(f"  f0 predicted {f0:.4f} | measured peaks " + ", ".join(f"{p:.4f}" for p in peaks))
    for label, w0, w1 in windings:
        print(f"  winding number {label}: {w0:+.2f} at t = 0 -> {w1:+.2f} at t = {res.t[-1]:g}")
    print(f"  plot  -> {png}\n  video -> {video}")
    return info


def smooth_split(Nx: int, L: float, left: tuple[float, float, float], right: tuple[float, float, float],
                 width: float = 0.03) -> np.ndarray:
    """Two states joined by a smooth tanh step at ``x = L/2``.

    Parameters
    ----------
    Nx : int
        Grid cells per axis.
    L : float
        Side length of the bounding square.
    left, right : tuple of float
        States for ``x < L/2`` and ``x > L/2``.
    width : float, default 0.03
        Width of the transition in units of ``L``.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, Nx, Nx)``.
    """
    X, _ = grid(Nx, L)
    w = 0.5 * (1 + np.tanh((X - L / 2) / (width * L)))
    a = np.asarray(left, dtype=float)[:, None, None]
    b = np.asarray(right, dtype=float)[:, None, None]
    return a * (1 - w) + b * w
