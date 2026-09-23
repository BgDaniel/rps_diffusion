"""Simulation of the cyclic Rock-Paper-Scissors reaction-diffusion PDE.

    ∂_t ρ_i = (σ²/2) ∇² ρ_i + λ(x) f_i(ρ),   i ∈ {S, R, P}

on arbitrarily shaped domains with no-flux boundary conditions.
"""

from .analysis import dominant_frequencies, frequency_spectrum, plot_spectrum
from .domain import Domain, LambdaField
from .initial import blobs, concentrated, homogeneous, random_perturbation, stripes
from .simulator import RPSSimulator, SimResult
from .visualize import make_video, plot_fractions, plot_snapshot, plot_surfaces

__all__ = [
    "Domain",
    "LambdaField",
    "RPSSimulator",
    "SimResult",
    "homogeneous",
    "random_perturbation",
    "stripes",
    "blobs",
    "concentrated",
    "make_video",
    "plot_fractions",
    "plot_snapshot",
    "plot_surfaces",
    "frequency_spectrum",
    "dominant_frequencies",
    "plot_spectrum",
]

__version__ = "0.1.0"
