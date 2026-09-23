"""Simulation of the cyclic Rock-Paper-Scissors reaction-diffusion PDE.

    ∂_t ρ_i = (σ²/2) ∇² ρ_i + f_i(ρ),   i ∈ {S, R, P}

    f_S = ρ_S (λ_S ρ_P − λ_R ρ_R)
    f_R = ρ_R (λ_R ρ_S − λ_P ρ_P)
    f_P = ρ_P (λ_P ρ_R − λ_S ρ_S)

with three constant interaction rates, on arbitrarily shaped domains with
no-flux boundary conditions.
"""

from .analysis import dominant_frequencies, frequency_spectrum, plot_spectrum, winding_number
from .domain import Domain
from .initial import blobs, concentrated, hills, homogeneous, random_perturbation, stripes, winding
from .simulator import RPSSimulator, SimResult, fixed_point, omega0
from .visualize import describe_parameters, make_video, plot_fractions, plot_snapshot, plot_surfaces

__all__ = [
    "Domain",
    "RPSSimulator",
    "SimResult",
    "fixed_point",
    "omega0",
    "homogeneous",
    "random_perturbation",
    "stripes",
    "blobs",
    "concentrated",
    "hills",
    "winding",
    "make_video",
    "plot_fractions",
    "plot_snapshot",
    "plot_surfaces",
    "describe_parameters",
    "frequency_spectrum",
    "dominant_frequencies",
    "plot_spectrum",
    "winding_number",
]

__version__ = "0.2.0"
