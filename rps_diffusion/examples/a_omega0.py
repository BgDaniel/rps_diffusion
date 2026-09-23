"""(a) Equal rates, concentrated IC: the mean fractions oscillate at ω₀ = λ/√3.

With equal rates λ the fixed point is (1/3, 1/3, 1/3). Linearised around
it, the reaction commutes with the Neumann Laplacian, so the domain average
of each species follows the plain linear ODE with frequency ω₀ = λ/√3,
independent of σ. Diffusion only smooths the spatial profile: mode (m, n)
decays at rate γ_mn = σ²π²(m²+n²)/(2L²).
"""

from __future__ import annotations

from rps_diffusion import RPSSimulator, concentrated
from rps_diffusion.examples._common import report_frequencies, save_outputs


def main() -> None:
    """Run scenario (a)."""
    Nx, L, lam, sigma = 64, 1.0, 2.0, 0.05
    print(f"(a) equal rates lambda = {lam}, concentrated IC")
    sim = RPSSimulator(lam, sigma=sigma, L=L, dt=0.02, Nx=Nx)
    res = sim.run(concentrated(Nx, L, u0=(0.40, 0.30, 0.30)), t_max=60.0, save_every=5)
    report_frequencies(res, n=1)
    save_outputs(res, "a_omega0",
                 r"(a) equal rates, concentrated IC: mean fractions oscillate at $\omega_0 = \lambda/\sqrt{3}$")


if __name__ == "__main__":
    main()
