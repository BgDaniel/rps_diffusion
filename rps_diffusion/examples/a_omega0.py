"""(a) Homogeneous λ, concentrated IC: the mean fractions oscillate at ω₀ = λ/√3.

With a spatially constant λ, the reaction term linearised around
(1/3, 1/3, 1/3) commutes with the Neumann Laplacian. The domain average of
each species therefore follows the plain linear ODE with frequency ω₀,
independent of σ, while diffusion only smooths the spatial profile
(mode (m, n) decays at rate γ_mn = σ²π²(m²+n²)/(2L²)).
"""

from __future__ import annotations

import numpy as np

from rps_diffusion import LambdaField, RPSSimulator, concentrated
from rps_diffusion.examples._common import report_frequencies, save_outputs


def main() -> None:
    """Run scenario (a)."""
    Nx, L, lam, sigma = 64, 1.0, 2.0, 0.05
    print(f"(a) homogeneous lambda = {lam}, concentrated IC")
    field = LambdaField(Nx, L).add_background(lam).build()
    sim = RPSSimulator(field, sigma=sigma, L=L, dt=0.005)
    res = sim.run(concentrated(Nx, L, u0=(0.40, 0.30, 0.30)), t_max=60.0, save_every=20)
    report_frequencies(res, {"f0": lam / np.sqrt(3) / (2 * np.pi)}, n=1)
    save_outputs(res, "a_omega0", stride=3, contrast=6.0)


if __name__ == "__main__":
    main()
