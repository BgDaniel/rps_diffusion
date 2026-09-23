"""(b) Three different rates: shifted fixed point and ω₀ = √(λ_S λ_R λ_P / Σλ).

With rates (λ_S, λ_R, λ_P) the coexistence point moves to
ρ* = (λ_P, λ_S, λ_R) / Σλ. Each species' share is set by the rate at which
its prey is eaten by the third species, so the species with the *strongest*
own attack is not the most abundant ("survival of the weakest"). A small
perturbation of ρ* oscillates at ω₀ = √(λ_S λ_R λ_P / Σλ).
"""

from __future__ import annotations

import numpy as np

from rps_diffusion import RPSSimulator, concentrated, fixed_point
from rps_diffusion.examples._common import report_frequencies, save_outputs


def main() -> None:
    """Run scenario (b)."""
    Nx, L, sigma = 64, 1.0, 0.05
    rates = (1.0, 2.0, 4.0)
    rs = np.array(fixed_point(rates))
    print(f"(b) asymmetric rates {rates}, fixed point {np.round(rs, 4)}")
    sim = RPSSimulator(rates, sigma=sigma, L=L, dt=0.02, Nx=Nx)
    rho0 = concentrated(Nx, L, radius=0.25, u0=tuple(rs + [0.06, -0.03, -0.03]), background=tuple(rs))
    res = sim.run(rho0, t_max=80.0, save_every=5)
    report_frequencies(res, n=1)
    late = res.t > res.t[-1] / 2
    print(f"  time-averaged fractions (2nd half): {np.round(res.fractions[late].mean(axis=0), 4)}")
    save_outputs(res, "b_asymmetric_rates",
                 r"(b) three different rates: fixed point $\rho^*$ shifts, $\omega_0 = \sqrt{\lambda_S\lambda_R\lambda_P/\Sigma\lambda}$")


if __name__ == "__main__":
    main()
