"""(e) Square, far from equilibrium: large, volatile oscillations.

The system starts with Scissors dominating, (0.80, 0.12, 0.08), far from
the fixed point ρ*, plus a gentle smooth tilt across the square so that
different regions are slightly out of phase. Far from ρ* the reaction is
strongly nonlinear. The mean fractions swing between near-extinction
(about 0.02) and dominance (about 0.8) for each species in turn. Their
period is about 20 % longer than the linear T₀ = 2π/ω₀, and the swings
slowly shrink as the regions drift out of phase and average out.
"""

from __future__ import annotations

import numpy as np

from rps_diffusion import RPSSimulator
from rps_diffusion.domain import grid
from rps_diffusion.examples._common import report_frequencies, save_outputs
from rps_diffusion.initial import normalise


def main() -> None:
    """Run scenario (e)."""
    Nx, L, sigma = 96, 1.0, 0.03
    rates = (2.0, 3.0, 4.0)
    print(f"(e) square, large cycles, rates {rates}")
    X, Y = grid(Nx, L)
    tilt = 0.08 * (X + Y - L) / L  # smooth, deterministic, in [-0.08, 0.08]
    rho0 = normalise(np.stack([0.80 + tilt, 0.12 - tilt / 2, 0.08 - tilt / 2]))
    sim = RPSSimulator(rates, sigma=sigma, L=L, dt=0.01, Nx=Nx)
    res = sim.run(rho0, t_max=60.0, save_every=10)
    report_frequencies(res)
    print(f"  range of mean fractions: min {res.fractions.min():.3f}, max {res.fractions.max():.3f}")
    save_outputs(res, "e_square_large_cycles",
                 "(e) square, start far from the fixed point: large, volatile oscillations")


if __name__ == "__main__":
    main()
