"""(c) Stripe IC with small σ: travelling invasion fronts.

The initial state is pure S | R | P stripes. Rock invades Scissors, Paper
invades Rock and Scissors invades Paper, so every interface turns into a
front that moves at a speed of order √(Dλ), with thickness of order
√(D/λ), where D = σ²/2.
"""

from __future__ import annotations

from rps_diffusion import RPSSimulator, stripes
from rps_diffusion.examples._common import report_frequencies, save_outputs


def main() -> None:
    """Run scenario (c)."""
    Nx, L, lam, sigma = 128, 1.0, 5.0, 0.03
    print(f"(c) stripes, lambda = {lam}, sigma = {sigma}")
    sim = RPSSimulator(lam, sigma=sigma, L=L, dt=0.02, Nx=Nx)
    res = sim.run(stripes(Nx, axis="x"), t_max=30.0, save_every=5)
    report_frequencies(res)
    save_outputs(res, "c_stripe_fronts", r"(c) stripe IC with small $\sigma$: travelling invasion fronts")


if __name__ == "__main__":
    main()
