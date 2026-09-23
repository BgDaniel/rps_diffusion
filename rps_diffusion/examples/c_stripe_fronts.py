"""(c) Stripe IC with small σ: travelling invasion fronts.

The initial state is pure S | R | P stripes. Rock invades Scissors, Paper
invades Rock and Scissors invades Paper, so every interface turns into a
front that moves at a speed of order √(Dλ), with thickness of order
√(D/λ), where D = σ²/2.
"""

from __future__ import annotations

from rps_diffusion import LambdaField, RPSSimulator, stripes
from rps_diffusion.examples._common import save_outputs


def main() -> None:
    """Run scenario (c)."""
    Nx, L, lam, sigma = 128, 1.0, 5.0, 0.03
    print(f"(c) stripes, lambda = {lam}, sigma = {sigma}")
    field = LambdaField(Nx, L).add_background(lam).build()
    sim = RPSSimulator(field, sigma=sigma, L=L, dt=0.005)
    res = sim.run(stripes(Nx, axis="x"), t_max=30.0, save_every=20)
    save_outputs(res, "c_stripe_fronts", stride=1, spectrum=False)


if __name__ == "__main__":
    main()
