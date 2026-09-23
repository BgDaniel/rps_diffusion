"""(f) Square with a non-flat start: one smooth hill per species.

Each species starts with a Gaussian hill at a different place: S at the
lower left, R at the lower right, P at the top. Where two hills meet, the
winner of that pair invades the loser, so the hills turn into rotating,
expanding fronts that reflect off the no-flux walls of the square.
"""

from __future__ import annotations

from rps_diffusion import RPSSimulator, hills
from rps_diffusion.examples._common import report_frequencies, save_outputs


def main() -> None:
    """Run scenario (f)."""
    Nx, L, sigma = 96, 1.0, 0.03
    rates = (2.0, 3.0, 4.0)
    print(f"(f) square, three hills, rates {rates}")
    sim = RPSSimulator(rates, sigma=sigma, L=L, dt=0.01, Nx=Nx)
    res = sim.run(hills(Nx, L, width=0.12, amplitude=1.5), t_max=40.0, save_every=10)
    report_frequencies(res)
    save_outputs(res, "f_square_hills", "(f) square, non-flat start: one smooth hill per species")


if __name__ == "__main__":
    main()
