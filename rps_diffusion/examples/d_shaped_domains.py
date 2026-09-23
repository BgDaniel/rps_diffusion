"""(d) Non-square domains: a ring, and a disk with obstacles.

The domain is a boolean mask. Mass moves only between neighbouring cells
that both lie inside it, so the no-flux (Neumann) condition holds on any
shape, including the boundaries of the holes.
"""

from __future__ import annotations

from rps_diffusion import Domain, RPSSimulator, blobs
from rps_diffusion.examples._common import report_frequencies, save_outputs


def main() -> None:
    """Run scenario (d) on two domains."""
    Nx, L, sigma = 128, 1.0, 0.02
    rates = (2.0, 3.0, 4.0)
    ring = Domain.annulus(Nx, L, inner_fraction=0.45)
    obstacles = (
        Domain.disk(Nx, L)
        .cut_disk(0.35, 0.6, 0.1)
        .cut_disk(0.65, 0.4, 0.12)
        .cut_rectangle(0.45, 0.8, 0.1, 0.2)
    )
    for name, label, dom in (("d_ring", "ring", ring), ("d_obstacles", "disk with holes and a notch", obstacles)):
        print(f"(d) {name}: {dom}")
        sim = RPSSimulator(rates, sigma=sigma, L=L, dt=0.02, domain=dom)
        res = sim.run(blobs(Nx, L, n_blobs=8, seed=1), t_max=30.0, save_every=5)
        report_frequencies(res)
        save_outputs(res, name, f"(d) shaped domain with no-flux boundary: {label}")


if __name__ == "__main__":
    main()
