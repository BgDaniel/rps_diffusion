"""(b) Two disks with different λ: the mean fractions carry two frequencies.

λ vanishes outside two disks with rates λ₁ and λ₂. Each disk oscillates at
its own ω_k = λ_k/√3. Weak diffusion (small σ) couples the disks only near
their edges, so the spectrum of the domain average shows both peaks.
"""

from __future__ import annotations

import numpy as np

from rps_diffusion import LambdaField, RPSSimulator, homogeneous
from rps_diffusion.examples._common import report_frequencies, save_outputs


def _freq(lam: float) -> float:
    return lam / np.sqrt(3) / (2 * np.pi)


def main() -> None:
    """Run scenario (b)."""
    Nx, L, sigma = 64, 1.0, 0.02
    lam1, lam2 = 1.5, 4.5
    print(f"(b) two disks, lambda1 = {lam1}, lambda2 = {lam2}")
    field = (
        LambdaField(Nx, L)
        .add_background(0.0)
        .add_disk(lam1, 0.28, 0.28, 0.2)
        .add_disk(lam2, 0.72, 0.72, 0.2)
        .build()
    )
    sim = RPSSimulator(field, sigma=sigma, L=L, dt=0.02)
    res = sim.run(homogeneous(Nx, (0.37, 0.315, 0.315)), t_max=80.0, save_every=5)
    report_frequencies(res, {"f(lam1)": _freq(lam1), "f(lam2)": _freq(lam2)}, n=2)
    save_outputs(res, "b_two_disks", contrast=5.0)


if __name__ == "__main__":
    main()
