"""Group 1: simply connected domains (no holes).

Every closed loop in these domains can be shrunk to a point. Any pattern
that winds around a point therefore needs a phase singularity there, which
turns into a spiral core. The group covers convex shapes (square, disk,
triangle) and non-convex ones (an L-shaped room, and a dumbbell of two
chambers joined by a narrow channel).
"""

from __future__ import annotations

import numpy as np

from rps_diffusion import Domain, concentrated, fixed_point, hills, stripes, winding
from rps_diffusion.domain import grid
from rps_diffusion.examples._common import Group, Scenario, smooth_split
from rps_diffusion.initial import normalise

RATES_ASYM = (2.0, 3.0, 4.0)
RS_ASYM = fixed_point(RATES_ASYM)


def _tilted_far_state(Nx: int, L: float) -> np.ndarray:
    X, Y = grid(Nx, L)
    tilt = 0.08 * (X + Y - L) / L
    return normalise(np.stack([0.80 + tilt, 0.12 - tilt / 2, 0.08 - tilt / 2]))


def _asym_concentrated(Nx: int, L: float) -> np.ndarray:
    rs = np.array(fixed_point((1.0, 2.0, 4.0)))
    return concentrated(Nx, L, radius=0.25, u0=tuple(rs + [0.06, -0.03, -0.03]), background=tuple(rs))


def _dumbbell(Nx: int, L: float) -> Domain:
    return (Domain(Nx, L, full=False)
            .add_disk(0.25 * L, 0.5 * L, 0.22 * L)
            .add_disk(0.75 * L, 0.5 * L, 0.22 * L)
            .add_rectangle(0.4 * L, 0.47 * L, 0.2 * L, 0.06 * L))


GROUP = Group(
    key="simply_connected",
    title="Simply connected domains (no holes)",
    description=__doc__.split("\n\n", 1)[1].strip(),
    scenarios=[
        Scenario(
            name="square_omega0",
            title=r"square, equal rates, small bump: mean fractions oscillate at $\omega_0 = \lambda/\sqrt{3}$",
            description="A small disk-shaped perturbation of (1/3, 1/3, 1/3). The domain average follows "
                        "the linear ODE exactly, so the spectrum peaks at f₀ = λ/(2π√3).",
            rates=2.0, sigma=0.05, Nx=64, dt=0.02, t_max=60.0, save_every=5,
            initial=lambda Nx, L: concentrated(Nx, L, u0=(0.40, 0.30, 0.30)),
        ),
        Scenario(
            name="square_asymmetric_rates",
            title=r"square, rates (1, 2, 4): fixed point $\rho^*$ shifts, "
                  r"$\omega_0 = \sqrt{\lambda_S\lambda_R\lambda_P/\Sigma\lambda}$",
            description="Small bump around ρ* = (4, 1, 2)/7. The mean fractions oscillate around ρ* at ω₀; "
                        "the species with the weakest attack (S) is the most abundant.",
            rates=(1.0, 2.0, 4.0), sigma=0.05, Nx=64, dt=0.02, t_max=80.0, save_every=5,
            initial=_asym_concentrated,
        ),
        Scenario(
            name="square_stripe_fronts",
            title=r"square, stripe start with small $\sigma$: travelling invasion fronts",
            description="Pure S | R | P stripes. Each interface becomes a front in which the winner of "
                        "the pair invades the loser.",
            rates=5.0, sigma=0.03, dt=0.02, t_max=30.0, save_every=5,
            initial=lambda Nx, L: stripes(Nx, axis="x"),
        ),
        Scenario(
            name="square_large_cycles",
            title="square, start far from the fixed point: large, volatile oscillations",
            description="Start at (0.80, 0.12, 0.08) with a gentle tilt. The mean fractions swing between "
                        "near-extinction and dominance; nonlinear cycles are slower than T₀.",
            rates=RATES_ASYM, sigma=0.03, t_max=60.0,
            initial=_tilted_far_state,
        ),
        Scenario(
            name="square_hills",
            title="square, non-flat start: one smooth hill per species",
            description="S hill lower left, R lower right, P at the top. The hills become rotating fronts "
                        "that reflect off the no-flux walls.",
            rates=RATES_ASYM, sigma=0.03, t_max=40.0,
            initial=lambda Nx, L: hills(Nx, L, width=0.12, amplitude=1.5),
        ),
        Scenario(
            name="disk_spiral",
            title="disk, species wound once around the centre: a rotating spiral",
            description="The phase winds once around the centre. In a simply connected domain the centre "
                        "is a phase singularity, so the pattern rotates as a spiral around it.",
            rates=RATES_ASYM, sigma=0.02, t_max=40.0,
            domain=lambda Nx, L: Domain.disk(Nx, L),
            initial=lambda Nx, L: winding(Nx, L, amplitude=0.8, background=RS_ASYM),
            loops=(("around the centre", (0.5, 0.5), 0.3),),
        ),
        Scenario(
            name="triangle_hills",
            title="triangle, one hill per species near each corner",
            description="A convex polygon. The fronts reflect off three walls meeting at 60° angles.",
            rates=3.0, sigma=0.03, t_max=40.0,
            domain=lambda Nx, L: Domain.polygon(Nx, L, [(0.05, 0.08), (0.95, 0.08), (0.5, 0.93)]),
            initial=lambda Nx, L: hills(Nx, L, centres=((0.3, 0.25), (0.7, 0.25), (0.5, 0.62)),
                                        width=0.1, amplitude=1.5),
        ),
        Scenario(
            name="l_shape_fronts",
            title="L-shaped room, stripe start: fronts turn the corner",
            description="A non-convex domain. The fronts entering the narrow arm are bent around the "
                        "re-entrant corner.",
            rates=4.0, sigma=0.03, dt=0.02, t_max=40.0,
            domain=lambda Nx, L: Domain.l_shape(Nx, L),
            initial=lambda Nx, L: stripes(Nx, axis="x"),
        ),
        Scenario(
            name="dumbbell_chambers",
            title="dumbbell: two chambers joined by a narrow channel, started out of phase",
            description="The left chamber starts S-rich and the right one P-rich. Each chamber cycles on its "
                        "own, and the thin channel slowly couples their phases.",
            rates=3.0, sigma=0.03, t_max=60.0,
            domain=_dumbbell,
            initial=lambda Nx, L: smooth_split(Nx, L, (0.6, 0.25, 0.15), (0.15, 0.25, 0.6)),
        ),
    ],
)
