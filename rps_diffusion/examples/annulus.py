"""Group 2: domains with one hole (annulus topology).

A loop around the hole cannot be shrunk to a point. The phase of the
S → R → P cycle can therefore wind around the hole without a phase
singularity anywhere in the domain, and the pattern travels around the
ring as a rotating wave. The winding number along the ring can only change
through a "phase slip", in which the oscillation amplitude vanishes across
the whole width of the ring. The gallery reports the measured winding
number at the start and at the end of each run. Compare
`ring_rotating_wave` with `disk_spiral` in the simply connected group:
it starts from the same initial condition, but the disk has no hole, so its
centre is a phase singularity (a spiral core).
"""

from __future__ import annotations

from rps_diffusion import Domain, fixed_point, hills, winding
from rps_diffusion.examples._common import Group, Scenario

RATES_ASYM = (2.0, 3.0, 4.0)
RS_ASYM = fixed_point(RATES_ASYM)

GROUP = Group(
    key="annulus",
    title="Domains with one hole (annulus topology)",
    description=__doc__.split("\n\n", 1)[1].strip(),
    scenarios=[
        Scenario(
            name="ring_rotating_wave",
            title="ring, species wound once around the hole: a rotating wave",
            description="The same winding start as `disk_spiral`, but the hole removes the phase "
                        "singularity. The pattern travels around the ring, keeping winding number 1.",
            rates=RATES_ASYM, sigma=0.02, t_max=40.0,
            domain=lambda Nx, L: Domain.annulus(Nx, L, inner_fraction=0.5),
            initial=lambda Nx, L: winding(Nx, L, amplitude=0.8, background=RS_ASYM),
            loops=(("around the hole", (0.5, 0.5), 0.37),),
        ),
        Scenario(
            name="ring_double_winding",
            title="ring, species wound twice around the hole: two wave crests",
            description="Winding number 2: the S → R → P sequence appears twice around the ring, so two "
                        "wave crests chase each other.",
            rates=RATES_ASYM, sigma=0.02, t_max=40.0,
            domain=lambda Nx, L: Domain.annulus(Nx, L, inner_fraction=0.5),
            initial=lambda Nx, L: winding(Nx, L, amplitude=0.8, background=RS_ASYM, winding_number=2),
            loops=(("around the hole", (0.5, 0.5), 0.37),),
        ),
        Scenario(
            name="ring_hills",
            title="ring, one hill per species",
            description="No winding at the start. The hills spread along the ring and collide.",
            rates=RATES_ASYM, sigma=0.02, dt=0.02, t_max=30.0,
            domain=lambda Nx, L: Domain.annulus(Nx, L, inner_fraction=0.45),
            initial=lambda Nx, L: hills(Nx, L, width=0.1, amplitude=1.5),
        ),
        Scenario(
            name="square_with_hole_winding",
            title="square with a round hole, species wound around the hole",
            description="The same topology as the ring but a different geometry: the wave rotates around "
                        "the hole and is deformed by the square's corners.",
            rates=3.0, sigma=0.03, t_max=40.0,
            domain=lambda Nx, L: Domain(Nx, L).cut_disk(0.5 * L, 0.5 * L, 0.18 * L),
            initial=lambda Nx, L: winding(Nx, L, amplitude=0.8),
            loops=(("around the hole", (0.5, 0.5), 0.3),),
        ),
    ],
)
