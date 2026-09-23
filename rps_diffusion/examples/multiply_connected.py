"""Group 3: domains with several holes (multiply connected).

With k holes there are k independent non-contractible loops, so a pattern
can wind around each hole separately. Obstacles also scatter and split
travelling fronts.
"""

from __future__ import annotations

from rps_diffusion import Domain, fixed_point, hills, stripes, winding
from rps_diffusion.examples._common import Group, Scenario

RATES_ASYM = (2.0, 3.0, 4.0)
RS_ASYM = fixed_point(RATES_ASYM)


def _disk_with_obstacles(Nx: int, L: float) -> Domain:
    return (Domain.disk(Nx, L)
            .cut_disk(0.35 * L, 0.6 * L, 0.1 * L)
            .cut_disk(0.65 * L, 0.4 * L, 0.12 * L)
            .cut_rectangle(0.45 * L, 0.8 * L, 0.1 * L, 0.2 * L))


def _porous_square(Nx: int, L: float) -> Domain:
    dom = Domain(Nx, L)
    for cx in (0.2, 0.5, 0.8):
        for cy in (0.2, 0.5, 0.8):
            dom.cut_disk(cx * L, cy * L, 0.07 * L)
    return dom


def _two_holes(Nx: int, L: float) -> Domain:
    return Domain(Nx, L).cut_disk(0.3 * L, 0.5 * L, 0.12 * L).cut_disk(0.7 * L, 0.5 * L, 0.12 * L)


GROUP = Group(
    key="multiply_connected",
    title="Domains with several holes (multiply connected)",
    description=__doc__.split("\n\n", 1)[1].strip(),
    scenarios=[
        Scenario(
            name="two_holes_winding_left",
            title="square with two holes, species wound around the left hole only",
            description="The phase winds around the left hole but not the right one. The rotating wave "
                        "circles the left hole and has to pass the right hole on both sides.",
            rates=RATES_ASYM, sigma=0.025, t_max=40.0,
            domain=_two_holes,
            initial=lambda Nx, L: winding(Nx, L, centre=(0.3, 0.5), amplitude=0.8, background=RS_ASYM),
            loops=(("around the left hole", (0.3, 0.5), 0.17), ("around the right hole", (0.7, 0.5), 0.17)),
        ),
        Scenario(
            name="disk_obstacles_hills",
            title="disk with two holes and a notch, one hill per species",
            description="The fronts wrap around the obstacles and meet again behind them.",
            rates=RATES_ASYM, sigma=0.02, dt=0.02, t_max=30.0,
            domain=_disk_with_obstacles,
            initial=lambda Nx, L: hills(Nx, L, width=0.1, amplitude=1.5),
        ),
        Scenario(
            name="porous_square_fronts",
            title="porous square (3 x 3 holes), stripe start: fronts scatter off obstacles",
            description="Stripe fronts travel through a lattice of holes; each hole splits the front, "
                        "and the pieces rejoin behind it.",
            rates=4.0, sigma=0.03, dt=0.02, t_max=40.0,
            domain=_porous_square,
            initial=lambda Nx, L: stripes(Nx, axis="x"),
        ),
    ],
)
