"""Example scenarios, grouped by the topology of the domain.

* :mod:`~rps_diffusion.examples.simply_connected`: no holes (square, disk,
  triangle, L-shape, dumbbell).
* :mod:`~rps_diffusion.examples.annulus`: one hole (rings, square with a hole).
* :mod:`~rps_diffusion.examples.multiply_connected`: several holes.

Run ``python -m rps_diffusion.examples --list`` to see all scenarios. Output
(summary figures and videos) goes to ``examples/output/<group>/`` below the
current directory.
"""

from rps_diffusion.examples import annulus, multiply_connected, simply_connected

GROUPS = [simply_connected.GROUP, annulus.GROUP, multiply_connected.GROUP]
