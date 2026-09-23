"""Tests for Domain and the initial-condition factories."""

from __future__ import annotations

import numpy as np
import pytest

from rps_diffusion import Domain, blobs, concentrated, hills, homogeneous, random_perturbation, stripes


def test_domain_shapes() -> None:
    Nx = 100
    disk = Domain.disk(Nx)
    assert disk.area == pytest.approx(np.pi / 4, rel=0.02)
    ring = Domain.annulus(Nx, inner_fraction=0.5)
    assert ring.area == pytest.approx(np.pi / 4 * 0.75, rel=0.03)
    assert Domain.l_shape(Nx).area == pytest.approx(0.75, rel=0.02)
    tri = Domain.polygon(Nx, 1.0, [(0, 0), (1, 0), (0, 1)])
    assert tri.area == pytest.approx(0.5, rel=0.03)
    assert Domain.square(Nx).is_full


def test_domain_disconnected_warns() -> None:
    dom = Domain(50, full=False).add_disk(0.2, 0.2, 0.1).add_disk(0.8, 0.8, 0.1)
    assert dom.n_components() == 2
    with pytest.warns(UserWarning):
        dom.validate()
    with pytest.raises(ValueError):
        Domain(50, full=False).validate()


@pytest.mark.parametrize(
    "rho",
    [
        homogeneous(16),
        homogeneous(16, (0.5, 0.3, 0.2)),
        random_perturbation(16, noise=0.2),
        stripes(16),
        stripes(16, axis="y"),
        blobs(16),
        concentrated(16),
        concentrated(16, background=(0.2, 0.3, 0.5)),
        random_perturbation(16, background=(0.5, 0.2, 0.3)),
        hills(16),
        hills(16, background=(0.2, 0.3, 0.5), amplitude=0.3),
    ],
)
def test_initial_conditions_are_simplex_valued(rho: np.ndarray) -> None:
    assert rho.shape == (3, 16, 16)
    assert rho.min() >= 0 and rho.max() <= 1
    np.testing.assert_allclose(rho.sum(axis=0), 1.0)


def test_stripes_order() -> None:
    rho = stripes(30)
    assert np.all(rho[0, :, :10] == 1) and np.all(rho[1, :, 10:20] == 1) and np.all(rho[2, :, 20:] == 1)
