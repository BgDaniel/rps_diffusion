"""Tests for Domain, LambdaField and the initial-condition factories."""

from __future__ import annotations

import numpy as np
import pytest

from rps_diffusion import Domain, LambdaField, blobs, concentrated, homogeneous, random_perturbation, stripes


def test_lambda_field_layers_in_order() -> None:
    lam = (
        LambdaField(40, 1.0)
        .add_background(1.0)
        .add_square(2.0, 0.0, 0.0, 0.5)
        .add_disk(3.0, 0.25, 0.25, 0.1)
        .add_annulus(4.0, 0.75, 0.75, 0.1, 0.2)
        .build()
    )
    assert lam.shape == (40, 40)
    assert lam[0, 39] == 1.0  # cell (x, y) ≈ (0.99, 0.01): outside the square -> background
    assert lam[2, 2] == 2.0  # inside square, outside disk
    assert lam[10, 10] == 3.0  # disk centre overwrites square
    assert lam[30, 30] == 1.0  # annulus hole
    assert set(np.unique(lam)) == {1.0, 2.0, 3.0, 4.0}


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
    ],
)
def test_initial_conditions_are_simplex_valued(rho: np.ndarray) -> None:
    assert rho.shape == (3, 16, 16)
    assert rho.min() >= 0 and rho.max() <= 1
    np.testing.assert_allclose(rho.sum(axis=0), 1.0)


def test_stripes_order() -> None:
    rho = stripes(30)
    assert np.all(rho[0, :, :10] == 1) and np.all(rho[1, :, 10:20] == 1) and np.all(rho[2, :, 20:] == 1)
