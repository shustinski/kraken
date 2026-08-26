from __future__ import annotations

import pytest

from cartograph.domain.coordinates import GridCoordinate, Translation2D
from cartograph.domain.topology import cycle_measurement_residual


def test_unit_square_cycle_matches_specified_residual() -> None:
    a = GridCoordinate(0, 0)
    b = GridCoordinate(0, 1)
    e = GridCoordinate(1, 1)
    d = GridCoordinate(1, 0)
    measurements = {
        (a, b): Translation2D(10.0, 0.0),
        (b, e): Translation2D(0.0, 10.0),
        (d, e): Translation2D(10.0, 1.0),
        (a, d): Translation2D(0.0, 10.0),
    }
    # d_AB + d_BE - d_DE - d_AD = (10,0)+(0,10)-(10,1)-(0,10) = (0,-1)
    residual = cycle_measurement_residual(measurements, (a, b, e, d))
    assert residual == pytest.approx(1.0)
