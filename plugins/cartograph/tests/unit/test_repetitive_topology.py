from __future__ import annotations

import pytest

from cartograph.domain.coordinates import Translation2D
from cartograph.domain.registration import PairHint, RegistrationParameters, RegistrationStatus
from cartograph.infrastructure.opencv.registrar import PythonPairRegistrar

from .helpers import buffer, crop_at, striped_scene


def test_expected_displacement_rejects_periodic_false_peak() -> None:
    period = 16
    scene = striped_scene(128, 220, period=period)
    true_dx = 5.0
    fixed = crop_at(scene, 20, 20, 96, 96)
    moving = crop_at(scene, 20 + true_dx, 20, 96, 96)
    registrar = PythonPairRegistrar(RegistrationParameters(search_radius_px=6.0, overlap_margin_px=4, top_k=5))
    result = registrar.register(
        buffer(fixed),
        buffer(moving),
        PairHint(expected=Translation2D(true_dx, 0.0), search_radius_px=6.0, overlap_margin_px=4),
    )
    assert result.status in {RegistrationStatus.OK, RegistrationStatus.LOW_CONFIDENCE}
    assert result.transform.dx == pytest.approx(true_dx, abs=0.75)
    assert abs(result.transform.dx - (true_dx - period)) > 2.0
    assert abs(result.transform.dx - (true_dx + period)) > 2.0
