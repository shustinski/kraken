from __future__ import annotations

import numpy as np
import pytest

from cartograph.domain.coordinates import Translation2D
from cartograph.domain.registration import PairHint, RegistrationParameters, RegistrationStatus, compute_confidence, ConfidenceWeights
from cartograph.infrastructure.opencv.registrar import PythonPairRegistrar

from .helpers import buffer, crop_at, subpixel_shift, unique_scene


def _register(fixed: np.ndarray, moving: np.ndarray, expected: tuple[float, float], radius: float = 12.0):
    registrar = PythonPairRegistrar(RegistrationParameters(search_radius_px=radius, overlap_margin_px=4, top_k=5))
    hint = PairHint(expected=Translation2D(*expected), search_radius_px=radius, overlap_margin_px=4)
    return registrar.register(buffer(fixed), buffer(moving), hint)


def test_integer_translation_is_recovered() -> None:
    scene = unique_scene(180, 180, seed=1)
    dx, dy = 7.0, -3.0
    fixed = crop_at(scene, 20, 20, 96, 96)
    moving = crop_at(scene, 20 + dx, 20 + dy, 96, 96)
    result = _register(fixed, moving, expected=(dx, dy))
    assert result.status is RegistrationStatus.OK
    assert result.transform.dx == pytest.approx(dx, abs=0.35)
    assert result.transform.dy == pytest.approx(dy, abs=0.35)
    assert result.raw_zncc > 0.7
    assert result.confidence > 0.5


def test_subpixel_translation_is_recovered() -> None:
    scene = unique_scene(256, 256, seed=2, sigma=2.0)
    dx, dy = 4.5, -2.6
    fixed = crop_at(scene, 40, 40, 128, 128)
    moving_int = crop_at(scene, 44, 37, 128, 128)
    moving = subpixel_shift(moving_int, -0.4, -0.4)
    result = _register(fixed, moving, expected=(4.0, -3.0), radius=8.0)
    assert result.status is RegistrationStatus.OK
    assert result.transform.dx == pytest.approx(dx, abs=0.25)
    assert result.transform.dy == pytest.approx(dy, abs=0.25)
    assert result.raw_zncc > 0.7


def test_confidence_formula_is_deterministic() -> None:
    weights = ConfidenceWeights(peak_ratio=1.0, raw_zncc=0.0, gradient_zncc=0.0, phase_response=0.0, expected_displacement_error=0.0)
    value = compute_confidence(
        phase_response=0.9,
        peak_ratio=4.0,
        raw_zncc=0.8,
        gradient_zncc=0.7,
        expected_displacement_error=10.0,
        search_radius_px=8.0,
        weights=weights,
    )
    assert value == pytest.approx(0.75)


def test_empty_tiles_are_not_registered() -> None:
    blank = np.full((64, 64), 18.0, dtype=np.float32)
    result = _register(blank, blank, expected=(16.0, 0.0))
    assert result.status is RegistrationStatus.EMPTY_TILE
    assert result.confidence == 0.0
    assert result.message
