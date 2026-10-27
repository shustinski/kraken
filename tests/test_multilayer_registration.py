from __future__ import annotations

from dataclasses import replace
import json
from uuid import uuid4

import pytest

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.multilayer import (
    ArtifactAnchor, ControlPoint, CoordinateCalibration, MultilayerModel, PhysicalLayer,
)
from kraken_manager.domain.project import LayerType
from kraken_manager.infrastructure.registration import fit_affine


def uid():
    return str(uuid4())


def calibration(**kwargs):
    return CoordinateCalibration(uid(), 0, ArtifactAnchor(uid(), "a" * 64), (1e-6, 2e-6), **kwargs)


def test_affine_solves_known_transform_after_calibration_with_measured_residuals():
    source = calibration(origin_m=(3e-6, -4e-6), y_axis="down", uncertainty_m=1e-9)
    target = calibration(origin_m=(-2e-6, 5e-6), y_axis="up", uncertainty_m=2e-9)
    expected = (1.5, 0.25, 7e-6, -0.2, 0.8, -3e-6)
    points = []
    for point in ((0, 0), (100, 0), (0, 100), (70, 80), (20, 30)):
        x, y = source.to_physical(point)
        physical = (expected[0] * x + expected[1] * y + expected[2],
                    expected[3] * x + expected[4] * y + expected[5])
        pixel = ((physical[0] - target.origin_m[0]) / target.pixel_size_m[0],
                 (physical[1] - target.origin_m[1]) / target.pixel_size_m[1])
        points.append(ControlPoint(point, pixel, 3e-9))
    fit = fit_affine(source, target, points, max_error_m=1e-12)
    assert fit.status == "fitted"
    assert fit.affine == pytest.approx(expected, abs=1e-12)
    assert len(fit.residuals_m) == 5
    assert fit.max_residual_m < 1e-12
    assert fit.control_error_bound_m > 5e-9
    assert fit.bound_scope == "supplied_control_points_only"
    assert fit.apply((1e-6, 2e-6)) == pytest.approx((9e-6, -1.6e-6), abs=1e-12)


@pytest.mark.parametrize("points,reason", [
    ([((0, 0), (0, 0)), ((1, 1), (1, 1))], "at_least_three"),
    ([((0, 0), (0, 0)), ((1, 1), (1, 1)), ((2, 2), (2, 2))], "collinear"),
    ([((0, 0), (0, 0))] * 4, "degenerate"),
])
def test_insufficient_duplicate_and_collinear_controls_never_publish_a_transform(points, reason):
    result = fit_affine(calibration(), calibration(), [ControlPoint(*p) for p in points], max_error_m=1e-9)
    assert result.status in {"insufficient", "rejected"}
    assert reason in result.reason
    assert result.affine is None
    with pytest.raises(DomainValidationError):
        result.apply((0, 0))


def test_large_outlier_is_rejected_without_silent_control_deletion():
    coordinates = [(0, 0), (100, 0), (0, 100), (100, 100), (50, 50)]
    points = [ControlPoint(point, point) for point in coordinates]
    points[-1] = ControlPoint((50, 50), (500, -300))
    fit = fit_affine(calibration(), calibration(), points, max_error_m=1e-9)
    assert fit.status == "rejected"
    assert "possible_outlier" in fit.reason
    assert len(fit.residuals_m) == len(points)
    assert fit.max_residual_m > 1e-6
    assert fit.affine is None


def test_unknown_calibration_uncertainty_is_not_reported_as_zero():
    points = [ControlPoint(p, p) for p in ((0, 0), (1, 0), (0, 1), (1, 1))]
    fit = fit_affine(calibration(), calibration(), points, max_error_m=1e-12)
    assert fit.status == "fitted"
    assert fit.control_error_bound_m is None


def test_nearly_collinear_controls_and_collapsed_target_are_rejected():
    near_line = [ControlPoint((0, 0), (0, 0)), ControlPoint((1, 1e-10), (1, 1e-10)),
                 ControlPoint((2, 0), (2, 0)), ControlPoint((3, 1e-10), (3, 1e-10))]
    fit = fit_affine(calibration(), calibration(), near_line, max_error_m=1e-9, max_condition=1000)
    assert fit.status == "rejected"
    assert "condition" in fit.reason
    collapsed = [ControlPoint(p, (p[0], 0)) for p in ((0, 0), (1, 0), (0, 1), (1, 1))]
    assert fit_affine(calibration(), calibration(), collapsed, max_error_m=1e-9).affine is None


@pytest.mark.parametrize("scale", [(0, 1), (-1, 1), (float("nan"), 1), (True, 1)])
def test_calibration_rejects_invalid_scales(scale):
    with pytest.raises(DomainValidationError):
        replace(calibration(), pixel_size_m=scale)


def test_stack_requires_physical_z_order_unique_layers_and_nonoverlapping_thickness():
    a = PhysicalLayer(uid(), LayerType.METAL, 0, 1e-7, True)
    b = PhysicalLayer(uid(), LayerType.CONTACT, 1e-6, 1e-7)
    model = MultilayerModel(uid(), layers=(a, b))
    assert MultilayerModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    for layers in ((b, a), (a, a), (a, replace(b, z_m=1e-8))):
        with pytest.raises(DomainValidationError):
            replace(model, layers=layers)
    with pytest.raises(DomainValidationError, match="unit"):
        MultilayerModel.from_dict({**model.to_dict(), "unit": "px"})
