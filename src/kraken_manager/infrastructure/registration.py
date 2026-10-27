"""Normalized affine least squares using optional NumPy SVD.

No automatic outlier deletion: inconsistent observations reject the candidate.
The returned uncertainty allowance applies only at the supplied control points.
"""
from __future__ import annotations

from math import isfinite

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.multilayer import RegistrationFit, number


def fit_affine(source, target, control_points, *, max_error_m, max_condition=1e6):
    number(max_error_m, "max_error_m", minimum=0)
    number(max_condition, "max_condition", minimum=1)
    if len(control_points) < 3:
        return RegistrationFit("insufficient", "at_least_three_noncollinear_control_points_required")
    try:
        import numpy as np
    except ImportError as exc:
        raise DomainValidationError("Affine registration requires the optional NumPy runtime") from exc
    x = np.asarray([source.to_physical(p.source_xy) for p in control_points], dtype=np.float64)
    y = np.asarray([target.to_physical(p.target_xy) for p in control_points], dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return RegistrationFit("rejected", "nonfinite_physical_coordinates")
    center_x, center_y = x.mean(axis=0), y.mean(axis=0)
    scale_x = float(np.sqrt(np.mean(np.sum((x - center_x) ** 2, axis=1))))
    scale_y = float(np.sqrt(np.mean(np.sum((y - center_y) ** 2, axis=1))))
    if scale_x <= 0 or scale_y <= 0 or not isfinite(scale_x + scale_y):
        return RegistrationFit("rejected", "degenerate_control_point_spread")
    design = np.column_stack(((x - center_x) / scale_x, np.ones(len(x))))
    try:
        coefficients, _residual_sum, rank, singular = np.linalg.lstsq(design, (y - center_y) / scale_y, rcond=None)
    except np.linalg.LinAlgError:
        return RegistrationFit("rejected", "svd_did_not_converge")
    if rank < 3 or singular[-1] <= 0:
        return RegistrationFit("rejected", "collinear_or_duplicate_control_points")
    condition = float(singular[0] / singular[-1])
    if not isfinite(condition) or condition > max_condition:
        return RegistrationFit("rejected", "ill_conditioned_control_points")
    linear = coefficients[:2, :].T * (scale_y / scale_x)
    translation = center_y + coefficients[2, :] * scale_y - linear @ center_x
    transform_condition = float(np.linalg.cond(linear))
    if not isfinite(transform_condition) or transform_condition > max_condition:
        return RegistrationFit("rejected", "singular_or_ill_conditioned_transform", design_condition=condition)
    predicted = x @ linear.T + translation
    residuals = tuple(float(v) for v in np.linalg.norm(predicted - y, axis=1))
    rms, maximum = float(np.sqrt(np.mean(np.square(residuals)))), max(residuals)
    # Zero requested tolerance allows roundoff on finite, normalized input.
    numeric_allowance = np.finfo(np.float64).eps * max(scale_y, scale_x) * 128
    if maximum > max_error_m + numeric_allowance:
        return RegistrationFit(
            "rejected", "control_residual_exceeds_tolerance_possible_outlier", residuals_m=residuals,
            rms_m=rms, max_residual_m=maximum, design_condition=condition, transform_condition=transform_condition,
        )
    bound = None
    if (source.uncertainty_m is not None and target.uncertainty_m is not None
            and all(p.uncertainty_m is not None for p in control_points)):
        amplification = float(np.linalg.norm(linear, ord=2))
        bound = maximum + amplification * source.uncertainty_m + target.uncertainty_m + max(
            p.uncertainty_m for p in control_points
        )
    affine = (float(linear[0, 0]), float(linear[0, 1]), float(translation[0]),
              float(linear[1, 0]), float(linear[1, 1]), float(translation[1]))
    return RegistrationFit(
        "fitted", "affine_fit_of_declared_control_points", affine, residuals, rms, maximum,
        condition, transform_condition, bound,
    )
