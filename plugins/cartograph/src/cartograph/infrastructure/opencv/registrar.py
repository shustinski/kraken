"""Python registration backend: phase correlation, Top-K peaks, ZNCC, gradient scoring."""

from __future__ import annotations

import logging

from cartograph.domain.coordinates import Translation2D
from cartograph.domain.registration import (
    PairHint,
    PairRegistrar,
    PhasePeak,
    RegistrationBackend,
    RegistrationMethod,
    RegistrationParameters,
    RegistrationResult,
    RegistrationStatus,
    compute_confidence,
    translation_result,
)
from cartograph.domain.tiles import ImageBuffer

from .phase import extract_overlap_crops, phase_correlation_map, top_k_peaks
from .scoring import gaussian_smooth, gradient_magnitude, mean_gradient, zncc_at_shift, zncc_translation_candidates

_LOGGER = logging.getLogger(__name__)


class PythonPairRegistrar:
    """In-process pair registrar. Replaceable later by NativeRegistrationBackend."""

    def __init__(self, parameters: RegistrationParameters | None = None) -> None:
        self._parameters = parameters or RegistrationParameters()

    def register(self, fixed: ImageBuffer, moving: ImageBuffer, hint: PairHint) -> RegistrationResult:
        params = self._parameters
        empty = _informationless(fixed, moving, params)
        if empty is not None:
            return empty

        prepared_fixed = gaussian_smooth(fixed.pixels, params.gaussian_sigma)
        prepared_moving = gaussian_smooth(moving.pixels, params.gaussian_sigma)
        crops = extract_overlap_crops(
            prepared_fixed,
            prepared_moving,
            hint.expected,
            margin_px=hint.overlap_margin_px,
            min_overlap_px=params.min_overlap_px,
        )
        if crops is None:
            return RegistrationResult.skipped(
                status=RegistrationStatus.FAILED,
                message="expected overlap is too small for constrained registration",
            )

        correlation = phase_correlation_map(crops.fixed, crops.moving)
        peaks = top_k_peaks(
            correlation,
            top_k=params.top_k,
            search_radius_px=hint.search_radius_px,
        )
        zncc_shifts = zncc_translation_candidates(
            crops.fixed,
            crops.moving,
            search_radius_px=hint.search_radius_px,
            top_k=params.top_k,
        )
        if not peaks and not zncc_shifts:
            return RegistrationResult.skipped(
                status=RegistrationStatus.FAILED,
                message="no phase-correlation peak inside the geometrically allowed search window",
            )

        gradient_fixed = gradient_magnitude(crops.fixed, params.gaussian_sigma)
        gradient_moving = gradient_magnitude(crops.moving, params.gaussian_sigma)

        template = peaks[0] if peaks else None
        candidate_peaks = _merge_candidate_peaks(peaks, zncc_shifts, template=template)
        scored: list[RegistrationResult] = []
        for peak in candidate_peaks:
            scored.append(
                _score_peak(
                    peak,
                    crops_fixed=crops.fixed,
                    crops_moving=crops.moving,
                    gradient_fixed=gradient_fixed,
                    gradient_moving=gradient_moving,
                    crop_expected=crops.integer_expected,
                    hint=hint,
                    params=params,
                )
            )
        scored.sort(key=lambda result: (result.raw_zncc, result.confidence), reverse=True)
        best = scored[0]
        _LOGGER.debug(
            "pair registration status=%s confidence=%.3f dx=%.3f dy=%.3f",
            best.status,
            best.confidence,
            best.transform.dx,
            best.transform.dy,
        )
        return best

    def register_pairs(
        self,
        pairs: tuple[tuple[ImageBuffer, ImageBuffer, PairHint], ...],
    ) -> tuple[RegistrationResult, ...]:
        return tuple(self.register(fixed, moving, hint) for fixed, moving, hint in pairs)


def _informationless(
    fixed: ImageBuffer,
    moving: ImageBuffer,
    params: RegistrationParameters,
) -> RegistrationResult | None:
    if fixed.std < params.empty_std_threshold or moving.std < params.empty_std_threshold:
        return RegistrationResult.skipped(
            status=RegistrationStatus.EMPTY_TILE,
            message="tile intensity variance is below the informationless threshold",
        )
    if mean_gradient(fixed.pixels, params.gaussian_sigma) < params.empty_gradient_mean_threshold:
        return RegistrationResult.skipped(
            status=RegistrationStatus.EMPTY_TILE,
            message="fixed tile gradient magnitude is below the informationless threshold",
        )
    if mean_gradient(moving.pixels, params.gaussian_sigma) < params.empty_gradient_mean_threshold:
        return RegistrationResult.skipped(
            status=RegistrationStatus.EMPTY_TILE,
            message="moving tile gradient magnitude is below the informationless threshold",
        )
    return None


def _merge_candidate_peaks(
    phase_peaks: tuple[PhasePeak, ...],
    zncc_shifts: tuple[Translation2D, ...],
    *,
    template: PhasePeak | None,
) -> tuple[PhasePeak, ...]:
    merged: list[PhasePeak] = list(phase_peaks)
    seen = {(round(peak.translation.dx, 4), round(peak.translation.dy, 4)) for peak in phase_peaks}
    fallback = template or (phase_peaks[0] if phase_peaks else None)
    for shift in zncc_shifts:
        key = (round(shift.dx, 4), round(shift.dy, 4))
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            PhasePeak(
                translation=shift,
                phase_response=fallback.phase_response if fallback is not None else 0.0,
                peak_value=fallback.peak_value if fallback is not None else 0.0,
                second_peak_value=fallback.second_peak_value if fallback is not None else 0.0,
                peak_ratio=fallback.peak_ratio if fallback is not None else 1.0,
            )
        )
    return tuple(merged)


def _score_peak(
    peak: PhasePeak,
    *,
    crops_fixed,
    crops_moving,
    gradient_fixed,
    gradient_moving,
    crop_expected: Translation2D,
    hint: PairHint,
    params: RegistrationParameters,
) -> RegistrationResult:
    residual = peak.translation
    raw_zncc = zncc_at_shift(crops_fixed, crops_moving, residual)
    gradient_zncc = zncc_at_shift(gradient_fixed, gradient_moving, residual)
    dx = crop_expected.dx + residual.dx
    dy = crop_expected.dy + residual.dy
    displacement_error = Translation2D(dx - hint.expected.dx, dy - hint.expected.dy).magnitude
    confidence = compute_confidence(
        phase_response=peak.phase_response,
        peak_ratio=peak.peak_ratio,
        raw_zncc=raw_zncc,
        gradient_zncc=gradient_zncc,
        expected_displacement_error=displacement_error,
        search_radius_px=hint.search_radius_px,
        weights=params.confidence_weights,
    )
    status = RegistrationStatus.OK if confidence >= params.low_confidence_threshold else RegistrationStatus.LOW_CONFIDENCE
    message = "" if status is RegistrationStatus.OK else "confidence below threshold; result is not treated as certain"
    return translation_result(
        dx,
        dy,
        confidence=confidence,
        phase_response=peak.phase_response,
        peak_ratio=peak.peak_ratio,
        raw_zncc=raw_zncc,
        gradient_zncc=gradient_zncc,
        expected_displacement_error=displacement_error,
        status=status,
        method=RegistrationMethod.PHASE_CORRELATION,
        candidates=(peak,),
        message=message,
    )


class PythonRegistrationBackend(RegistrationBackend, PairRegistrar):
    def __init__(self, parameters: RegistrationParameters | None = None) -> None:
        self._registrar = PythonPairRegistrar(parameters)

    def register(self, fixed: ImageBuffer, moving: ImageBuffer, hint: PairHint) -> RegistrationResult:
        return self._registrar.register(fixed, moving, hint)

    def register_pairs(
        self,
        pairs: tuple[tuple[ImageBuffer, ImageBuffer, PairHint], ...],
    ) -> tuple[RegistrationResult, ...]:
        return self._registrar.register_pairs(pairs)
