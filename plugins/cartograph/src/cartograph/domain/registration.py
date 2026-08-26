"""Pairwise registration contracts. Results always carry diagnostics, never a bare (dx, dy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .coordinates import GridCoordinate, LocalTransform, Translation2D, TransformKind
from .tiles import ImageBuffer


class RegistrationStatus(StrEnum):
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    EMPTY_TILE = "empty_tile"
    FAILED = "failed"
    SKIPPED = "skipped"


class RegistrationMethod(StrEnum):
    PHASE_CORRELATION = "phase_correlation"
    FEATURE_FALLBACK = "feature_fallback"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ConfidenceWeights:
    """Configurable confidence mix. All weights must be non-negative."""

    peak_ratio: float = 0.25
    raw_zncc: float = 0.30
    gradient_zncc: float = 0.25
    phase_response: float = 0.20
    expected_displacement_error: float = 0.15

    def __post_init__(self) -> None:
        for name in (
            "peak_ratio",
            "raw_zncc",
            "gradient_zncc",
            "phase_response",
            "expected_displacement_error",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"confidence weight {name} must be non-negative")

    @property
    def positive_sum(self) -> float:
        return self.peak_ratio + self.raw_zncc + self.gradient_zncc + self.phase_response


@dataclass(frozen=True, slots=True)
class RegistrationParameters:
    """Inputs that affect registration; hashed for cache invalidation."""

    search_radius_px: float = 24.0
    overlap_margin_px: int = 8
    top_k: int = 5
    min_overlap_px: int = 16
    empty_std_threshold: float = 1.0
    empty_gradient_mean_threshold: float = 0.5
    low_confidence_threshold: float = 0.45
    cycle_residual_threshold_px: float = 2.0
    huber_delta_px: float = 2.0
    huber_iterations: int = 8
    gaussian_sigma: float = 0.8
    min_diagonal_overlap_px: int = 24
    include_diagonals: bool = False
    confidence_weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    schema_version: str = "cartograph.registration.v1"

    def __post_init__(self) -> None:
        if self.search_radius_px <= 0.0:
            raise ValueError("search_radius_px must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.min_overlap_px < 4:
            raise ValueError("min_overlap_px must be >= 4")
        if self.confidence_weights.positive_sum <= 0.0:
            raise ValueError("at least one positive confidence weight is required")

    def to_fingerprint_dict(self) -> dict[str, float | int | bool | str]:
        weights = self.confidence_weights
        return {
            "schema_version": self.schema_version,
            "search_radius_px": self.search_radius_px,
            "overlap_margin_px": self.overlap_margin_px,
            "top_k": self.top_k,
            "min_overlap_px": self.min_overlap_px,
            "empty_std_threshold": self.empty_std_threshold,
            "empty_gradient_mean_threshold": self.empty_gradient_mean_threshold,
            "low_confidence_threshold": self.low_confidence_threshold,
            "cycle_residual_threshold_px": self.cycle_residual_threshold_px,
            "huber_delta_px": self.huber_delta_px,
            "huber_iterations": self.huber_iterations,
            "gaussian_sigma": self.gaussian_sigma,
            "min_diagonal_overlap_px": self.min_diagonal_overlap_px,
            "include_diagonals": self.include_diagonals,
            "w_peak_ratio": weights.peak_ratio,
            "w_raw_zncc": weights.raw_zncc,
            "w_gradient_zncc": weights.gradient_zncc,
            "w_phase_response": weights.phase_response,
            "w_expected_displacement_error": weights.expected_displacement_error,
        }


@dataclass(frozen=True, slots=True)
class PairHint:
    """Geometrically expected displacement of moving origin relative to fixed origin."""

    expected: Translation2D
    search_radius_px: float
    overlap_margin_px: int = 8


@dataclass(frozen=True, slots=True)
class PhasePeak:
    translation: Translation2D
    phase_response: float
    peak_value: float
    second_peak_value: float
    peak_ratio: float


@dataclass(frozen=True, slots=True)
class RegistrationScore:
    phase_response: float
    peak_ratio: float
    raw_zncc: float
    gradient_zncc: float
    expected_displacement_error: float
    confidence: float


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    transform: LocalTransform
    confidence: float
    phase_response: float
    peak_ratio: float
    raw_zncc: float
    gradient_zncc: float
    expected_displacement_error: float
    cycle_residual: float | None
    method: RegistrationMethod
    status: RegistrationStatus
    candidates: tuple[PhasePeak, ...] = ()
    message: str = ""
    source: GridCoordinate | None = None
    target: GridCoordinate | None = None

    @property
    def is_usable(self) -> bool:
        return self.status in {RegistrationStatus.OK, RegistrationStatus.LOW_CONFIDENCE}

    @classmethod
    def skipped(
        cls,
        *,
        status: RegistrationStatus,
        message: str,
        method: RegistrationMethod = RegistrationMethod.SKIPPED,
        source: GridCoordinate | None = None,
        target: GridCoordinate | None = None,
    ) -> RegistrationResult:
        return cls(
            transform=LocalTransform.translation(0.0, 0.0),
            confidence=0.0,
            phase_response=0.0,
            peak_ratio=0.0,
            raw_zncc=0.0,
            gradient_zncc=0.0,
            expected_displacement_error=0.0,
            cycle_residual=None,
            method=method,
            status=status,
            message=message,
            source=source,
            target=target,
        )


def peak_uniqueness(peak_ratio: float) -> float:
    """Map peak/second-peak ratio to [0, 1]. Ratio < 1 means the 'peak' is not first."""

    if peak_ratio <= 1.0:
        return 0.0
    return min(1.0, 1.0 - 1.0 / peak_ratio)


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def compute_confidence(
    *,
    phase_response: float,
    peak_ratio: float,
    raw_zncc: float,
    gradient_zncc: float,
    expected_displacement_error: float,
    search_radius_px: float,
    weights: ConfidenceWeights,
) -> float:
    """Deterministic confidence in [0, 1]. Formula is fully defined by ``weights``."""

    uniqueness = peak_uniqueness(peak_ratio)
    positive = (
        weights.peak_ratio * uniqueness
        + weights.raw_zncc * max(0.0, raw_zncc)
        + weights.gradient_zncc * max(0.0, gradient_zncc)
        + weights.phase_response * clamp01(phase_response)
    )
    radius = max(float(search_radius_px), 1.0)
    penalty = weights.expected_displacement_error * min(1.0, max(0.0, expected_displacement_error) / radius)
    return clamp01((positive - penalty) / weights.positive_sum)


class PairRegistrar(Protocol):
    """Coarse-grained pair API. A native backend can replace the Python implementation."""

    def register(self, fixed: ImageBuffer, moving: ImageBuffer, hint: PairHint) -> RegistrationResult:
        """Return translation of the moving origin relative to the fixed origin."""


class FeaturePairRegistrar(Protocol):
    """Optional fallback used only when phase-correlation confidence or cycle residual fails."""

    def register(self, fixed: ImageBuffer, moving: ImageBuffer, hint: PairHint) -> RegistrationResult:
        """Feature/RANSAC pair registration. Not implemented in Cartograph v1."""


class RegistrationBackend(Protocol):
    """Batch pair API so a later NativeRegistrationBackend can amortize work."""

    def register_pairs(
        self,
        pairs: tuple[tuple[ImageBuffer, ImageBuffer, PairHint], ...],
    ) -> tuple[RegistrationResult, ...]:
        """Register many pairs. Order of results matches ``pairs``."""


def translation_result(
    dx: float,
    dy: float,
    *,
    confidence: float,
    phase_response: float,
    peak_ratio: float,
    raw_zncc: float,
    gradient_zncc: float,
    expected_displacement_error: float,
    status: RegistrationStatus,
    method: RegistrationMethod = RegistrationMethod.PHASE_CORRELATION,
    candidates: tuple[PhasePeak, ...] = (),
    message: str = "",
    cycle_residual: float | None = None,
    source: GridCoordinate | None = None,
    target: GridCoordinate | None = None,
) -> RegistrationResult:
    return RegistrationResult(
        transform=LocalTransform(TransformKind.TRANSLATION, dx, dy),
        confidence=confidence,
        phase_response=phase_response,
        peak_ratio=peak_ratio,
        raw_zncc=raw_zncc,
        gradient_zncc=gradient_zncc,
        expected_displacement_error=expected_displacement_error,
        cycle_residual=cycle_residual,
        method=method,
        status=status,
        candidates=candidates,
        message=message,
        source=source,
        target=target,
    )
