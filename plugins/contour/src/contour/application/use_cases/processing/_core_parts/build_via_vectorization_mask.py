"""_core: build via vectorization mask."""
from __future__ import annotations
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
import numpy as np
from contour.domain import PolygonData
from contour.pipeline import PreprocessingPipeline
from contour.utils import ensure_binary_mask
from contour.application.preview_cancellation import raise_if_preview_cancelled
from contour.application.processing import (
    ContourDebugCandidate,
    ContourExtractionSettings,
)

from contour.application.use_cases.processing._core_parts.module_state import (
    _VIA_DETECTION_CACHE,
    _VIA_DETECTION_CACHE_MAX_ITEMS,
)

from contour.application.use_cases.processing._core_parts.build_modern_via_vectorization import _build_modern_via_vectorization_mask

from contour.application.use_cases.processing._core_parts.refine_conductor_mask_by import (
    _refine_conductor_mask_by_gradient,
    _via_grayscale,
)

@dataclass(frozen=True, slots=True)
class PreviewProcessingRequest:
    image_path: str
    pipeline_config: dict[str, Any]
    contour_settings: ContourExtractionSettings
    source_image: Any | None = None
    preprocessed_image: Any | None = None
    passthrough_polygons: tuple[PolygonData, ...] | None = None

@dataclass(frozen=True, slots=True)
class PreparedImageRequest:
    image_path: str
    source_image: Any
    pipeline_config: dict[str, Any]
    queued_at: float = field(default_factory=perf_counter, repr=False, compare=False)

@dataclass(frozen=True, slots=True)
class _ViaCandidate:
    center_x: float
    center_y: float
    width: float
    height: float
    score: float
    source: str
    roundness: float = 100.0
    contrast: float = 0.0
    edge_strength: float = 0.0
    reason: str = ""

def _make_image_signature(gray: np.ndarray) -> str:
    digest = hashlib.sha1(gray.tobytes()).hexdigest()
    return f"{gray.shape[0]}x{gray.shape[1]}:{digest}"

def _bounded_cache_set(key: tuple[str, str, str], value: Any) -> None:
    if key in _VIA_DETECTION_CACHE:
        _VIA_DETECTION_CACHE[key] = value
        return
    if len(_VIA_DETECTION_CACHE) >= _VIA_DETECTION_CACHE_MAX_ITEMS:
        oldest = next(iter(_VIA_DETECTION_CACHE))
        _VIA_DETECTION_CACHE.pop(oldest, None)
    _VIA_DETECTION_CACHE[key] = value

def build_preview_request_signature(request: PreviewProcessingRequest) -> tuple[str, str, str, int]:
    return (
        request.image_path,
        json.dumps(request.pipeline_config, ensure_ascii=False, sort_keys=True),
        json.dumps(request.contour_settings.to_dict(), ensure_ascii=False, sort_keys=True),
        len(request.passthrough_polygons or ()),
    )

def build_prepared_image_signature(request: PreparedImageRequest) -> tuple[str, str]:
    return (
        request.image_path,
        json.dumps(request.pipeline_config, ensure_ascii=False, sort_keys=True),
    )

def prepare_image_for_preview(
    source_image: Any,
    pipeline_config: dict[str, Any],
    *,
    step_timing_callback: Callable[[int, str, float], None] | None = None,
) -> Any:
    raise_if_preview_cancelled()
    return PreprocessingPipeline.from_dict(pipeline_config).apply(
        source_image,
        timing_callback=step_timing_callback,
    )

def _recognition_image(source_image: Any, preprocessed_image: Any | None) -> Any:
    if preprocessed_image is not None:
        return preprocessed_image
    return source_image

def apply_via_vectorization_mask(image: Any, settings: ContourExtractionSettings) -> Any:
    mask, _debug_candidates = build_via_vectorization_mask(image, settings)
    return mask

def build_conductor_vectorization_mask(
    source_image: Any,
    preprocessed_image: Any,
    settings: ContourExtractionSettings,
) -> np.ndarray:
    recognition_image = _recognition_image(source_image, preprocessed_image)
    base_mask = ensure_binary_mask(recognition_image)
    legacy = str(getattr(settings, "algorithm_backend", "legacy")).lower() == "legacy"
    conductor_polygon = (
        settings.object_type == "conductor" and str(getattr(settings, "output_mode", "polygon")) != "box"
    )
    # Legacy conductors: binary mask from the pipeline only, then findContours — no gradient refinement.
    if legacy and conductor_polygon:
        return base_mask
    if settings.object_type == "via" or settings.output_mode == "box" or not settings.conductor_gradient_enabled:
        return base_mask
    return _refine_conductor_mask_by_gradient(recognition_image, base_mask, settings)

def build_via_vectorization_mask(
    image: Any, settings: ContourExtractionSettings
) -> tuple[Any, list[ContourDebugCandidate]]:
    """Detect via candidates and render them into a binary mask.

    Dispatches only to the selected modern mode: saved-template matching or
    heuristic local analysis. Old blob/top-hat/DoG detector stacks are not part
    of this runtime path.
    """

    if settings.object_type != "via" and settings.output_mode != "box":
        return ensure_binary_mask(image), []

    gray = _via_grayscale(image)
    if gray.size == 0:
        return ensure_binary_mask(image), []

    return _build_modern_via_vectorization_mask(gray, settings)
