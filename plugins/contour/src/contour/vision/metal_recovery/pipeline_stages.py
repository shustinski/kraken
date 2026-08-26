"""Incremental conductor segmentation: contrast/Otsu → morphology → mask."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from ...utils import ensure_binary_mask, ensure_uint8
from ..schemas import SemPolarity
from .gradient_watershed import (
    GradientWatershedConfig,
    gradient_watershed_config_from_object,
)
from .seeded_segmentation import seeded_segmentation_mask
from .segmentation import (
    MetalSegmentationConfig,
    MetalSegmentationResult,
    apply_topology_repair,
    filter_mask_components,
    is_seeded_segmentation_strategy,
    normalize_metal_adaptive_method,
    normalize_metal_segmentation_strategy,
    otsu_segmentation_mask,
)
from .strategy_contracts import StrategyUnavailableError
from .strategy_registry import IMPLEMENTED_NEW_STRATEGIES, normalize_strategy_parameters, strategy_spec


def image_signature(gray: np.ndarray) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(gray).tobytes()).hexdigest()
    return f"{int(gray.shape[0])}x{int(gray.shape[1])}:{digest}"


@dataclass(slots=True)
class _SegmentationStageCache:
    image_sig: str = ""
    polarity: SemPolarity | None = None
    requested_strategy: str | None = None
    structural_variant: str | None = None
    extra_debug: dict[str, np.ndarray] | None = None
    instance_labels: np.ndarray | None = None
    watershed_key: (
        tuple[float, float, float, int, int, int, float, float, int, int, int, float, float] | None
    ) = None
    adaptive_key: tuple[int, float, str] | None = None
    strategy: str | None = None
    gray: np.ndarray | None = None
    raw_segmentation: np.ndarray | None = None
    min_contrast: float | None = None
    contrast_filtered: np.ndarray | None = None
    gap_bridge_px: int | None = None
    speckle_removal_px: int | None = None
    min_component_area: int | None = None
    after_topology: np.ndarray | None = None
    mask: np.ndarray | None = None


_SEGMENTATION_CACHE: dict[str, _SegmentationStageCache] = {}
_SEGMENTATION_CACHE_MAX_ITEMS = 8
_SEGMENTATION_CACHE_MAX_BYTES = 256 * 1024 * 1024


def clear_metal_segmentation_cache() -> None:
    _SEGMENTATION_CACHE.clear()


def _store_cache_entry(image_sig: str, entry: _SegmentationStageCache) -> None:
    _SEGMENTATION_CACHE.pop(image_sig, None)
    while len(_SEGMENTATION_CACHE) >= _SEGMENTATION_CACHE_MAX_ITEMS:
        oldest = next(iter(_SEGMENTATION_CACHE))
        _SEGMENTATION_CACHE.pop(oldest, None)
    _SEGMENTATION_CACHE[image_sig] = entry
    while len(_SEGMENTATION_CACHE) > 1 and _segmentation_cache_bytes() > _SEGMENTATION_CACHE_MAX_BYTES:
        oldest = next(iter(_SEGMENTATION_CACHE))
        _SEGMENTATION_CACHE.pop(oldest, None)


def _segmentation_cache_bytes() -> int:
    seen: set[int] = set()
    total = 0
    for entry in _SEGMENTATION_CACHE.values():
        for array in (
            entry.gray,
            entry.raw_segmentation,
            entry.contrast_filtered,
            entry.after_topology,
            entry.mask,
        ):
            if array is not None and id(array) not in seen:
                seen.add(id(array))
                total += int(array.nbytes)
    return total


def _invalidate_topology(entry: _SegmentationStageCache) -> None:
    entry.gap_bridge_px = None
    entry.speckle_removal_px = None
    entry.min_component_area = None
    entry.after_topology = None
    entry.mask = None


def _invalidate_contrast(entry: _SegmentationStageCache) -> None:
    entry.min_contrast = None
    entry.contrast_filtered = None
    _invalidate_topology(entry)


def _adaptive_block_size(gray: np.ndarray, requested: int) -> int:
    shortest = max(3, min(gray.shape[:2]))
    if requested <= 0:
        block_size = min(63, max(15, (shortest // 16) | 1))
    else:
        block_size = max(3, int(requested) | 1)
    if block_size >= shortest:
        block_size = max(3, (shortest - 1) | 1)
    return int(block_size)


def _adaptive_segmentation_mask(
    gray: np.ndarray,
    *,
    dark_foreground: bool,
    block_size: int = 0,
    c: float = 0.0,
    method: str = "gaussian",
) -> np.ndarray:
    window = _adaptive_block_size(gray, block_size)
    mode = cv2.THRESH_BINARY_INV if dark_foreground else cv2.THRESH_BINARY
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_MEAN_C
        if normalize_metal_adaptive_method(method) == "mean"
        else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    )
    return ensure_binary_mask(
        cv2.adaptiveThreshold(
            gray,
            255,
            adaptive_method,
            mode,
            window,
            float(c),
        )
    )


def render_gradient_field_bgr(gradient_x: np.ndarray, gradient_y: np.ndarray) -> np.ndarray:
    """Encode Sobel direction as hue and magnitude as brightness (no baked arrows)."""
    height, width = int(gradient_x.shape[0]), int(gradient_x.shape[1])
    if height <= 0 or width <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    angle = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)
    hsv = np.empty((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.clip(angle * 0.5, 0, 179).astype(np.uint8)
    hsv[:, :, 1] = 255
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    if peak <= 1e-6:
        hsv[:, :, 2] = 0
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hsv[:, :, 2] = np.clip(magnitude * (255.0 / peak), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _filter_instance_labels_by_area(
    labels: np.ndarray,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop small instance IDs without merging neighbouring labels."""

    if labels.size == 0 or not np.any(labels > 0):
        empty = np.zeros(labels.shape, dtype=np.uint8)
        return labels.astype(np.int32, copy=False), empty
    max_id = int(labels.max())
    areas = np.bincount(np.clip(labels, 0, None).ravel(), minlength=max_id + 1)
    keep = areas >= max(1, int(min_area))
    keep[0] = False
    filtered = np.where(keep[np.clip(labels, 0, None)], labels, 0).astype(np.int32)
    mask = np.where(filtered > 0, 255, 0).astype(np.uint8)
    return filtered, mask


def axis_gradient_debug_images(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Sobel ∂I/∂x, ∂I/∂y and vector-field preview for debug overlay."""
    source = ensure_uint8(gray)
    if source.ndim != 2 or source.size == 0:
        height, width = (source.shape[:2] if source.ndim >= 2 else (0, 0))
        empty = np.zeros((height, width), dtype=np.uint8)
        empty_f32 = np.zeros((height, width), dtype=np.float32)
        empty_field = np.zeros((height, width, 3), dtype=np.uint8)
        return {
            "metal_gradient_x": empty,
            "metal_gradient_y": empty,
            "metal_gradient_x_f32": empty_f32,
            "metal_gradient_y_f32": empty_f32,
            "metal_gradient_field": empty_field,
        }
    gradient_x = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "metal_gradient_x": cv2.convertScaleAbs(gradient_x),
        "metal_gradient_y": cv2.convertScaleAbs(gradient_y),
        "metal_gradient_x_f32": gradient_x,
        "metal_gradient_y_f32": gradient_y,
        "metal_gradient_field": render_gradient_field_bgr(gradient_x, gradient_y),
    }


def _segmentation_quality(gray: np.ndarray, mask: np.ndarray) -> float:
    """Prefer connected, edge-aligned masks without near-empty/full flooding."""
    binary = (mask > 0).astype(np.uint8)
    fill = float(binary.mean())
    if fill <= 0.002 or fill >= 0.998:
        return -1_000.0
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int32)
    tiny_fraction = float(np.count_nonzero(areas < 12)) / max(1, len(areas))
    largest_share = float(areas.max()) / max(1.0, float(areas.sum())) if len(areas) else 0.0
    boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    edge_agreement = float(gradient[boundary].mean()) / (float(gradient.mean()) + 1e-6) if boundary.any() else 0.0
    fill_penalty = max(0.0, abs(fill - 0.35) - 0.30) * 4.0
    return edge_agreement + 0.35 * largest_share - 0.55 * tiny_fraction - fill_penalty


def _watershed_cache_key(
    config: GradientWatershedConfig,
) -> tuple[float, float, float, int, int, int, float, float, int, int, int, float, float]:
    return (
        float(config.smoothing_sigma),
        float(config.core_margin),
        float(config.groove_margin),
        int(config.rim_probe_px),
        int(config.seed_speckle_px),
        int(config.valley_span_px),
        float(config.valley_depth),
        float(config.random_walker_beta),
        int(config.random_walker_iterations),
        int(config.graph_cut_iterations),
        int(config.reconstruction_erode_px),
        float(config.boundary_relief),
        float(config.boundary_background_sigma),
    )


def _segment(
    gray: np.ndarray,
    *,
    strategy: str,
    watershed_config: GradientWatershedConfig | None = None,
    adaptive_block_size: int = 0,
    adaptive_c: float = 0.0,
    adaptive_method: str = "gaussian",
) -> tuple[np.ndarray, SemPolarity, str]:
    if is_seeded_segmentation_strategy(strategy):
        mask = seeded_segmentation_mask(
            gray,
            strategy,
            watershed_config or GradientWatershedConfig(),
        )
        return mask, SemPolarity.BRIGHT_FOREGROUND, strategy
    strategies = ("legacy_otsu", "local_adaptive") if strategy == "auto" else (strategy,)
    candidates: list[tuple[float, np.ndarray, SemPolarity, str]] = []
    for candidate_strategy in strategies:
        polarities = (
            (
                (False, SemPolarity.BRIGHT_FOREGROUND),
                (True, SemPolarity.DARK_FOREGROUND),
            )
            if strategy == "auto"
            else ((False, SemPolarity.BRIGHT_FOREGROUND),)
        )
        for dark_foreground, polarity in polarities:
            if candidate_strategy == "local_adaptive":
                mask = _adaptive_segmentation_mask(
                    gray,
                    dark_foreground=dark_foreground,
                    block_size=adaptive_block_size,
                    c=adaptive_c,
                    method=adaptive_method,
                )
            else:
                mask = otsu_segmentation_mask(
                    gray,
                    otsu_offset=0.0,
                    dark_foreground=dark_foreground,
                )
            candidates.append((_segmentation_quality(gray, mask), mask, polarity, candidate_strategy))
    _score, selected_mask, selected_polarity, selected_strategy = max(candidates, key=lambda item: item[0])
    return selected_mask, selected_polarity, selected_strategy


def _apply_minimum_contrast(
    gray: np.ndarray,
    raw_mask: np.ndarray,
    *,
    polarity: SemPolarity,
    min_contrast: float,
) -> np.ndarray:
    """Build pixel candidates from the requested source-to-background contrast."""
    threshold = max(1.0, min(255.0, float(min_contrast)))
    mask = ensure_binary_mask(raw_mask)
    foreground = mask > 0
    background_values = gray[~foreground]
    if background_values.size == 0:
        return np.zeros_like(mask)
    background = float(np.median(background_values))
    gray_float = gray.astype(np.float32, copy=False)
    # A small blur stabilizes weak, textured conductors.  Keep the stronger of
    # the source and smoothed responses so thin high-contrast traces are not
    # weakened by smoothing.
    smoothed = cv2.GaussianBlur(gray, (9, 9), 0).astype(np.float32, copy=False)
    if polarity == SemPolarity.DARK_FOREGROUND:
        contrast = np.maximum(background - gray_float, background - smoothed)
    else:
        contrast = np.maximum(gray_float - background, smoothed - background)
    return np.where(contrast >= threshold, 255, 0).astype(np.uint8)


def _retain_seeded_contrast_components(
    contrast_mask: np.ndarray,
    raw_mask: np.ndarray,
    *,
    min_seed_pixels: int = 3,
) -> np.ndarray:
    """Keep weak-contrast regions connected to a reliable segmentation seed."""
    candidate = ensure_binary_mask(contrast_mask)
    component_count, labels = cv2.connectedComponents(
        (candidate > 0).astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return candidate
    seed_counts = np.bincount(
        labels.ravel(),
        weights=(ensure_binary_mask(raw_mask) > 0).ravel(),
        minlength=component_count,
    )
    keep = seed_counts >= max(1, int(min_seed_pixels))
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def build_metal_segmentation_mask_staged(
    gray: np.ndarray,
    config: MetalSegmentationConfig,
) -> MetalSegmentationResult:
    """Run segmentation stages with per-image cache (later stages reuse earlier work)."""
    g0 = ensure_uint8(gray)
    if g0.size == 0:
        z = np.zeros_like(g0)
        return MetalSegmentationResult(
            mask=z,
            preprocessed=z,
            raw_segmentation=z,
            after_topology=z,
            strategy=normalize_metal_segmentation_strategy(config.segmentation_strategy),
            polarity=SemPolarity.AUTO,
        )

    requested_strategy = normalize_metal_segmentation_strategy(config.segmentation_strategy)
    if requested_strategy in IMPLEMENTED_NEW_STRATEGIES:
        from .features import build_metal_structural_features

        parameters = normalize_strategy_parameters(
            requested_strategy,
            config.strategy_parameters.get(requested_strategy),
        )
        feature_started = perf_counter()
        features = build_metal_structural_features(
            g0,
            smoothing_sigma=float(parameters.get("contour_smoothing_sigma", 1.0)),
            orientation_smoothing_sigma=float(parameters.get("orientation_smoothing_sigma", 2.0)),
        )
        feature_build_ms = (perf_counter() - feature_started) * 1000.0
        backend = strategy_spec(requested_strategy).load_backend()
        if backend is None:
            raise StrategyUnavailableError(f"{requested_strategy} backend is unavailable")
        strategy_result = backend(g0, features, parameters)
        timings_ms = dict(strategy_result.timings_ms)
        backend_total_ms = float(timings_ms.get("total", 0.0))
        timings_ms["feature_build"] = feature_build_ms
        timings_ms["total"] = feature_build_ms + backend_total_ms
        instance_labels = strategy_result.instance_labels
        if instance_labels is not None:
            instance_labels, mask = _filter_instance_labels_by_area(
                instance_labels,
                int(config.min_component_area),
            )
        else:
            mask = filter_mask_components(strategy_result.binary_mask, int(config.min_component_area))
        debug = {
            "metal_source_gray": g0,
            "metal_preprocessed": g0,
            "metal_raw_segmentation": ensure_binary_mask(strategy_result.binary_mask),
            "metal_after_topology": ensure_binary_mask(mask),
            "metal_threshold_mask": ensure_binary_mask(strategy_result.binary_mask),
            **axis_gradient_debug_images(g0),
            **strategy_result.debug_images,
        }
        return MetalSegmentationResult(
            mask=ensure_binary_mask(mask),
            preprocessed=g0,
            raw_segmentation=ensure_binary_mask(strategy_result.binary_mask),
            after_topology=ensure_binary_mask(mask),
            strategy=requested_strategy,
            polarity=SemPolarity.BRIGHT_FOREGROUND,
            debug_images=debug,
            instance_labels=instance_labels,
            boundary_map=strategy_result.boundary_map,
            confidence_map=strategy_result.confidence_map,
            debug_data=dict(strategy_result.debug_data),
            timings_ms=timings_ms,
        )

    sig = image_signature(g0)
    entry = _SEGMENTATION_CACHE.get(sig)
    if entry is None or entry.image_sig != sig:
        entry = _SegmentationStageCache(image_sig=sig, gray=g0.copy())
        _store_cache_entry(sig, entry)
    elif entry.gray is None:
        entry.gray = g0.copy()

    watershed_config = gradient_watershed_config_from_object(config)
    watershed_key = _watershed_cache_key(watershed_config)
    structural_variant = str(getattr(config, "structural_variant", "s2") or "s2")
    adaptive_key = (
        max(0, min(255, int(config.adaptive_block_size))),
        float(config.adaptive_c),
        normalize_metal_adaptive_method(config.adaptive_method),
    )
    if (
        entry.raw_segmentation is None
        or entry.requested_strategy != requested_strategy
        or entry.structural_variant != structural_variant
        or entry.watershed_key != watershed_key
        or entry.adaptive_key != adaptive_key
    ):
        raise_if_preview_cancelled()
        extra_debug: dict[str, np.ndarray] = {}
        if requested_strategy == "structural_watershed":
            from .structural_watershed import (
                run_structural_watershed,
                structural_watershed_config_from_object,
            )

            structural_result = run_structural_watershed(
                g0,
                watershed_config,
                structural_watershed_config_from_object(config),
            )
            raw = structural_result.mask
            polarity = SemPolarity.BRIGHT_FOREGROUND
            selected_strategy = "structural_watershed"
            extra_debug = dict(structural_result.debug_images)
            instance_labels = structural_result.instance_labels
        else:
            raw, polarity, selected_strategy = _segment(
                g0,
                strategy=requested_strategy,
                watershed_config=watershed_config,
                adaptive_block_size=adaptive_key[0],
                adaptive_c=adaptive_key[1],
                adaptive_method=adaptive_key[2],
            )
            instance_labels = None
        entry.polarity = polarity
        entry.requested_strategy = requested_strategy
        entry.structural_variant = structural_variant
        entry.watershed_key = watershed_key
        entry.adaptive_key = adaptive_key
        entry.strategy = selected_strategy
        entry.raw_segmentation = raw
        entry.extra_debug = extra_debug
        entry.instance_labels = instance_labels
        _invalidate_contrast(entry)

    assert entry.polarity is not None
    min_contrast = max(1.0, min(255.0, float(config.min_contrast)))
    if entry.contrast_filtered is None or entry.min_contrast != min_contrast:
        raise_if_preview_cancelled()
        if is_seeded_segmentation_strategy(requested_strategy):
            # Seeded algorithms already decided metal vs substrate per region; a
            # global contrast cut would discard the mid-grey fill they recovered.
            entry.contrast_filtered = ensure_binary_mask(entry.raw_segmentation)
        else:
            entry.contrast_filtered = _apply_minimum_contrast(
                g0,
                entry.raw_segmentation,
                polarity=entry.polarity,
                min_contrast=min_contrast,
            )
        entry.min_contrast = min_contrast
        _invalidate_topology(entry)

    topo_key = (
        int(config.gap_bridge_px),
        int(config.speckle_removal_px),
        int(config.min_component_area),
    )
    if (
        entry.after_topology is None
        or entry.gap_bridge_px != topo_key[0]
        or entry.speckle_removal_px != topo_key[1]
        or entry.min_component_area != topo_key[2]
    ):
        raise_if_preview_cancelled()
        from .structural_watershed import INSTANCE_IDENTITY_VARIANTS

        if (
            requested_strategy == "structural_watershed"
            and structural_variant in INSTANCE_IDENTITY_VARIANTS
            and entry.instance_labels is not None
        ):
            filtered_labels, after_topo = _filter_instance_labels_by_area(
                entry.instance_labels,
                int(config.min_component_area),
            )
            entry.instance_labels = filtered_labels
            mask = after_topo
        else:
            after_topo = apply_topology_repair(entry.contrast_filtered, config)
            after_topo = _retain_seeded_contrast_components(
                after_topo,
                entry.raw_segmentation,
            )
            mask = filter_mask_components(after_topo, int(config.min_component_area))
        entry.gap_bridge_px = topo_key[0]
        entry.speckle_removal_px = topo_key[1]
        entry.min_component_area = topo_key[2]
        entry.after_topology = after_topo
        entry.mask = mask

    assert entry.gray is not None
    assert entry.raw_segmentation is not None
    assert entry.contrast_filtered is not None
    assert entry.after_topology is not None
    assert entry.mask is not None
    assert entry.polarity is not None

    debug = {
        "metal_source_gray": entry.gray,
        "metal_preprocessed": entry.gray,
        "metal_raw_segmentation": ensure_binary_mask(entry.raw_segmentation),
        "metal_after_topology": ensure_binary_mask(entry.after_topology),
        "metal_threshold_mask": ensure_binary_mask(entry.contrast_filtered),
        **axis_gradient_debug_images(entry.gray),
        **(entry.extra_debug or {}),
    }
    return MetalSegmentationResult(
        mask=ensure_binary_mask(entry.mask),
        preprocessed=entry.gray,
        raw_segmentation=ensure_binary_mask(entry.raw_segmentation),
        after_topology=ensure_binary_mask(entry.after_topology),
        strategy=entry.strategy or requested_strategy,
        polarity=entry.polarity,
        debug_images=debug,
        instance_labels=entry.instance_labels,
    )
