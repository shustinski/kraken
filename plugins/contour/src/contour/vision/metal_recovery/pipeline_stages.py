"""Incremental conductor segmentation: contrast/Otsu → morphology → mask."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from ...utils import ensure_binary_mask, ensure_uint8
from ..schemas import SemPolarity
from .segmentation import (
    MetalSegmentationConfig,
    MetalSegmentationResult,
    apply_topology_repair,
    contrast_bias_to_otsu_offset,
    filter_mask_components,
    normalize_metal_segmentation_strategy,
    otsu_segmentation_mask,
)


def image_signature(gray: np.ndarray) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(gray).tobytes()).hexdigest()
    return f"{int(gray.shape[0])}x{int(gray.shape[1])}:{digest}"


@dataclass(slots=True)
class _SegmentationStageCache:
    image_sig: str = ""
    contrast_bias: float | None = None
    polarity: SemPolarity | None = None
    strategy: str | None = None
    gray: np.ndarray | None = None
    raw_segmentation: np.ndarray | None = None
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
        for array in (entry.gray, entry.raw_segmentation, entry.after_topology, entry.mask):
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


def _adaptive_segmentation_mask(
    gray: np.ndarray,
    *,
    contrast_bias: float,
    dark_foreground: bool,
) -> np.ndarray:
    shortest = max(3, min(gray.shape[:2]))
    block_size = min(63, max(15, (shortest // 16) | 1))
    if block_size >= shortest:
        block_size = max(3, (shortest - 1) | 1)
    mode = cv2.THRESH_BINARY_INV if dark_foreground else cv2.THRESH_BINARY
    c_value = float(-contrast_bias) * 0.12
    return ensure_binary_mask(
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            mode,
            block_size,
            c_value,
        )
    )


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


def _segment(
    gray: np.ndarray,
    *,
    strategy: str,
    contrast_bias: float,
) -> tuple[np.ndarray, SemPolarity, str]:
    otsu_offset = contrast_bias_to_otsu_offset(contrast_bias)
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
                    contrast_bias=contrast_bias,
                    dark_foreground=dark_foreground,
                )
            else:
                mask = otsu_segmentation_mask(
                    gray,
                    otsu_offset=otsu_offset,
                    dark_foreground=dark_foreground,
                )
            candidates.append((_segmentation_quality(gray, mask), mask, polarity, candidate_strategy))
    _score, selected_mask, selected_polarity, selected_strategy = max(candidates, key=lambda item: item[0])
    return selected_mask, selected_polarity, selected_strategy


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

    sig = image_signature(g0)
    entry = _SEGMENTATION_CACHE.get(sig)
    if entry is None or entry.image_sig != sig:
        entry = _SegmentationStageCache(image_sig=sig, gray=g0.copy())
        _store_cache_entry(sig, entry)
    elif entry.gray is None:
        entry.gray = g0.copy()

    contrast_bias = float(config.contrast_bias)
    requested_strategy = normalize_metal_segmentation_strategy(config.segmentation_strategy)
    if (
        entry.raw_segmentation is None
        or entry.contrast_bias != contrast_bias
        or entry.strategy != requested_strategy
    ):
        raise_if_preview_cancelled()
        raw, polarity, selected_strategy = _segment(
            g0,
            strategy=requested_strategy,
            contrast_bias=contrast_bias,
        )
        entry.contrast_bias = contrast_bias
        entry.polarity = polarity
        entry.strategy = selected_strategy
        entry.raw_segmentation = raw
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
        after_topo = apply_topology_repair(entry.raw_segmentation, config)
        mask = filter_mask_components(after_topo, int(config.min_component_area))
        entry.gap_bridge_px = topo_key[0]
        entry.speckle_removal_px = topo_key[1]
        entry.min_component_area = topo_key[2]
        entry.after_topology = after_topo
        entry.mask = mask

    assert entry.gray is not None
    assert entry.raw_segmentation is not None
    assert entry.after_topology is not None
    assert entry.mask is not None
    assert entry.polarity is not None

    debug = {
        "metal_source_gray": entry.gray,
        "metal_preprocessed": entry.gray,
        "metal_raw_segmentation": ensure_binary_mask(entry.raw_segmentation),
        "metal_after_topology": ensure_binary_mask(entry.after_topology),
        "metal_threshold_mask": ensure_binary_mask(entry.raw_segmentation),
    }
    return MetalSegmentationResult(
        mask=ensure_binary_mask(entry.mask),
        preprocessed=entry.gray,
        raw_segmentation=ensure_binary_mask(entry.raw_segmentation),
        after_topology=ensure_binary_mask(entry.after_topology),
        strategy=entry.strategy or requested_strategy,
        polarity=entry.polarity,
        debug_images=debug,
    )
