"""Incremental conductor segmentation: contrast/Otsu → morphology → mask."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
    gray: np.ndarray | None = None
    raw_segmentation: np.ndarray | None = None
    gap_bridge_px: int | None = None
    speckle_removal_px: int | None = None
    min_component_area: int | None = None
    after_topology: np.ndarray | None = None
    mask: np.ndarray | None = None


_SEGMENTATION_CACHE: dict[str, _SegmentationStageCache] = {}
_SEGMENTATION_CACHE_MAX_ITEMS = 8


def clear_metal_segmentation_cache() -> None:
    _SEGMENTATION_CACHE.clear()


def _store_cache_entry(image_sig: str, entry: _SegmentationStageCache) -> None:
    if image_sig in _SEGMENTATION_CACHE:
        del _SEGMENTATION_CACHE[image_sig]
    while len(_SEGMENTATION_CACHE) >= _SEGMENTATION_CACHE_MAX_ITEMS:
        oldest = next(iter(_SEGMENTATION_CACHE))
        _SEGMENTATION_CACHE.pop(oldest, None)
    _SEGMENTATION_CACHE[image_sig] = entry


def _invalidate_topology(entry: _SegmentationStageCache) -> None:
    entry.gap_bridge_px = None
    entry.speckle_removal_px = None
    entry.min_component_area = None
    entry.after_topology = None
    entry.mask = None


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
            strategy="legacy_otsu",
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
    if entry.raw_segmentation is None or entry.contrast_bias != contrast_bias:
        raise_if_preview_cancelled()
        # Conductors on SEM are bright features on a darker field; Otsu uses THRESH_BINARY.
        polarity = SemPolarity.BRIGHT_FOREGROUND
        otsu_offset = contrast_bias_to_otsu_offset(contrast_bias)
        raw = otsu_segmentation_mask(g0, otsu_offset=otsu_offset, dark_foreground=False)
        entry.contrast_bias = contrast_bias
        entry.polarity = polarity
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
        strategy="legacy_otsu",
        polarity=entry.polarity,
        debug_images=debug,
    )
