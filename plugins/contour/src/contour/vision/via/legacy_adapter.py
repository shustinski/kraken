"""Bridge to the existing ``_core`` via pipeline (template + hybrid + blob as configured)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...application.processing import ContourExtractionSettings, normalize_via_search_mode
from ...application.use_cases.processing._core import (
    _detect_via_candidates,
    _iou_nms,
    _via_grayscale,
)
from ..schemas import ViaHit


def run_legacy_core_via(
    image: Any,
    settings: ContourExtractionSettings,
    log: list[str],
    *,
    template_only: bool = False,
) -> list[ViaHit]:
    """Runs the in-product detector; *template_only* matches only the saved OpenCV templates."""

    gray = _via_grayscale(image)
    if gray.size == 0:
        return []
    s: ContourExtractionSettings
    if template_only:
        s = replace(settings, via_search_mode="template")
        log.append("legacy_core: template_only=True")
    else:
        s = settings
        log.append(f"legacy_core: via_search_mode={normalize_via_search_mode(s.via_search_mode)}")
    acc, _rej = _detect_via_candidates(gray, s)
    acc, _dups = _iou_nms(acc, iou_threshold=0.35)
    if template_only:
        acc = [c for c in acc if str(getattr(c, "source", "")) == "template"]
    return [_to_hit(x) for x in acc]


def _to_hit(c: Any) -> ViaHit:
    coverage = max(0.0, min(1.0, float(getattr(c, "roundness", 0.0)) / 100.0))
    return ViaHit(
        center_x=float(c.center_x),
        center_y=float(c.center_y),
        width=float(c.width),
        height=float(c.height),
        score=float(c.score),
        strategy=f"legacy_{getattr(c, 'source', 'unknown')}",
        contrast=float(getattr(c, "contrast", 0.0)),
        edge_strength=float(getattr(c, "edge_strength", 0.0)),
        annulus_coverage=coverage,
        extra={},
    )
