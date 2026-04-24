"""
Composite via detection: new primary (multi-cue + NMS) + optional legacy (existing core).

*Migration note:* the legacy path delegates to
``application.use_cases.processing._core._detect_via_candidates`` so behaviour
stays bit-compatible with 3.x/4.x releases until the shared module is split out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..schemas import AppMode, ImageRef, OutputShapeKind, ViaDetectionOutput, ViaHit
from .legacy_adapter import run_legacy_core_via
from .primary_sem import sem_primary_hits

try:
    from ...application.processing import ContourExtractionSettings, normalize_via_search_mode
except ImportError:  # pragma: no cover
    ContourExtractionSettings = Any  # type: ignore[assignment]
    normalize_via_search_mode = None  # type: ignore[assignment]


class ViaStrategyName(StrEnum):
    SEM_PRIMARY = "sem_primary"  # new: LoG/ring/radial cues (no "circularity only")
    LEGACY_CORE = "legacy_core"  # existing pipeline from _core
    TEMPLATE_PREFERRED = "template_preferred"  # legacy with template-only or hybrid, ordered by score


@dataclass(frozen=True, slots=True)
class ViaRunConfig:
    """What the UI exposes: mode + one preset id; the rest is internal."""

    use_legacy_core: bool = False
    """If true, use only :meth:`run_legacy_core_via` (for A/B and regression)."""

    prefer_template_when_available: bool = True
    """Merge legacy with template pass when user saved templates."""


@dataclass(slots=True)
class CompositeViaDetector:
    image_ref: ImageRef
    config: ViaRunConfig = field(default_factory=ViaRunConfig)

    def run(
        self,
        gray: Any,
        *,
        shape: OutputShapeKind,
        legacy_settings: ContourExtractionSettings,
    ) -> ViaDetectionOutput:
        """``legacy_settings`` is the existing dataclass; keeps persistence paths unchanged."""

        log: list[str] = []
        hits: list[ViaHit] = []
        mode_used = ViaStrategyName.SEM_PRIMARY

        if self.config.use_legacy_core:
            hits = run_legacy_core_via(gray, legacy_settings, log)
            mode_used = ViaStrategyName.LEGACY_CORE
        else:
            primary = sem_primary_hits(gray, legacy_settings, log)
            if self.config.prefer_template_when_available and bool(
                getattr(legacy_settings, "via_template_images", None)
            ):
                if normalize_via_search_mode is not None:
                    mode = normalize_via_search_mode(legacy_settings.via_search_mode)
                else:
                    mode = str(legacy_settings.via_search_mode or "hybrid")
                tmpl: list[ViaHit] = []
                if mode in {"hybrid", "template"}:
                    tmpl = run_legacy_core_via(gray, legacy_settings, log, template_only=True)
                if tmpl:
                    hits = self._merge_strategies(primary, tmpl, iou=0.4)
                    mode_used = ViaStrategyName.TEMPLATE_PREFERRED
                else:
                    hits = primary
            else:
                hits = primary
            if not hits and not self.config.use_legacy_core:
                log.append("primary_empty: falling back to legacy_core")
                hits = run_legacy_core_via(gray, legacy_settings, log)
                mode_used = ViaStrategyName.LEGACY_CORE

        return ViaDetectionOutput(
            image=self.image_ref,
            mode=AppMode.VIA,
            output_kind=shape,
            hits=hits,
            selected_strategy=str(mode_used),
            attempt_log=list(log),
        )

    @staticmethod
    def _merge_strategies(a: list[ViaHit], b: list[ViaHit], *, iou: float) -> list[ViaHit]:
        combined = list(a) + list(b)
        combined.sort(key=lambda h: h.score, reverse=True)
        kept: list[ViaHit] = []
        for h in combined:
            box = h.to_axis_aligned_box()
            if any(_iou(box, k.to_axis_aligned_box()) >= iou for k in kept):
                continue
            kept.append(h)
        return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ar, ab = ax + aw, ay + ah
    br, bb = bx + bw, by + bh
    iw = max(0, min(ar, br) - max(ax, bx))
    ih = max(0, min(ab, bb) - max(ay, by))
    inter = float(iw * ih)
    union = float(aw * ah + bw * bh) - inter + 1e-6
    return inter / union
