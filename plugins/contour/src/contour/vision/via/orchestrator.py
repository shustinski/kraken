"""
Composite via detection: «По шаблону» (matchTemplate) или эвристический pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from ..schemas import AppMode, ImageRef, OutputShapeKind, ViaDetectionOutput, ViaHit
from ..via_detection.heuristic_detector import detect_vias_heuristic
from ..via_detection.result import DetectionResult
from ..via_detection.settings_bridge import (
    fixed_via_diameters_from_settings,
    heuristic_config_from_settings,
    template_config_from_settings,
)
from ..via_detection.template_detector import detect_vias_template

try:
    from ...application.processing import (
        VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
        VIA_SEARCH_MODE_HEURISTIC,
        VIA_SEARCH_MODE_TEMPLATE,
        ContourExtractionSettings,
        normalize_via_search_mode,
    )
except ImportError:  # pragma: no cover
    ContourExtractionSettings = Any  # type: ignore[assignment]
    normalize_via_search_mode = None  # type: ignore[assignment]
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG = "bright_tophat_dog"  # type: ignore[assignment]
    VIA_SEARCH_MODE_HEURISTIC = "heuristic"  # type: ignore[assignment]
    VIA_SEARCH_MODE_TEMPLATE = "template"  # type: ignore[assignment]


class ViaStrategyName(StrEnum):
    HEURISTIC = "heuristic"
    TEMPLATE = "template"
    BRIGHT_TOPHAT_DOG = "bright_tophat_dog"


@dataclass(frozen=True, slots=True)
class ViaRunConfig:
    use_legacy_core: bool = False
    prefer_template_when_available: bool = True


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
        log: list[str] = []
        fixed_output_diameters = fixed_via_diameters_from_settings(legacy_settings)
        if normalize_via_search_mode is not None:
            mode = normalize_via_search_mode(legacy_settings.via_search_mode)
        else:
            mode = str(legacy_settings.via_search_mode or VIA_SEARCH_MODE_HEURISTIC)

        if mode == VIA_SEARCH_MODE_TEMPLATE:
            tcfg = template_config_from_settings(legacy_settings)
            result = detect_vias_template(gray, tcfg)
            strategy = str(ViaStrategyName.TEMPLATE)
            log.append(f"template: n_templates={len(tcfg.templates)} min_corr={tcfg.min_correlation:.3f}")
            hits = [_detection_to_hit(d, strategy, fixed_output_diameters) for d in result.accepted]
        elif mode == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG:
            from ..via_detection.result import DetectionResult, ViaDetection
            from .bright_tophat_dog import BrightViaDetectorConfig, detect_bright_vias

            cfg = BrightViaDetectorConfig.from_legacy_settings(legacy_settings)
            bright = detect_bright_vias(gray, cfg)
            result = DetectionResult(
                method=VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
                accepted=[
                    ViaDetection(
                        x=float(det.center[0]),
                        y=float(det.center[1]),
                        bbox=det.bbox,
                        score=float(det.final_score),
                        diameter_estimate=float((det.bbox[2] + det.bbox[3]) * 0.5),
                        contrast=float(det.brightness_score),
                        prominence=float(det.tophat_response + det.dog_response) * 0.5,
                        compactness=float(det.circularity),
                        aspect=float(det.aspect),
                        polarity_hypothesis="bright",
                        reject_reason=det.hard_reason or None,
                    )
                    for det in bright.detections
                ],
                debug_images=dict(bright.debug_images),
                parameters_snapshot={"config": repr(cfg)},
            )
            strategy = str(ViaStrategyName.BRIGHT_TOPHAT_DOG)
            log.append(
                f"bright_tophat_dog: diameter={cfg.diameter_min}-{cfg.diameter_max} min_score={cfg.min_final_score:.1f}"
            )
            hits = [_detection_to_hit(d, strategy, fixed_output_diameters) for d in result.accepted]
        else:
            hcfg = heuristic_config_from_settings(legacy_settings)
            result = detect_vias_heuristic(gray, hcfg)
            strategy = str(ViaStrategyName.HEURISTIC)
            log.append(f"heuristic: polar={hcfg.polarity!r}")
            ad = hcfg.allowed_diameters()
            log.append(f"heuristic: diameters={ad[:6]!r}{'...' if len(ad) > 6 else ''}")
            hits = [_detection_to_hit(d, strategy, fixed_output_diameters) for d in result.accepted]

        dbg = _result_debug(result, strategy)
        return ViaDetectionOutput(
            image=self.image_ref,
            mode=AppMode.VIA,
            output_kind=shape,
            hits=hits,
            selected_strategy=strategy,
            attempt_log=list(log),
            debug=dbg,
        )


def _detection_to_hit(
    detection: Any,
    strategy: str,
    fixed_output_diameters: list[int] | tuple[int, ...] = (),
) -> ViaHit:
    bbox = getattr(detection, "bbox", (0, 0, 0, 0))
    bw = float(bbox[2]) if isinstance(bbox, (tuple, list)) and len(bbox) >= 4 else 0.0
    bh = float(bbox[3]) if isinstance(bbox, (tuple, list)) and len(bbox) >= 4 else 0.0
    d_est = float(getattr(detection, "diameter_estimate", 0.0) or 0.0)
    observed_diameter = d_est if d_est > 0.0 else 0.5 * (bw + bh)
    configured_diameters = [float(value) for value in fixed_output_diameters if int(value) > 0]
    if configured_diameters:
        output_diameter = min(configured_diameters, key=lambda value: abs(value - observed_diameter))
        w = h = output_diameter
    elif bw > 0.0 and bh > 0.0:
        w, h = bw, bh
    elif d_est > 0.0:
        w = h = d_est
    else:
        w = h = 4.0
    # OpenCV contour/extrema centers are expressed at integer pixel sample
    # locations, while polygons, Qt scenes and CIF boxes use continuous image
    # coordinates where a pixel occupies [x, x + 1).  Convert sample centers to
    # that coordinate system. Template matching already adds half the template
    # width/height and therefore already contains this half-pixel correction.
    pixel_center_offset = 0.0 if strategy == str(ViaStrategyName.TEMPLATE) else 0.5
    return ViaHit(
        center_x=float(detection.x) + pixel_center_offset,
        center_y=float(detection.y) + pixel_center_offset,
        width=float(w),
        height=float(h),
        score=float(detection.score),
        strategy=strategy,
        contrast=float(getattr(detection, "contrast", 0.0)),
        edge_strength=float(getattr(detection, "prominence", 0.0)),
        annulus_coverage=float(getattr(detection, "compactness", 0.0)),
        extra={
            "diameter_estimate": float(getattr(detection, "diameter_estimate", 0.0)),
            "aspect": float(getattr(detection, "aspect", 0.0)),
            "polarity_hypothesis": str(getattr(detection, "polarity_hypothesis", "")),
            "final_score": float(getattr(detection, "score", 0.0)),
        },
    )


def _result_debug(result: DetectionResult, strategy: str) -> dict[str, Any]:
    d: dict[str, Any] = {**dict(result.debug_images), "parameters": dict(result.parameters_snapshot)}
    d["strategy"] = strategy
    d["rejected"] = [asdict(v) for v in result.rejected] if result.rejected else []
    d["below_threshold"] = [asdict(v) for v in result.below_threshold] if result.below_threshold else []
    d["candidates"] = "see overlay layer names in debug images"
    return d
