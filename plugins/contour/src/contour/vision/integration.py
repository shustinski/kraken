"""High-level glue for gradual migration from the widget / use cases."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..domain import PolygonData, compute_polygon_metrics
from .contour_extraction import SemContourConfig, SemContourExtractor
from .io_normalize import make_image_ref, to_gray_u8
from .schemas import AppMode, ContourExtractionOutput, HierarchicalComponent, OutputShapeKind, ViaDetectionOutput, ViaHit
from .via import CompositeViaDetector, ViaRunConfig


def output_kind_from_text(value: Any) -> OutputShapeKind:
    text = str(value or "").strip().lower()
    if text in {"box", "axis_aligned_box", "axis-aligned-box"}:
        return OutputShapeKind.AXIS_ALIGNED_BOX
    if text in {"rbox", "rotated_box", "rotated-box"}:
        return OutputShapeKind.ROTATED_BOX
    return OutputShapeKind.POLYGON


def run_contour_filled_mask(
    image: Any,
    *,
    image_path: str | None,
    output_kind: OutputShapeKind,
    noise_level: str = "medium",
    hierarchy_epsilon: float = 1.2,
    legacy_settings: Any | None = None,
) -> ContourExtractionOutput:
    """Single entry: grayscale/BGR in -> :class:`ContourExtractionOutput`."""

    if legacy_settings is not None:
        config = SemContourConfig.from_legacy_settings(legacy_settings)
    else:
        config = SemContourConfig(output_kind=output_kind, hierarchy_epsilon=hierarchy_epsilon)
    config = _replace_contour_output(config, output_kind=output_kind, noise_level=noise_level)
    return SemContourExtractor(config).extract(image, image_path=image_path)


def run_via_detection(
    image: Any,
    *,
    image_path: str | None,
    output_kind: OutputShapeKind,
    legacy_settings: Any,
) -> ViaDetectionOutput:
    gray = to_gray_u8(image)
    from ..application.processing import ALGORITHM_BACKEND_SEM

    if str(getattr(legacy_settings, "algorithm_backend", "")).lower() == ALGORITHM_BACKEND_SEM:
        return _run_sem_via_detection(gray, image, image_path, output_kind, legacy_settings)

    detector = CompositeViaDetector(
        make_image_ref(image_path, gray),
        ViaRunConfig(
            use_legacy_core=str(getattr(legacy_settings, "algorithm_backend", "")).lower() == "legacy_via",
            prefer_template_when_available=True,
        ),
    )
    return detector.run(gray, shape=output_kind, legacy_settings=legacy_settings)


def _run_sem_via_detection(
    gray: Any,
    image: Any,
    image_path: str | None,
    output_kind: OutputShapeKind,
    legacy_settings: Any,
) -> ViaDetectionOutput:
    """SEM backend: exactly one selected via detector, template or heuristic."""

    from ..application.processing import (
        VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
        VIA_SEARCH_MODE_HEURISTIC,
        VIA_SEARCH_MODE_TEMPLATE,
        normalize_via_search_mode,
    )
    from .via.bright_tophat_dog import BrightViaDetectorConfig, detect_bright_vias
    from .via.orchestrator import _detection_to_hit, _result_debug
    from .via_detection.heuristic_detector import detect_vias_heuristic
    from .via_detection.result import DetectionResult, ViaDetection
    from .via_detection.settings_bridge import heuristic_config_from_settings, template_config_from_settings
    from .via_detection.template_detector import detect_vias_template

    image_ref = make_image_ref(image_path, gray)
    mode = normalize_via_search_mode(getattr(legacy_settings, "via_search_mode", ""))
    log: list[str] = []

    if mode == VIA_SEARCH_MODE_TEMPLATE:
        tcfg = template_config_from_settings(legacy_settings)
        result = detect_vias_template(gray, tcfg)
        hits = [_detection_to_hit(d, "template") for d in result.accepted]
        dbg = _result_debug(result, "template")
        return ViaDetectionOutput(
            image=image_ref,
            mode=AppMode.VIA,
            output_kind=output_kind,
            hits=hits,
            selected_strategy="template",
            attempt_log=[f"template: n_templates={len(tcfg.templates)} min_corr={tcfg.min_correlation:.3f}"],
            debug=dbg,
        )

    if mode == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG:
        cfg = BrightViaDetectorConfig.from_legacy_settings(legacy_settings)
        bright = detect_bright_vias(gray, cfg)
        accepted = [
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
        ]
        result = DetectionResult(
            method=VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
            accepted=accepted,
            rejected=[],
            debug_images=dict(bright.debug_images),
            parameters_snapshot={"config": repr(cfg)},
        )
        hits = [_detection_to_hit(d, VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG) for d in result.accepted]
        return ViaDetectionOutput(
            image=image_ref,
            mode=AppMode.VIA,
            output_kind=output_kind,
            hits=hits,
            selected_strategy=VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
            attempt_log=[
                f"bright_tophat_dog: diameter={cfg.diameter_min}-{cfg.diameter_max} min_score={cfg.min_final_score:.1f}"
            ],
            debug=_result_debug(result, VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG),
        )

    if mode != VIA_SEARCH_MODE_HEURISTIC:
        mode = VIA_SEARCH_MODE_HEURISTIC
    hcfg = heuristic_config_from_settings(legacy_settings)
    result = detect_vias_heuristic(gray, hcfg)
    hits = [_detection_to_hit(d, "heuristic") for d in result.accepted]
    dbg = _result_debug(result, "heuristic")
    ad = hcfg.allowed_diameters()
    log.append(f"heuristic: polar={hcfg.polarity!r}")
    log.append(f"heuristic: diameters={ad[:12]!r}{'...' if len(ad) > 12 else ''}")
    return ViaDetectionOutput(
        image=image_ref,
        mode=AppMode.VIA,
        output_kind=output_kind,
        hits=hits,
        selected_strategy="heuristic",
        attempt_log=log,
        debug=dbg,
    )


def contour_output_to_polygons(output: ContourExtractionOutput, *, category: str = "conductor") -> list[PolygonData]:
    polygons: list[PolygonData] = []
    for index, component in enumerate(output.components, start=1):
        points = list(component.points)
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygons.append(
            PolygonData(
                id=index,
                points=points,
                is_hole=bool(component.is_hole),
                parent_id=component.parent_id,
                category=category,
                shape_hint=_shape_hint(output.output_kind),
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        )
    return polygons


def via_output_to_polygons(output: ViaDetectionOutput) -> list[PolygonData]:
    polygons: list[PolygonData] = []
    for index, hit in enumerate(output.hits, start=1):
        points = _via_hit_points(hit, output.output_kind)
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygons.append(
            PolygonData(
                id=index,
                points=points,
                is_hole=False,
                parent_id=None,
                category="via",
                shape_hint=_shape_hint(output.output_kind),
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        )
    return polygons


def components_to_mask_components(output: ContourExtractionOutput) -> list[HierarchicalComponent]:
    """Small public helper for tests/debug views that should not touch ``debug`` arrays."""

    return list(output.components)


def _replace_contour_output(
    config: SemContourConfig,
    *,
    output_kind: OutputShapeKind,
    noise_level: str,
) -> SemContourConfig:
    from .preprocessing import NoiseLevel, PreprocessConfig

    noise = NoiseLevel(noise_level) if str(noise_level) in {"low", "medium", "high"} else config.noise_level
    return replace(config, output_kind=output_kind, noise_level=noise, preprocess=PreprocessConfig(denoise=noise))


def _shape_hint(kind: OutputShapeKind) -> str:
    if kind is OutputShapeKind.AXIS_ALIGNED_BOX:
        return "box"
    if kind is OutputShapeKind.ROTATED_BOX:
        return "rbox"
    return "polygon"


def _via_hit_points(hit: ViaHit, kind: OutputShapeKind) -> list[tuple[float, float]]:
    if kind is OutputShapeKind.POLYGON:
        return _ellipse_points(hit.center_x, hit.center_y, hit.width * 0.5, hit.height * 0.5, vertices=20)
    x_coord, y_coord, width, height = hit.to_axis_aligned_box()
    left = float(x_coord)
    top = float(y_coord)
    right = float(x_coord + width)
    bottom = float(y_coord + height)
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _ellipse_points(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    *,
    vertices: int,
) -> list[tuple[float, float]]:
    count = max(8, int(vertices))
    result: list[tuple[float, float]] = []
    for index in range(count):
        angle = 2.0 * np.pi * float(index) / float(count)
        result.append((float(center_x + np.cos(angle) * radius_x), float(center_y + np.sin(angle) * radius_y)))
    return result
