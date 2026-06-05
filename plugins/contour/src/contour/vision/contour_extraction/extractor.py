"""High-level SEM contour extraction pipeline.

This module is intentionally independent from the legacy ``contour_extractor``:
it receives a grayscale/BGR image, builds a filled foreground mask, extracts a
hierarchy with ``RETR_TREE``, filters vector components, and finally exposes
geometry as polygons or boxes through the shared ``vision.schemas`` dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from ..io_normalize import make_image_ref, to_gray_u8
from ..preprocessing import NoiseLevel, PreprocessConfig
from ..schemas import (
    AppMode,
    ContourExtractionOutput,
    HierarchicalComponent,
    ImageRef,
    OutputShapeKind,
    RotatedBox,
    SemPolarity,
)
from ...contour_extractor import estimate_effective_polygon_width_px
from .hierarchy import build_hierarchy_from_mask
from .sem_filled_mask import FilledMaskResult, FilledMaskSegmentationConfig, extract_filled_mask

SEM_BACKEND_LEGACY = "legacy"
SEM_BACKEND_AUTO = "sem"


@dataclass(frozen=True, slots=True)
class SemContourConfig:
    """Internal config generated from presets or a legacy UI settings object."""

    output_kind: OutputShapeKind = OutputShapeKind.POLYGON
    noise_level: NoiseLevel = NoiseLevel.MEDIUM
    polarity: SemPolarity = SemPolarity.AUTO
    segmentation: FilledMaskSegmentationConfig = field(default_factory=FilledMaskSegmentationConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    hierarchy_epsilon: float = 1.2
    preserve_hierarchy: bool = True
    min_area: float = 10.0
    max_area: float | None = None
    min_perimeter: float = 0.0
    max_perimeter: float | None = None
    min_bbox_width: int = 0
    max_bbox_width: int | None = None
    min_bbox_height: int = 0
    max_bbox_height: int | None = None
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float | None = None
    exclude_border_touching: bool = False
    min_hierarchy_depth: int = 0
    max_hierarchy_depth: int | None = None
    min_inner_hole_area: float = 100.0
    max_hole_area_ratio: float | None = None
    box_from_holes: bool = False
    min_polygon_width_px: float = 0.0

    @classmethod
    def from_legacy_settings(cls, settings: Any) -> SemContourConfig:
        output_kind = _output_kind_from_text(getattr(settings, "output_mode", "polygon"))
        noise_level = _noise_from_text(getattr(settings, "sem_noise_level", "medium"))
        polarity = _polarity_from_text(getattr(settings, "sem_polarity", "auto"))
        min_area = float(getattr(settings, "min_area", 10.0))
        segmentation = FilledMaskSegmentationConfig(
            min_component_area=max(1, round(min_area * 0.5)),
            close_radius=2 if noise_level != NoiseLevel.HIGH else 1,
            max_hole_fill_area=80 if bool(getattr(settings, "max_hole_area_ratio", None)) else 200,
        )
        preprocess = PreprocessConfig(denoise=noise_level)
        return cls(
            output_kind=output_kind,
            noise_level=noise_level,
            polarity=polarity,
            segmentation=segmentation,
            preprocess=preprocess,
            hierarchy_epsilon=float(getattr(settings, "epsilon", 1.2) or 1.2),
            preserve_hierarchy=bool(getattr(settings, "sem_preserve_hierarchy", True)),
            min_area=min_area,
            max_area=_none_if_zero(getattr(settings, "max_area", None), float),
            min_perimeter=float(getattr(settings, "min_perimeter", 0.0) or 0.0),
            max_perimeter=_none_if_zero(getattr(settings, "max_perimeter", None), float),
            min_bbox_width=max(0, int(getattr(settings, "min_bbox_width", 0) or 0)),
            max_bbox_width=_none_if_zero(getattr(settings, "max_bbox_width", None), int),
            min_bbox_height=max(0, int(getattr(settings, "min_bbox_height", 0) or 0)),
            max_bbox_height=_none_if_zero(getattr(settings, "max_bbox_height", None), int),
            min_aspect_ratio=max(0.0, float(getattr(settings, "min_aspect_ratio", 0.0) or 0.0)),
            max_aspect_ratio=_none_if_zero(getattr(settings, "max_aspect_ratio", None), float),
            exclude_border_touching=bool(getattr(settings, "exclude_border_touching", False)),
            min_hierarchy_depth=max(0, int(getattr(settings, "min_hierarchy_depth", 0) or 0)),
            max_hierarchy_depth=_none_if_zero(getattr(settings, "max_hierarchy_depth", None), int),
            min_inner_hole_area=max(0.0, float(getattr(settings, "min_inner_hole_area", 100.0) or 100.0)),
            max_hole_area_ratio=_none_if_zero(getattr(settings, "max_hole_area_ratio", None), float),
            min_polygon_width_px=max(0.0, float(getattr(settings, "min_polygon_width_px", 0.0) or 0.0)),
        )


@dataclass(slots=True)
class SemContourExtractor:
    """Filled-mask contour extraction for noisy SEM topology."""

    config: SemContourConfig

    def extract(self, image: Any, *, image_path: str | None = None) -> ContourExtractionOutput:
        gray = to_gray_u8(image)
        image_ref: ImageRef = make_image_ref(image_path, gray)
        seg = extract_filled_mask(
            gray,
            config=self.config.segmentation,
            preprocess=self.config.preprocess,
            polarity=self.config.polarity,
        )
        _contours, components = build_hierarchy_from_mask(seg.mask, epsilon=max(0.0, self.config.hierarchy_epsilon))
        components = self._filter_components(components, gray.shape, seg)
        if self.config.output_kind is OutputShapeKind.AXIS_ALIGNED_BOX:
            components = [self._component_as_box(c) for c in components if self.config.box_from_holes or not c.is_hole]
        elif self.config.output_kind is OutputShapeKind.ROTATED_BOX:
            components = [
                self._component_as_rotated_box(c) for c in components if self.config.box_from_holes or not c.is_hole
            ]
        components = self._remap_parent_ids(components)
        return ContourExtractionOutput(
            image=image_ref,
            mode=AppMode.CONTOUR,
            output_kind=self.config.output_kind,
            filled_mask=seg.mask,
            components=components,
            strategy_used=str(seg.strategy),
            quality_notes=[
                f"polarity={seg.polarity}",
                "alternatives=" + ", ".join(f"{name}:{score:.2f}" for name, score in seg.alternatives[:4]),
            ],
            debug={
                "preprocessed": seg.preprocessed,
                "alternatives": list(seg.alternatives),
            },
        )

    def _filter_components(
        self,
        components: list[HierarchicalComponent],
        image_shape: tuple[int, ...],
        seg: FilledMaskResult,
    ) -> list[HierarchicalComponent]:
        del seg
        image_height, image_width = int(image_shape[0]), int(image_shape[1])
        area_by_id = {component.id: float(component.area) for component in components}
        kept: list[HierarchicalComponent] = []
        for component in components:
            if not self.config.preserve_hierarchy and component.is_hole:
                continue
            if component.area < self.config.min_area:
                continue
            if self.config.max_area is not None and component.area > self.config.max_area:
                continue
            perimeter = _perimeter(component.points)
            if perimeter < self.config.min_perimeter:
                continue
            if self.config.max_perimeter is not None and perimeter > self.config.max_perimeter:
                continue
            x_coord, y_coord, width, height = component.bbox_xywh
            if width < self.config.min_bbox_width or height < self.config.min_bbox_height:
                continue
            if self.config.max_bbox_width is not None and width > self.config.max_bbox_width:
                continue
            if self.config.max_bbox_height is not None and height > self.config.max_bbox_height:
                continue
            aspect = float(width) / float(max(1, height))
            if aspect < self.config.min_aspect_ratio:
                continue
            if self.config.max_aspect_ratio is not None and aspect > self.config.max_aspect_ratio:
                continue
            if self.config.exclude_border_touching and (
                x_coord <= 0 or y_coord <= 0 or x_coord + width >= image_width or y_coord + height >= image_height
            ):
                continue
            if component.depth < self.config.min_hierarchy_depth:
                continue
            if self.config.max_hierarchy_depth is not None and component.depth > self.config.max_hierarchy_depth:
                continue
            if component.is_hole and component.area < self.config.min_inner_hole_area:
                continue
            if self.config.max_hole_area_ratio is not None and component.is_hole and component.parent_id is not None:
                parent_area = area_by_id.get(component.parent_id, 0.0)
                if parent_area > 0.0 and component.area / parent_area > self.config.max_hole_area_ratio:
                    continue
            if self.config.min_polygon_width_px > 0.0 and len(component.points) >= 3:
                zero = np.zeros((image_height, image_width), dtype=np.uint8)
                ctr = np.array(
                    [[int(round(p[0])), int(round(p[1]))] for p in component.points],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                cv2.fillPoly(zero, [ctr], 255)
                w_est, _ = estimate_effective_polygon_width_px(zero, ctr)
                if w_est < float(self.config.min_polygon_width_px):
                    continue
            kept.append(replace(component, score=1.0, source_strategy="sem_filled_mask"))
        return kept

    @staticmethod
    def _component_as_box(component: HierarchicalComponent) -> HierarchicalComponent:
        x_coord, y_coord, width, height = component.bbox_xywh
        points = _box_points(x_coord, y_coord, width, height)
        return replace(
            component,
            is_hole=False,
            parent_id=None,
            depth=0,
            points=points,
            area=float(width * height),
            source_strategy=f"{component.source_strategy}:box",
        )

    @staticmethod
    def _component_as_rotated_box(component: HierarchicalComponent) -> HierarchicalComponent:
        if len(component.points) < 3:
            return SemContourExtractor._component_as_box(component)
        arr = np.array(component.points, dtype=np.float32).reshape(-1, 1, 2)
        (cx, cy), (width, height), angle = cv2.minAreaRect(arr)
        box = cv2.boxPoints(((cx, cy), (width, height), angle))
        points = [(float(x), float(y)) for x, y in box]
        return replace(
            component,
            is_hole=False,
            parent_id=None,
            depth=0,
            points=points,
            area=float(max(1.0, width) * max(1.0, height)),
            bbox_xywh=tuple(int(v) for v in cv2.boundingRect(box.astype(np.float32))),
            rotated_box=RotatedBox(float(cx), float(cy), float(width), float(height), float(angle)),
            source_strategy=f"{component.source_strategy}:rbox",
        )

    @staticmethod
    def _remap_parent_ids(components: list[HierarchicalComponent]) -> list[HierarchicalComponent]:
        old_to_new: dict[int, int] = {}
        remapped: list[HierarchicalComponent] = []
        for new_id, component in enumerate(components, start=1):
            old_to_new[component.id] = new_id
        for new_id, component in enumerate(components, start=1):
            parent_id = old_to_new.get(component.parent_id) if component.parent_id is not None else None
            remapped.append(replace(component, id=new_id, parent_id=parent_id))
        return remapped


def _output_kind_from_text(value: Any) -> OutputShapeKind:
    text = str(value or "").strip().lower()
    if text in {"box", "axis_aligned_box", "axis-aligned-box"}:
        return OutputShapeKind.AXIS_ALIGNED_BOX
    if text in {"rbox", "rotated_box", "rotated-box"}:
        return OutputShapeKind.ROTATED_BOX
    return OutputShapeKind.POLYGON


def _noise_from_text(value: Any) -> NoiseLevel:
    text = str(value or "").strip().lower()
    if text in {NoiseLevel.LOW, NoiseLevel.MEDIUM, NoiseLevel.HIGH}:
        return NoiseLevel(text)
    return NoiseLevel.MEDIUM


def _polarity_from_text(value: Any) -> SemPolarity:
    text = str(value or "").strip().lower()
    if text in {SemPolarity.DARK_FOREGROUND, "dark", "dark_foreground"}:
        return SemPolarity.DARK_FOREGROUND
    if text in {SemPolarity.BRIGHT_FOREGROUND, "bright", "bright_foreground"}:
        return SemPolarity.BRIGHT_FOREGROUND
    return SemPolarity.AUTO


def _none_if_zero(value: Any, caster: type[float] | type[int]) -> Any | None:
    if value in (None, "", 0, 0.0):
        return None
    parsed = caster(value)
    return None if parsed == 0 else parsed


def _perimeter(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    accum = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        accum += float(np.hypot(next_point[0] - point[0], next_point[1] - point[1]))
    return accum


def _box_points(x_coord: int, y_coord: int, width: int, height: int) -> list[tuple[float, float]]:
    left = float(x_coord)
    top = float(y_coord)
    right = float(x_coord + max(1, width))
    bottom = float(y_coord + max(1, height))
    return [(left, top), (right, top), (right, bottom), (left, bottom)]
