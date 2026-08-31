from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath

from ..domain import PolygonData, integer_points
from ..serializers import (
    _cif_paint_mask_from_ring,
    _dedupe_closed_points,
    _has_duplicate_points,
    _polygon_uses_authored_cif_paint_ring,
)


@dataclass(frozen=True, slots=True)
class InternalContourFamilyIssue:
    outer_id: int
    hole_ids: tuple[int, ...]
    hole_centers_filled: int


@dataclass(frozen=True, slots=True)
class InternalContourAnalysis:
    issues: tuple[InternalContourFamilyIssue, ...] = field(default_factory=tuple)
    checked_families: int = 0
    skipped_klayout_keyholes: int = 0

    @property
    def needs_fix(self) -> bool:
        return bool(self.issues)


@dataclass(frozen=True, slots=True)
class InternalContourFixStats:
    checked_cif_files: int = 0
    checked_families: int = 0
    fixed_cif_files: int = 0
    fixed_families: int = 0
    fixed_hole_regions: int = 0
    skipped_klayout_keyholes: int = 0
    unchanged_cif_files: int = 0
    failed: tuple[str, ...] = field(default_factory=tuple)
    cancelled: bool = False


def _hole_center(point_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, width, height = point_bbox
    return int(round(left + width / 2.0)), int(round(top + height / 2.0))


def _mask_sample(mask: np.ndarray, left: int, top: int, x_coord: int, y_coord: int) -> int:
    local_x = int(x_coord) - int(left)
    local_y = int(y_coord) - int(top)
    if local_y < 0 or local_x < 0 or local_y >= mask.shape[0] or local_x >= mask.shape[1]:
        return 0
    return int(mask[local_y, local_x] > 0)


def _rasterize_winding_path_to_mask(path: QPainterPath, image_size: tuple[int, int]) -> np.ndarray:
    image_width, image_height = int(image_size[0]), int(image_size[1])
    image = QImage(image_width, image_height, QImage.Format.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        fill_path = QPainterPath(path)
        fill_path.setFillRule(Qt.FillRule.WindingFill)
        painter.fillPath(fill_path, QBrush(QColor(255, 255, 255)))
    finally:
        painter.end()
    bytes_per_line = int(image.bytesPerLine())
    return (
        np.frombuffer(bytes(image.constBits().asarray(image.sizeInBytes())), dtype=np.uint8)
        .reshape((image_height, bytes_per_line))[:, :image_width]
        .copy()
    )


def _legacy_cutout_path_for_family(outer: PolygonData, holes: list[PolygonData]) -> QPainterPath:
    from ..graphics_items import _cutout_path_for_polygon, _display_path_for_polygon

    path = QPainterPath()
    path.addPath(_display_path_for_polygon(outer))
    for hole in holes:
        path.addPath(_cutout_path_for_polygon(hole, outer=outer))
    path.setFillRule(Qt.FillRule.WindingFill)
    return path


def _authored_paint_path_for_outer(outer: PolygonData) -> QPainterPath:
    path = QPainterPath()
    ring = outer.cif_paint_ring
    if len(ring) < 3:
        return path
    first_x, first_y = ring[0]
    path.moveTo(float(first_x), float(first_y))
    for x_coord, y_coord in ring[1:]:
        path.lineTo(float(x_coord), float(y_coord))
    path.closeSubpath()
    path.setFillRule(Qt.FillRule.WindingFill)
    return path


def _is_klayout_keyhole_slot(
    outer: PolygonData,
    holes: list[PolygonData],
    *,
    authored_ring: list[tuple[float, float]] | None = None,
) -> bool:
    if len(holes) != 1:
        return False
    ring = authored_ring if authored_ring is not None else outer.cif_paint_ring
    if not ring or not _has_duplicate_points(_dedupe_closed_points(ring)):
        return False
    paint_mask, left, top = _cif_paint_mask_from_ring(ring)
    hole = holes[0]
    center_x, center_y = _hole_center(hole.bbox)
    return _mask_sample(paint_mask, left, top, center_x, center_y) > 0


def analyze_internal_contour_display(
    polygons: list[PolygonData],
    image_size: tuple[int, int],
) -> InternalContourAnalysis:
    holes_by_parent: dict[int, list[PolygonData]] = {}
    for polygon in polygons:
        if polygon.is_hole and polygon.parent_id is not None:
            holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)

    issues: list[InternalContourFamilyIssue] = []
    checked_families = 0
    skipped_klayout_keyholes = 0
    for outer in polygons:
        if outer.is_hole or not _polygon_uses_authored_cif_paint_ring(outer):
            continue
        holes = holes_by_parent.get(int(outer.id), [])
        if not holes:
            continue
        checked_families += 1
        if _is_klayout_keyhole_slot(outer, holes):
            skipped_klayout_keyholes += 1
            continue

        cutout_mask = _rasterize_winding_path_to_mask(
            _legacy_cutout_path_for_family(outer, holes),
            image_size,
        )
        paint_mask = _rasterize_winding_path_to_mask(_authored_paint_path_for_outer(outer), image_size)

        filled_centers = 0
        for hole in holes:
            center_x, center_y = _hole_center(hole.bbox)
            if cutout_mask[center_y, center_x] == 0 and paint_mask[center_y, center_x] > 0:
                filled_centers += 1
        if filled_centers > 0:
            issues.append(
                InternalContourFamilyIssue(
                    outer_id=int(outer.id),
                    hole_ids=tuple(int(hole.id) for hole in holes),
                    hole_centers_filled=filled_centers,
                )
            )
    return InternalContourAnalysis(
        issues=tuple(issues),
        checked_families=checked_families,
        skipped_klayout_keyholes=skipped_klayout_keyholes,
    )


def fix_internal_contour_display(
    polygons: list[PolygonData],
    image_size: tuple[int, int],
) -> tuple[list[PolygonData], InternalContourAnalysis, bool]:
    analysis = analyze_internal_contour_display(polygons, image_size)
    if not analysis.needs_fix:
        return polygons, analysis, False

    issue_outer_ids = {issue.outer_id for issue in analysis.issues}
    fixed: list[PolygonData] = []
    changed = False
    for polygon in polygons:
        if polygon.id in issue_outer_ids and not polygon.is_hole:
            if integer_points(polygon.cif_paint_ring) != integer_points(polygon.points):
                clone = polygon.clone()
                clone.cif_paint_ring = list(polygon.points)
                fixed.append(clone)
                changed = True
                continue
        fixed.append(polygon.clone())
    return fixed, analysis, changed


def should_use_cutout_display_for_keyhole_family(
    authored_ring: list[tuple[int, int]],
    outer_points: list[tuple[float, float]],
    hole_rings: list[list[tuple[float, float]]],
    image_size: tuple[int, int],
) -> bool:
    if not hole_rings:
        return False
    outer_area, outer_perimeter, outer_bbox = _metrics_for_ring(outer_points)
    outer = PolygonData(
        id=1,
        points=outer_points,
        is_hole=False,
        parent_id=None,
        category="conductor",
        shape_hint="polygon",
        area=outer_area,
        perimeter=outer_perimeter,
        bbox=outer_bbox,
        cif_paint_ring=[(float(x_coord), float(y_coord)) for x_coord, y_coord in authored_ring],
    )
    holes: list[PolygonData] = []
    for index, hole_points in enumerate(hole_rings, start=2):
        hole_area, hole_perimeter, hole_bbox = _metrics_for_ring(hole_points)
        holes.append(
            PolygonData(
                id=index,
                points=hole_points,
                is_hole=True,
                parent_id=1,
                category="conductor",
                shape_hint="polygon",
                area=hole_area,
                perimeter=hole_perimeter,
                bbox=hole_bbox,
            )
        )
    if _is_klayout_keyhole_slot(outer, holes, authored_ring=[(float(x), float(y)) for x, y in authored_ring]):
        return False
    analysis = analyze_internal_contour_display([outer, *holes], image_size)
    return analysis.needs_fix


def _metrics_for_ring(points: list[tuple[float, float]]) -> tuple[float, float, tuple[int, int, int, int]]:
    from ..domain import compute_polygon_metrics

    return compute_polygon_metrics(points)
