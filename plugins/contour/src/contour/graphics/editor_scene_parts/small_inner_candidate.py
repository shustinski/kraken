"""editor_scene: small inner candidate."""
from __future__ import annotations
from shapely import make_valid, unary_union
from contour.graphics.brush_vector import (
    QUAD_SEGS_BRUSH_DEFAULT,
    geometry_area_components,
    polygon_paint_footprint_geom,
    region_geometry,
    tool_geometry,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)


class SmallInnerCandidateMixin:
    def _is_small_inner_candidate(self, points: list[tuple[float, float]], thickness: float | None) -> bool:
        """True when every connected tool region is below min area inside existing metal or void."""

        min_area = float(self._vector_geometry_settings.min_hole_area_to_remove_px2)
        if min_area <= 0.0 or not self._polygons or not points:
            return False
        try:
            candidate_shape = tool_geometry(points, thickness, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
        except Exception:
            return False
        if candidate_shape.is_empty:
            return False

        tool_components = geometry_area_components(make_valid(candidate_shape))
        if not tool_components:
            return False
        # Threshold applies to each inner region separately, never to their combined area.
        if any(float(component.area) >= min_area for component in tool_components):
            return False

        candidate_union = unary_union(tool_components)
        root_ids = sorted(
            polygon_id
            for polygon_id, polygon in self._polygons.items()
            if not polygon.is_hole
            and polygon.parent_id is None
            and not _is_via_polygon(polygon)
            and len(polygon.points) >= 3
        )
        for root_id in root_ids:
            try:
                metal = make_valid(region_geometry(self._polygons, self._polygon_family_ids(root_id)))
            except ValueError:
                continue
            if metal.is_empty:
                continue

            if metal.intersects(candidate_union):
                carved = make_valid(metal.intersection(candidate_union))
                carved_components = geometry_area_components(carved)
                if carved_components and all(float(part.area) < min_area for part in carved_components):
                    return True

            try:
                envelope = make_valid(polygon_paint_footprint_geom(self._polygons[root_id].points))
            except Exception:
                continue
            if envelope.is_empty:
                continue
            void_geom = make_valid(envelope.difference(metal))
            if not void_geom.is_empty and void_geom.intersects(candidate_union):
                void_touch = make_valid(void_geom.intersection(candidate_union))
                void_components = geometry_area_components(void_touch)
                if void_components and all(float(part.area) < min_area for part in void_components):
                    return True

        return False
