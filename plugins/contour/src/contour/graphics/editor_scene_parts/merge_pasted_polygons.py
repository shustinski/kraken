"""editor_scene: merge pasted polygons."""
from __future__ import annotations
from math import ceil, floor
from shapely import make_valid, unary_union
from contour.commands import (
    ReplacePolygonSetCommand,
)
from contour.domain import PolygonData
from contour.graphics.brush_vector import (
    region_geometry,
    shapely_to_polygon_data_list,
)


class MergePastedPolygonsMixin:
    def _merge_pasted_polygons(self, pasted_polygons: list[PolygonData]) -> list[int]:
        try:
            pasted_map = {polygon.id: polygon.clone() for polygon in pasted_polygons}
            pasted_region = region_geometry(pasted_map, list(pasted_map))
        except Exception:
            return []
        overlapping_root_ids: list[int] = []
        min_x, min_y, max_x, max_y = pasted_region.bounds
        candidate_ids = self._polygon_spatial_index.query_bbox(
            (
                floor(min_x),
                floor(min_y),
                max(1, ceil(max_x - min_x)),
                max(1, ceil(max_y - min_y)),
            )
        )
        for polygon_id in sorted(candidate_ids):
            polygon = self._polygons.get(polygon_id)
            if polygon is None:
                continue
            if polygon.is_hole:
                continue
            try:
                existing_region = region_geometry(
                    self._polygons,
                    self._polygon_family_ids(polygon_id),
                )
            except ValueError:
                continue
            if existing_region.intersects(pasted_region):
                overlapping_root_ids.append(polygon_id)
        if not overlapping_root_ids:
            return self._add_pasted_polygon_set(pasted_polygons)
        replaced_ids = self._render_polygon_ids(overlapping_root_ids)
        try:
            if replaced_ids:
                existing_region = region_geometry(self._polygons, replaced_ids)
                merged_region = unary_union(
                    make_valid(existing_region.union(pasted_region))
                )
            else:
                merged_region = unary_union(make_valid(pasted_region))
            rebuilt = shapely_to_polygon_data_list(merged_region)
        except Exception:
            return []
        if not rebuilt:
            return []
        reference = (
            self._polygons[overlapping_root_ids[0]]
            if overlapping_root_ids
            else next(
                (polygon for polygon in pasted_polygons if not polygon.is_hole),
                pasted_polygons[0],
            )
        )
        for candidate in rebuilt:
            candidate.category = reference.category
            candidate.shape_hint = reference.shape_hint
        rebuilt = self._assign_polygon_ids(rebuilt, replaced_ids)
        before = self.get_polygons()
        replaced_id_set = set(replaced_ids)
        after = [
            polygon.clone()
            for polygon in before
            if polygon.id not in replaced_id_set
        ]
        after.extend(polygon.clone() for polygon in rebuilt)
        after.sort(key=lambda polygon: polygon.id)
        self.undo_stack.push(
            ReplacePolygonSetCommand(self, before, after, "Paste and merge polygons")
        )
        selected_ids = [polygon.id for polygon in rebuilt]
        self.select_polygons(selected_ids)
        return selected_ids
