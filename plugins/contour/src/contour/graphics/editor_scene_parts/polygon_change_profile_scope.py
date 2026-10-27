"""editor_scene: polygon change profile scope."""
from __future__ import annotations
from contextlib import contextmanager
from ...application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_overlap_repair_patch,
    polygons_needing_repair,
)
from contour.domain import PolygonData
from ...infrastructure.polygon_change_profiler import PolygonChangeProfile
from ...infrastructure.profiling import polygon_change_profiling_enabled
from contour.graphics.brush_vector import (
    polygon_equivalent_preserved,
)
from contour.graphics.tool_mode_logic import (
    available_editor_tools,
    can_add_polygon,
    can_add_polygon_set,
    can_add_via,
    is_recognized_via,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)
from contour.graphics.tools import EditorTool


class PolygonChangeProfileScopeMixin:
    def end_zoom_vector_render_mode(self) -> None:
        if not self._zoom_contact_composite_items and not self._zoom_hidden_contact_ids:
            return
        for polygon_id in self._zoom_hidden_contact_ids:
            polygon = self._polygons.get(polygon_id)
            item = self._polygon_items.get(polygon_id)
            if polygon is None or item is None:
                continue
            category = str(getattr(polygon, "category", "") or "")
            item.setVisible(
                self._polygon_overlays_visible
                and self._polygon_category_visible.get(category, True)
            )
        self._zoom_hidden_contact_ids.clear()
        for composite in self._zoom_contact_composite_items:
            self.removeItem(composite)
        self._zoom_contact_composite_items.clear()

    def polygon_overlays_visible(self) -> bool:
        return self._polygon_overlays_visible

    def get_polygons(self) -> list[PolygonData]:
        return [self._polygons[polygon_id].clone() for polygon_id in sorted(self._polygons)]

    def contact_count(self) -> int:
        return sum(1 for polygon in self._polygons.values() if _is_via_polygon(polygon))

    def selected_contact_count(self) -> int:
        return sum(
            1
            for polygon_id in self._selected_polygon_ids
            if (
                polygon_id in self._polygons
                and _is_via_polygon(self._polygons[polygon_id])
            )
        )

    def selected_object_counts(self) -> tuple[int, int]:
        selected_count = sum(
            1
            for polygon_id in self._selected_polygon_ids
            if polygon_id in self._polygons
        )
        contact_count = self.selected_contact_count()
        return contact_count, selected_count - contact_count

    def set_protect_recognized_vias(self, enabled: bool) -> None:
        self._protect_recognized_vias = bool(enabled)

    def set_minimum_contact_distance(self, distance: float) -> None:
        self._minimum_contact_distance = max(0.0, float(distance))

    def can_add_polygon(self) -> bool:
        return can_add_polygon(self._polygons.values())

    def can_add_via(self) -> bool:
        return can_add_via(self._polygons.values())

    def can_add_polygon_set(self, polygons: list[PolygonData]) -> bool:
        return can_add_polygon_set(self._polygons.values(), polygons)

    def available_editor_tools(self) -> frozenset[EditorTool]:
        return available_editor_tools(self._polygons.values())

    def polygon_is_deletable(self, polygon: PolygonData | None) -> bool:
        if polygon is None:
            return False
        return not (self._protect_recognized_vias and is_recognized_via(polygon))

    def set_vector_geometry_settings(self, settings: VectorGeometrySettings | None) -> None:
        self._vector_geometry_settings = settings if settings is not None else VectorGeometrySettings()
        if self._polygon_items:
            self._refresh_all_items()
        else:
            self._polygons_needing_repair = {}

    def _rebuild_polygons_needing_repair_cache(self, *, defer_overlap: bool = False) -> None:
        polygons = list(self._polygons.values())
        self._polygons_needing_repair = polygons_needing_repair(
            polygons,
            self._vector_geometry_settings,
            include_overlap=False,
        )
        overlap_check_roots = {
            polygon.id
            for polygon in polygons
            if polygon.parent_id is None
            and not polygon.is_hole
            and str(polygon.category) != "via"
            and str(polygon.shape_hint) != "box"
        }
        if defer_overlap:
            self._schedule_overlap_repair_patch(overlap_check_roots)
            return
        self._polygons_needing_repair = apply_overlap_repair_patch(
            self._polygons_needing_repair,
            polygons,
            overlap_check_roots,
        )

    def apply_polygons_needing_repair(
        self,
        reasons: dict[int, list[str]] | None,
        *,
        refresh_items: bool = True,
    ) -> None:
        """Install a precomputed repair map without rescanning geometry."""

        previous_ids = set(self._polygons_needing_repair)
        if reasons is None:
            self._polygons_needing_repair = {}
        else:
            self._polygons_needing_repair = {
                int(polygon_id): list(codes) for polygon_id, codes in reasons.items()
            }
        if not refresh_items or not self._polygon_items:
            return
        changed_ids = previous_ids | set(self._polygons_needing_repair)
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        for polygon_id in changed_ids:
            item = self._polygon_items.get(polygon_id)
            if item is not None:
                self._refresh_polygon_item(polygon_id, item, editable_vertex_ids=editable_vertex_ids)

    def polygon_needs_repair(self, polygon_id: int) -> bool:
        return polygon_id in self._polygons_needing_repair

    def polygons_needing_repair_map(self) -> dict[int, list[str]]:
        return {polygon_id: list(reasons) for polygon_id, reasons in self._polygons_needing_repair.items()}

    def _selection_reference_polygon(self) -> PolygonData | None:
        if self._selected_polygon_id is None:
            return None
        polygon = self._polygons.get(self._selected_polygon_id)
        return polygon.clone() if polygon is not None else None

    def _restore_selection_fallback_if_needed(self, reference: PolygonData | None) -> None:
        if reference is None or self._selected_polygon_id is not None:
            return
        for polygon in self._polygons.values():
            if polygon_equivalent_preserved(polygon, [reference]):
                self.select_polygon(polygon.id)
                return

    def _polygon_change_operation_from_label(self, label: str) -> str:
        normalized = str(label).strip().lower()
        mapping = {
            "erase brush stroke": "erase_brush",
            "erase rectangle": "erase_rectangle",
            "erase trace": "erase_trace",
            "erase polygon": "erase_polygon",
            "add brush stroke": "add_brush",
            "add trace": "add_trace",
            "add rectangle": "add_rectangle",
            "add polygon": "add_polygon",
            "delete vertices in area": "delete_vertices",
            "vector geometry cleanup": "postprocess",
        }
        return mapping.get(normalized, normalized.replace(" ", "_"))

    @contextmanager
    def _polygon_change_profile_scope(self, operation: str):
        if not polygon_change_profiling_enabled():
            yield None
            return
        previous = self._polygon_change_profile
        if previous is not None:
            self._emit_polygon_change_profile(previous, status="superseded")
        profile = PolygonChangeProfile.begin(
            operation=operation,
            polygon_count_before=len(self._polygons),
        )
        self._polygon_change_profile = profile
        status = "completed"
        try:
            yield profile
        except Exception:
            status = "error"
            raise
        finally:
            if self._polygon_change_profile is profile:
                self._polygon_change_profile = None
            profile.set_metadata(polygons_after=len(self._polygons))
            profile.finish()
            self._emit_polygon_change_profile(profile, status=status)

    def _note_polygon_change_phase(
        self,
        profile: PolygonChangeProfile | None,
        name: str,
        started_at: float,
    ) -> None:
        if profile is not None:
            profile.note_phase(name, started_at)

    def _emit_polygon_change_profile(self, profile: PolygonChangeProfile, *, status: str) -> None:
        from contour.graphics.editor_scene import write_profile_report

        summary = profile.format_summary(status=status)
        stats = profile.format_stats()
        write_profile_report(summary, stats)
