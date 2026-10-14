"""editor_view: mouse release event."""
from __future__ import annotations
import cProfile
from time import perf_counter
from PyQt6.QtCore import (
    QPointF,
    QRectF,
    Qt,
)
from PyQt6.QtGui import (
    QMouseEvent,
)
from ...application.vector_geometry_postprocess import (
    apply_edge_translation_to_clone,
    apply_polygon_points_to_clone,
    collapse_redundant_vertices_in_polygons,
    merge_overlapping_root_families_near_polygons,
    postprocess_changed_polygon_edit,
    postprocess_vertex_move_edit,
    resolve_focus_id_after_geometry_pass,
)
from contour.commands import MovePolygonCommand
from ...domain.polygon_ring import is_valid_closed_polygon_edge_move, is_valid_closed_polygon_vertex_move
from ...infrastructure.profiling import (
    try_disable_profiler,
    try_enable_profiler,
    vertex_move_profiling_enabled,
)
from contour.graphics.geometry import (
    _points_different,
    _polygon_points_different,
    is_valid_closed_polygon_ring,
)
from contour.graphics.tools import EditorTool


class MouseReleaseEventMixin:
    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        contact_drag_commit_started_at = (
            perf_counter()
            if self._contact_drag_profile is not None
            else None
        )
        contact_drag_status = "released"
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan_active:
            self._middle_pan_active = False
            self._middle_pan_last_viewport = None
            self._set_polygon_overlay_hide_hold("middle", False)
            event.accept()
            return
        if self._tool == EditorTool.PAN:
            super().mouseReleaseEvent(event)
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._tool == EditorTool.SELECT
            and self._drag_kind is None
            and self._select_press_polygon_id is not None
        ):
            self._select_press_polygon_id = None
            self._select_press_start = None
            event.accept()
            return
        if self._drag_kind == "rect_polygon":
            event.accept()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and self._drag_kind is not None:
            if self._drag_kind == "brush":
                release_pos = self.mapToScene(event.position().toPoint())
                self._commit_brush_drag(release_pos)
            elif self._drag_kind == "ruler" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                target_pos = self._ruler_target(self._drag_start_scene_pos, release_pos, event.modifiers())
                measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
                self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
                self.rulerMeasurementChanged.emit(measurement_text)
            elif self._drag_kind == "delete_area" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._drag_start_scene_pos, release_pos)
                self._commit_delete_vertices_in_area(rect)
            elif self._drag_kind == "select_area" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._drag_start_scene_pos, release_pos).normalized()
                self._editor_scene.clear_preview_rect()
                self.contactMultiSelectionApplyStarted.emit()
                if rect.width() < self._scene_tolerance(3) and rect.height() < self._scene_tolerance(3):
                    polygon_id = self._editor_scene.polygon_at(release_pos)
                    self._editor_scene.select_polygon(
                        polygon_id, additive=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    )
                else:
                    self._editor_scene.select_polygons_in_rect(
                        rect,
                        additive=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
                    )
                self.contactMultiSelectionFinished.emit(self._selected_contact_count())
            elif self._drag_kind == "antialias_area" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._drag_start_scene_pos, release_pos).normalized()
                self._editor_scene.clear_preview_rect()
                if rect.width() < self._scene_tolerance(3) and rect.height() < self._scene_tolerance(3):
                    polygon_id = self._editor_scene.polygon_at(release_pos)
                    if polygon_id is not None:
                        self._editor_scene.antialias_polygon(polygon_id, self._antialias_grade)
                else:
                    self._editor_scene.antialias_polygons_in_rect(rect, self._antialias_grade)
            elif self._drag_kind == "image_region" and self._drag_start_scene_pos is not None:
                rect = QRectF(self._drag_start_scene_pos, self.mapToScene(event.position().toPoint())).normalized()
                image_rect = self._editor_scene.main_image_rect()
                clipped = rect.intersected(image_rect)
                self._editor_scene.clear_preview_rect()
                if clipped.width() >= 2.0 and clipped.height() >= 2.0:
                    self.imageRegionSelected.emit(clipped.x(), clipped.y(), clipped.width(), clipped.height())
            elif (
                self._drag_kind == "vertex"
                and self._drag_polygon_id is not None
                and self._drag_vertex_index is not None
                and self._drag_origin_points is not None
                and self._drag_polygons_snapshot is not None
            ):
                profile_timings: dict[str, float] = {}
                profile_total_start = perf_counter()
                profiler = cProfile.Profile() if vertex_move_profiling_enabled() else None
                profiler_enabled = False
                if profiler is not None:
                    profiler_enabled = try_enable_profiler(profiler)
                new_points = list(
                    self._drag_preview_points
                    or self._editor_scene.polygon_edit_preview_points(self._drag_polygon_id)
                    or self._drag_origin_points
                )
                old_point = self._drag_origin_points[self._drag_vertex_index]
                new_point = (
                    new_points[self._drag_vertex_index]
                    if new_points and self._drag_vertex_index < len(new_points)
                    else old_point
                )
                if (
                    new_points
                    and self._drag_vertex_index < len(new_points)
                    and _points_different(old_point, new_point)
                ):
                    phase_start = perf_counter()
                    if not is_valid_closed_polygon_vertex_move(new_points, self._drag_vertex_index):
                        profile_timings["validate"] = (perf_counter() - phase_start) * 1000.0
                        self._editor_scene.preview_vertex_move(
                            self._drag_polygon_id, self._drag_vertex_index, QPointF(old_point[0], old_point[1])
                        )
                        self._editor_scene.warn_invalid_polygon_geometry()
                    else:
                        profile_timings["validate"] = (perf_counter() - phase_start) * 1000.0
                        phase_start = perf_counter()
                        processed, accepted, focus_id = postprocess_vertex_move_edit(
                            self._drag_polygons_snapshot,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                            vertex_index=self._drag_vertex_index,
                            new_point=new_point,
                        )
                        profile_timings["postprocess"] = (perf_counter() - phase_start) * 1000.0
                        if not accepted:
                            self._editor_scene.preview_vertex_move(
                                self._drag_polygon_id,
                                self._drag_vertex_index,
                                QPointF(old_point[0], old_point[1]),
                            )
                            self._editor_scene.warn_invalid_polygon_geometry()
                        else:
                            phase_start = perf_counter()
                            if focus_id is None:
                                focus_id = self._drag_polygon_id
                            profile_timings["focus"] = (perf_counter() - phase_start) * 1000.0
                            phase_start = perf_counter()
                            committed = self._editor_scene._try_commit_single_polygon_points_change(
                                self._drag_polygons_snapshot,
                                processed,
                                description="Move vertex",
                                select_polygon_id=focus_id,
                            )
                            if not committed:
                                self._editor_scene._push_polygon_set_change(
                                    self._drag_polygons_snapshot,
                                    processed,
                                    "Move vertex",
                                    select_polygon_id=focus_id,
                                )
                            profile_timings["undo_push"] = (perf_counter() - phase_start) * 1000.0
                    profile_timings["total_wall"] = (perf_counter() - profile_total_start) * 1000.0
                    if profiler_enabled and profiler is not None:
                        try_disable_profiler(profiler)
                    self._emit_vertex_move_profile(
                        profile_timings,
                        polygon_count=len(self._drag_polygons_snapshot),
                        vertex_count=sum(len(polygon.points) for polygon in self._drag_polygons_snapshot),
                        profiler=profiler if profiler_enabled else None,
                    )
                elif profiler_enabled and profiler is not None:
                    try_disable_profiler(profiler)
            elif (
                self._drag_kind == "edge"
                and self._drag_polygon_id is not None
                and self._drag_edge_index is not None
                and self._drag_origin_points is not None
                and self._drag_polygons_snapshot is not None
                and self._drag_start_scene_pos is not None
            ):
                release_pos = self.mapToScene(event.position().toPoint())
                delta = (
                    release_pos.x() - self._drag_start_scene_pos.x(),
                    release_pos.y() - self._drag_start_scene_pos.y(),
                )
                new_points = list(
                    self._drag_preview_points
                    or self._editor_scene.polygon_edit_preview_points(self._drag_polygon_id)
                    or self._drag_origin_points
                )
                if new_points and _polygon_points_different(self._drag_origin_points, new_points):
                    if not is_valid_closed_polygon_edge_move(new_points, self._drag_edge_index):
                        self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                        self._editor_scene.warn_invalid_polygon_geometry()
                    else:
                        trial = apply_edge_translation_to_clone(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            self._drag_edge_index,
                            delta,
                        )
                        processed, accepted, _changed = postprocess_changed_polygon_edit(
                            trial,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                        )
                        if accepted:
                            processed = collapse_redundant_vertices_in_polygons(processed)
                            if self._vector_geometry_settings.merge_overlapping_on_edit:
                                processed = merge_overlapping_root_families_near_polygons(
                                    processed,
                                    self._drag_polygon_id,
                                )
                            processed = collapse_redundant_vertices_in_polygons(processed)
                        if not accepted:
                            self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                            self._editor_scene.warn_invalid_polygon_geometry()
                        else:
                            focus_id = resolve_focus_id_after_geometry_pass(
                                self._drag_polygons_snapshot,
                                self._drag_polygon_id,
                                processed,
                            )
                            self._editor_scene._push_polygon_set_change(
                                self._drag_polygons_snapshot,
                                processed,
                                "Move edge",
                                select_polygon_id=focus_id,
                            )
            elif (
                self._drag_kind == "polygon"
                and self._drag_polygon_id is not None
                and self._drag_origin_points is not None
                and (
                    self._drag_polygon_is_contact
                    or self._drag_polygons_snapshot is not None
                )
            ):
                new_points = list(
                    self._drag_preview_points
                    or self._editor_scene.polygon_edit_preview_points(self._drag_polygon_id)
                    or self._drag_origin_points
                )
                if new_points and _polygon_points_different(self._drag_origin_points, new_points):
                    if not is_valid_closed_polygon_ring(new_points):
                        contact_drag_status = "rejected"
                        self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                        self._editor_scene.warn_invalid_polygon_geometry()
                    elif self._drag_polygon_is_contact:
                        contact_drag_status = "displayed"
                        self.undo_stack.push(
                            MovePolygonCommand(
                                self._editor_scene,
                                self._drag_polygon_id,
                                self._drag_origin_points,
                                new_points,
                            )
                        )
                    else:
                        assert self._drag_polygons_snapshot is not None
                        trial = apply_polygon_points_to_clone(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            new_points,
                        )
                        processed, accepted, _changed = postprocess_changed_polygon_edit(
                            trial,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                        )
                        if accepted:
                            processed = collapse_redundant_vertices_in_polygons(processed)
                            if self._vector_geometry_settings.merge_overlapping_on_edit:
                                processed = merge_overlapping_root_families_near_polygons(
                                    processed,
                                    self._drag_polygon_id,
                                )
                            processed = collapse_redundant_vertices_in_polygons(processed)
                        if not accepted:
                            contact_drag_status = "rejected"
                            self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                            self._editor_scene.warn_invalid_polygon_geometry()
                        else:
                            contact_drag_status = "displayed"
                            focus_id = resolve_focus_id_after_geometry_pass(
                                self._drag_polygons_snapshot,
                                self._drag_polygon_id,
                                processed,
                            )
                            self._editor_scene._push_polygon_set_change(
                                self._drag_polygons_snapshot,
                                processed,
                                "Move polygon",
                                select_polygon_id=focus_id,
                            )
            self._drag_kind = None
            self._drag_polygon_id = None
            self._drag_vertex_index = None
            self._drag_edge_index = None
            self._drag_origin_points = None
            self._drag_preview_points = None
            self._drag_start_scene_pos = None
            self._drag_polygons_snapshot = None
            self._drag_polygon_is_contact = False
            self._drag_erases = False
            self._brush_pan_guard = False
            self._editor_scene.cancel_polygon_edit_preview()
            self._editor_scene.clear_move_target_preview()
            self._update_tool_cursors()
            if contact_drag_commit_started_at is not None:
                self._finish_contact_drag_profile(
                    contact_drag_status,
                    commit_ms=(
                        perf_counter() - contact_drag_commit_started_at
                    )
                    * 1000.0,
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)
