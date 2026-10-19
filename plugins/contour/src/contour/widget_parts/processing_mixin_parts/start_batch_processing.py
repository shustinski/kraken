"""processing_mixin: start batch processing."""
from __future__ import annotations

from typing import Any

from contour.widget_parts._imports import (
    BatchStartRequest,
    PolygonData,
    mp,
)


class StartBatchProcessingMixin:
    @staticmethod
    def _uniform_contact_polygons_for_save(
        polygons: list[PolygonData],
        *,
        diameter: int,
    ) -> list[PolygonData]:
        """Return save-only copies with every contact represented by one square size."""

        size = max(1, int(diameter))
        half = float(size) * 0.5
        normalized: list[PolygonData] = []
        for source in polygons:
            polygon = source.clone()
            if str(polygon.category).lower() != "via" or not polygon.points:
                normalized.append(polygon)
                continue
            xs = [float(point[0]) for point in polygon.points]
            ys = [float(point[1]) for point in polygon.points]
            center_x = 0.5 * (min(xs) + max(xs))
            center_y = 0.5 * (min(ys) + max(ys))
            x0 = round(center_x - half)
            y0 = round(center_y - half)
            x1 = x0 + size
            y1 = y0 + size
            polygon.points = [
                (x0, y0),
                (x1, y0),
                (x1, y1),
                (x0, y1),
            ]
            polygon.bbox = (x0, y0, size, size)
            polygon.area = float(size * size)
            polygon.perimeter = float(4 * size)
            polygon.shape_hint = "box"
            normalized.append(polygon)
        return normalized

    def _redraw_recognized_contacts_with_diameter(self: Any, diameter: int) -> None:
        current_path = self._workspace.current_image_path
        current_state = self._workspace.current_state
        states_by_id = {
            id(state): (image_path, state)
            for image_path, state in self._workspace.cached_states()
        }
        if current_path is not None and current_state is not None:
            states_by_id[id(current_state)] = (current_path, current_state)

        for image_path, state in states_by_id.values():
            original = list(state.polygons or [])
            resized = self._recognized_contact_polygons_with_diameter(
                original,
                diameter=diameter,
            )
            if all(
                original_polygon.points == resized_polygon.points
                for original_polygon, resized_polygon in zip(
                    original,
                    resized,
                    strict=True,
                )
            ):
                continue
            state.polygons = resized
            state.polygons_dirty = None
            if image_path == current_path:
                self._workspace.update_current_polygons(resized)
                self._sync_polygons_to_editor(image_path, resized)
                self._update_frame_item_status(image_path)

        if hasattr(self, "_request_neighbor_frame_sync"):
            self._request_neighbor_frame_sync(delay_ms=0)

    def _recognized_contact_polygons_with_diameter(
        self: Any,
        polygons: list[PolygonData],
        *,
        diameter: int,
    ) -> list[PolygonData]:
        recognized_ids = {
            polygon.id
            for polygon in polygons
            if str(polygon.category).lower() == "via"
            and polygon.recognition_score is not None
        }
        if not recognized_ids:
            return [polygon.clone() for polygon in polygons]
        normalized = self._uniform_contact_polygons_for_save(
            polygons,
            diameter=diameter,
        )
        return [
            normalized_polygon
            if original_polygon.id in recognized_ids
            else original_polygon.clone()
            for original_polygon, normalized_polygon in zip(
                polygons,
                normalized,
                strict=True,
            )
        ]

    def start_batch_processing(
        self: Any,
        image_paths: list[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        if self._batch_controller.is_running:
            self._append_log(self._tr("batch_already_running_log"))
            return
        if getattr(self, "_antialias_running", False):
            self._append_log(
                self._tr(
                    "antialias_already_running_log",
                    "Сглаживание векторов уже выполняется."
                    if self._ui_language == "ru"
                    else "Vector antialiasing is already running.",
                )
            )
            return
        paths = image_paths or list(self._workspace.image_paths)
        if not paths:
            self._append_log(self._tr("batch_no_images_log"))
            return
        output_directory = self.output_dir_edit.text().strip() or None
        save_options = self._current_save_options()
        started = self._batch_controller.start(
            BatchStartRequest(
                image_paths=list(paths),
                pipeline_config=self.get_pipeline(),
                contour_settings=self._current_contour_settings(),
                display_settings=self._display_settings,
                save_options=save_options,
                output_directory=output_directory,
                max_workers=mp.cpu_count() if max_workers is None else max(1, int(max_workers)),
                chunk_size=max(1, int(getattr(self, "batch_chunk_size", 16))),
                )
            )
        if not started:
            return
        self._show_batch_progress(len(paths))
        self._set_progress_status("batch_started_status")

    def stop_batch_processing(self: Any) -> None:
        self._batch_controller.stop()
        self._cancel_antialias_job()
        self._cancel_fix_internal_contours_job()
