"""processing_mixin: load cif overlay polygons."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    PolygonData,
    QTimer,
    load_polygons_vector,
)

from contour.widget_parts.processing_mixin_parts.module_state import _LOGGER


class LoadCifOverlayPolygonsMixin:
    def _flush_pending_thumbnail_grid_rebuild(self: Any) -> None:
        if not self._frame_matrix_enabled():
            self._pending_thumbnail_rebuild_after_vectors = False
            self._thumbnail_flush_retry_count = 0
            return
        if not bool(getattr(self, "_pending_thumbnail_rebuild_after_vectors", False)):
            return
        if getattr(self, "_frame_load_running_path", None) is not None or getattr(self, "_loading_image_path", None):
            attempts = int(getattr(self, "_thumbnail_flush_retry_count", 0)) + 1
            self._thumbnail_flush_retry_count = attempts
            if attempts > 600:
                self._pending_thumbnail_rebuild_after_vectors = False
                self._thumbnail_flush_retry_count = 0
                QTimer.singleShot(0, self._rebuild_thumbnail_grid)
                return
            QTimer.singleShot(100, self._flush_pending_thumbnail_grid_rebuild)
            return
        self._pending_thumbnail_rebuild_after_vectors = False
        self._thumbnail_flush_retry_count = 0
        QTimer.singleShot(0, self._rebuild_thumbnail_grid)

    def _mark_thumbnail_grid_rebuild_pending(self: Any) -> None:
        if not self._frame_matrix_enabled():
            self._pending_thumbnail_rebuild_after_vectors = False
            self._thumbnail_flush_retry_count = 0
            return
        self._pending_thumbnail_rebuild_after_vectors = True
        self._thumbnail_flush_retry_count = 0

    def _defer_frame_chrome_updates(self: Any, image_path: str) -> None:
        normalized = str(Path(image_path))
        session = self._frame_switch_profile_for_path(normalized)
        if session is not None:
            session.mark_pending("deferred_chrome")
        if not hasattr(self, "_frame_chrome_update_timer"):
            self._frame_chrome_update_timer = QTimer(self)
            self._frame_chrome_update_timer.setSingleShot(True)
            self._frame_chrome_update_timer.timeout.connect(self._apply_deferred_frame_chrome_updates)
        self._pending_frame_chrome_path = normalized
        self._frame_chrome_update_timer.stop()
        self._frame_chrome_update_timer.start(0 if not self._uses_large_frame_list() else 32)

    def _apply_deferred_frame_chrome_updates(self: Any) -> None:
        image_path = str(getattr(self, "_pending_frame_chrome_path", "") or "")
        if not image_path or str(Path(self._workspace.current_image_path or "")) != image_path:
            return
        session = self._frame_switch_profile_for_path(image_path)
        if session is not None:
            session.complete_pending("deferred_chrome", suffix="_apply")
        self._image_list_model.invalidate_path(image_path)
        self._update_thumbnail_grid_selection()
        if not self._thumbnail_loading_blocked():
            self._schedule_visible_thumbnail_loads()
            if self._frame_matrix_thumbnails_enabled() and hasattr(self, "_resume_thumbnail_radial_fill"):
                self._resume_thumbnail_radial_fill()

    def _schedule_thumbnail_grid_rebuild(self: Any, *, force: bool = False) -> None:
        if not self._frame_matrix_enabled():
            self._disable_frame_matrix_runtime()
            return
        if force and not getattr(self, "_frame_load_running_path", None) and not getattr(self, "_loading_image_path", None):
            self._pending_thumbnail_rebuild_after_vectors = False
            self._thumbnail_flush_retry_count = 0
            QTimer.singleShot(0, self._rebuild_thumbnail_grid)
            return
        self._mark_thumbnail_grid_rebuild_pending()
        self._flush_pending_thumbnail_grid_rebuild()

    def _load_cif_overlay_polygons(self: Any, image_path: str) -> list[PolygonData]:
        if self._workspace.vectors_are_cleared(image_path):
            return []
        stem_key = Path(image_path).stem.lower()
        cif_path = self._find_matching_cif_path(image_path)
        if not cif_path:
            self._cif_load_failure_stems.discard(stem_key)
            return []
        try:
            if self._frame_switch_profiling_active():
                from ...infrastructure.cif_operation_profiler import cif_operation_profiling

                with cif_operation_profiling() as cif_timings:
                    referenced_image, image_size, polygons = load_polygons_vector(cif_path)
                session = self._frame_switch_profile_for_path(image_path)
                if session is not None:
                    session.merge_timings(cif_timings)
            else:
                referenced_image, image_size, polygons = load_polygons_vector(cif_path)
        except Exception as exc:
            _LOGGER.exception("Failed to load CIF %s for image %s", cif_path, image_path)
            self._cif_load_failure_stems.add(stem_key)
            self._append_log(self._tr("cif_load_failed_log", file_name=Path(cif_path).name, error=exc))
            return []
        self._cif_load_failure_stems.discard(stem_key)
        if referenced_image and Path(referenced_image).stem.lower() != Path(image_path).stem.lower():
            self._append_log(
                self._tr(
                    "cif_reference_name_diff_log",
                    file_name=Path(cif_path).name,
                    referenced_image=referenced_image,
                )
            )
        if image_size is not None:
            self._append_log(
                self._tr(
                    "cif_overlay_loaded_with_size_log",
                    file_name=Path(cif_path).name,
                    width=image_size[0],
                    height=image_size[1],
                    count=len(polygons),
                )
            )
        else:
            self._append_log(self._tr("cif_overlay_loaded_log", file_name=Path(cif_path).name, count=len(polygons)))
        return polygons
