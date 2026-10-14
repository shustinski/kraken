"""processing_mixin: enqueue prefetch frame."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    FrameLoadPayload,
    FrameLoadRunnable,
    Path,
)


class EnqueuePrefetchFrameMixin:
    def _enqueue_prefetch_frame(
        self: Any,
        image_path: str,
        generation: int,
        *,
        vectors_only: bool = False,
        neighborhood_set: frozenset[str] | None = None,
    ) -> None:
        self._prefetch_request_serial = int(getattr(self, "_prefetch_request_serial", 0)) + 1
        request_id = self._prefetch_request_serial
        self._prefetch_queued_paths.add(image_path)
        active_neighborhood = neighborhood_set or frozenset(self._prefetch_neighborhood_paths())

        runnable = FrameLoadRunnable(
            request_id,
            image_path,
            load_source_image=None if vectors_only else self._load_scene_source_image_cached,
            load_cif_overlay=self._load_cif_overlay_polygons,
            load_vectors=True,
            vectors_only=vectors_only,
            scan_repair=None,
        )

        def _on_result(_req_id: int, payload_obj: object) -> None:
            if bool(getattr(self, "_closing", False)):
                return
            self._prefetch_queued_paths.discard(image_path)
            if generation != int(getattr(self, "_prefetch_generation", 0)):
                latest = getattr(self, "_prefetch_neighborhood_set", active_neighborhood)
                if image_path not in latest:
                    return
            if not isinstance(payload_obj, FrameLoadPayload):
                return
            loaded_path = payload_obj.image_path
            if loaded_path not in getattr(self, "_image_path_to_index", {}):
                loaded_path = str(Path(payload_obj.image_path))
            latest = getattr(self, "_prefetch_neighborhood_set", active_neighborhood)
            if loaded_path not in latest:
                return
            if self._workspace.image_has_changes(loaded_path):
                return
            cif_path = self._find_matching_cif_path(loaded_path)
            repair_reasons = self._payload_repair_reasons(payload_obj)
            if payload_obj.vectors_only:
                if not self._workspace.has_cached_source(loaded_path):
                    return
                if not self._workspace.needs_vector_overlay(loaded_path):
                    return
                self._workspace.apply_frame_vectors(
                    loaded_path,
                    polygons=list(payload_obj.polygons),
                    loaded_cif_path=cif_path,
                    polygons_needing_repair=repair_reasons,
                )
                return
            if self._workspace.has_cached_source(loaded_path):
                # Source arrived from another path; still fill missing overlays.
                if self._workspace.needs_vector_overlay(loaded_path):
                    self._workspace.apply_frame_vectors(
                        loaded_path,
                        polygons=list(payload_obj.polygons),
                        loaded_cif_path=cif_path,
                        polygons_needing_repair=repair_reasons,
                    )
                return
            self._workspace.apply_loaded_frame(
                loaded_path,
                source_image=payload_obj.source_image,
                polygons=list(payload_obj.polygons),
                make_current=False,
                loaded_cif_path=cif_path,
                polygons_needing_repair=repair_reasons,
            )

        def _on_error(_req_id: int, failed_path: str, _message: str) -> None:
            self._prefetch_queued_paths.discard(failed_path)

        runnable.signals.result.connect(_on_result)
        runnable.signals.error.connect(_on_error)
        self._prefetch_thread_pool.start(runnable)
