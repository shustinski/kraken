"""processing_mixin: handle gamification after save."""
from __future__ import annotations
import hashlib
from time import time_ns
from typing import Any
from contour.widget_parts._imports import (
    CorrectionEvent,
    CorrectionType,
    ImageProcessingState,
    Path,
    PolygonData,
    RewardEventType,
)

from contour.widget_parts.processing_mixin_parts.module_state import _LOGGER


class HandleGamificationAfterSaveMixin:
    def _handle_gamification_after_save(
        self: Any,
        *,
        image_path: str,
        state: ImageProcessingState,
        polygons: list[PolygonData],
        saved_files: dict[str, str],
        had_vector_edits: bool,
    ) -> None:
        if not hasattr(self, "_gamification_service"):
            return
        try:
            reference_polygons = list(getattr(state, "reference_polygons", []))
            event = CorrectionEvent(
                correction_id=self._build_gamification_correction_id(
                    image_path=image_path,
                    polygons=polygons,
                    saved_files=saved_files,
                ),
                image_id=str(Path(image_path)),
                has_real_mask_changes=bool(had_vector_edits),
                accepted_without_changes=not bool(had_vector_edits),
                correction_type=CorrectionType.UNKNOWN if had_vector_edits else None,
                edit_count=None,
                added_objects=max(0, len(polygons) - len(reference_polygons)),
                removed_objects=max(0, len(reference_polygons) - len(polygons)),
            )
            results = self._gamification_service.handle_correction_event(event)
            if hasattr(self, "gamification_panel"):
                if results:
                    self.gamification_panel.set_last_results(results)
                elif event.accepted_without_changes:
                    self.gamification_panel.react_to_event(RewardEventType.IMAGE_ACCEPTED_WITHOUT_CHANGES)
            else:
                for result in results:
                    if result.success and result.message:
                        self._append_log(result.message)
        except Exception as exc:
            _LOGGER.exception("Gamification event handling failed")
            self._append_log(f"Gamification error: {exc}")

    def _handle_gamification_ui_event(self: Any, event_type: RewardEventType) -> None:
        try:
            if hasattr(self, "gamification_panel"):
                self.gamification_panel.react_to_event(event_type)
        except Exception as exc:
            _LOGGER.exception("Gamification UI event handling failed")
            self._append_log(f"Gamification UI error: {exc}")

    @staticmethod
    def _build_gamification_correction_id(
        *,
        image_path: str,
        polygons: list[PolygonData],
        saved_files: dict[str, str],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(str(Path(image_path)).encode("utf-8", errors="replace"))
        for polygon in sorted(polygons, key=lambda item: int(item.id)):
            digest.update(
                (
                    f"|{polygon.id}:{polygon.is_hole}:{polygon.parent_id}:"
                    f"{polygon.category}:{polygon.shape_hint}:"
                    f"{polygon.bbox}:{len(polygon.points)}"
                ).encode("utf-8", errors="replace")
            )
            for x_coord, y_coord in polygon.points:
                digest.update(f":{float(x_coord):.6f},{float(y_coord):.6f}".encode("ascii"))
        for key, value in sorted(saved_files.items()):
            digest.update(f"|{key}:{value}".encode("utf-8", errors="replace"))
        return f"{Path(image_path).stem}:{time_ns()}:{digest.hexdigest()[:16]}"
