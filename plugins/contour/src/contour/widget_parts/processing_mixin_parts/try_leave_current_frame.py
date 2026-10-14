"""processing_mixin: try leave current frame."""
from __future__ import annotations
import json
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QMessageBox,
    QTimer,
    TransitionPromptChoice,
    navigation_allowed_after_autosave_attempt,
    navigation_allowed_after_prompt,
)


class TryLeaveCurrentFrameMixin:
    def _warn_transition_blocked_after_failed_autosave(self: Any) -> None:
        detail = str(getattr(self, "_last_vector_persist_error", "") or "").strip()
        body = self._tr(
            "autosave_transition_blocked_text",
            "Автосохранение не выполнено; переход отменён, данные не потеряны."
            if self._ui_language == "ru"
            else "Autosave failed; navigation cancelled — your edits were kept.",
        )
        if detail:
            body = f"{body}\n\n{detail}"
        QMessageBox.warning(
            self,
            self._tr(
                "autosave_transition_blocked_title",
                "Не удалось сохранить" if self._ui_language == "ru" else "Save failed",
            ),
            body,
        )

    def _warn_transition_blocked_after_failed_manual_save(self: Any) -> None:
        detail = str(getattr(self, "_last_vector_persist_error", "") or "").strip()
        body = self._tr(
            "manual_save_transition_blocked_text",
            "Сохранение не выполнено; переход отменён, данные не потеряны."
            if self._ui_language == "ru"
            else "Save failed; navigation cancelled — your edits were kept.",
        )
        if detail:
            body = f"{body}\n\n{detail}"
        QMessageBox.warning(
            self,
            self._tr(
                "manual_save_transition_blocked_title",
                "Не удалось сохранить" if self._ui_language == "ru" else "Save failed",
            ),
            body,
        )

    def _try_leave_current_frame(self: Any) -> bool:
        if self._is_extraction_mode_enabled():
            return True
        self._sync_editor_polygons_to_current_workspace()
        dirty = self._workspace.current_image_has_changes()
        if not dirty:
            self._update_vector_edit_status_label(sync_editor=False)
            return True

        autosave_on = bool(
            hasattr(self, "autosave_on_frame_transition_checkbox")
            and self.autosave_on_frame_transition_checkbox.isChecked()
        )
        if autosave_on:
            save_ok = self._persist_current_overlay_changes()
            allowed = navigation_allowed_after_autosave_attempt(dirty=True, save_ok=save_ok)
            if not allowed:
                self._warn_transition_blocked_after_failed_autosave()
            return allowed

        choice = self._prompt_transition_vector_save_dialog()
        if choice == TransitionPromptChoice.CANCEL:
            return False

        if choice == TransitionPromptChoice.DISCARD:
            self._discard_current_vector_changes()
            return True

        save_ok = self._persist_current_overlay_changes()
        allowed = navigation_allowed_after_prompt(dirty=True, choice=TransitionPromptChoice.SAVE, save_ok=save_ok)
        if not allowed:
            self._warn_transition_blocked_after_failed_manual_save()
        return allowed

    def confirm_ok_to_leave_current_vectors(self: Any) -> bool:
        """Ask before closing or reloading images when the active frame has unsaved vector edits."""

        return self._try_leave_current_frame()

    def _source_preview_active(self: Any) -> bool:
        checkbox = getattr(self, "show_applied_filters_checkbox", None)
        return bool(
            (checkbox is not None and not checkbox.isChecked())
            or getattr(self, "_show_source_while_filter_hotkey_held", False)
        )

    def _on_show_applied_filters_changed(self: Any, _checked: bool) -> None:
        self._refresh_current_display_image_only(preserve_view=True)

    def _editor_display_cache_key(self: Any, image_path: str, state) -> tuple[str, str, str]:
        if state is not None and self._source_preview_active() and state.source_image is not None:
            # The source frame is already cached while the image is loaded.
            # Reuse it when the processed-filter display is disabled.
            return (str(Path(image_path)), "source", "")
        if state is not None and state.preprocessed_image is not None:
            pipeline_key = json.dumps(
                getattr(state, "pipeline_config", None) or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            return (str(Path(image_path)), "preprocessed", pipeline_key)
        return (str(Path(image_path)), "source", "")

    def _polygons_editor_signature(self: Any, image_path: str, polygons: list) -> tuple[object, ...]:
        geometry = []
        for polygon in polygons:
            points = getattr(polygon, "points", ())
            first_point = points[0] if points else None
            last_point = points[-1] if points else None
            geometry.append(
                (
                    getattr(polygon, "id", None),
                    len(points),
                    getattr(polygon, "bbox", None),
                    first_point,
                    last_point,
                    getattr(polygon, "category", None),
                    getattr(polygon, "shape_hint", None),
                    bool(getattr(polygon, "is_hole", False)),
                    getattr(polygon, "parent_id", None),
                )
            )
        return (str(Path(image_path)), tuple(geometry))

    def _editor_polygons_match_hole_topology(self: Any, editor_polygons: list, polygons: list) -> bool:
        if len(editor_polygons) != len(polygons):
            return False
        by_id = {getattr(polygon, "id", None): polygon for polygon in editor_polygons}
        if len(by_id) != len(polygons):
            return False
        for polygon in polygons:
            existing = by_id.get(getattr(polygon, "id", None))
            if existing is None:
                return False
            if bool(getattr(existing, "is_hole", False)) != bool(getattr(polygon, "is_hole", False)):
                return False
            if getattr(existing, "parent_id", None) != getattr(polygon, "parent_id", None):
                return False
        return True

    def _sync_polygons_to_editor(self: Any, image_path: str, polygons: list) -> None:
        signature = self._polygons_editor_signature(image_path, polygons)
        if getattr(self, "_editor_polygons_signature", None) == signature:
            editor_polygons = self.polygon_editor.get_polygons()
            if self._editor_polygons_match_hole_topology(editor_polygons, polygons):
                if hasattr(self.polygon_editor, "refresh_polygon_overlays"):
                    # Keep overlays in sync without rescanning repair geometry on frame switch.
                    scene = getattr(self.polygon_editor, "_editor_scene", None)
                    if scene is not None and hasattr(scene, "_refresh_all_items"):
                        scene._refresh_all_items(
                            rebuild_repair_cache=False,
                            defer_hit_paths=True,
                            defer_object_colors=True,
                        )
                    else:
                        self.polygon_editor.refresh_polygon_overlays()
                return
        repair_reasons = self._repair_reasons_for_image_path(image_path)
        self.polygon_editor.set_polygons(
            polygons,
            emit_signal=False,
            repair_reasons=repair_reasons,
            scan_repair=False,
            defer_hit_paths=True,
            defer_object_colors=True,
        )
        self._editor_polygons_signature = signature
        if self._frame_switch_profile_for_path(image_path) is None and hasattr(
            self.polygon_editor, "flush_deferred_object_colors"
        ):
            QTimer.singleShot(0, self.polygon_editor.flush_deferred_object_colors)

    def _repair_reasons_for_image_path(self: Any, image_path: str) -> dict[int, list[str]] | None:
        state = self._workspace.cached_state(image_path)
        if state is None:
            return None
        reasons = getattr(state, "polygons_needing_repair", None)
        if reasons is None:
            return None
        return {int(polygon_id): list(codes) for polygon_id, codes in reasons.items()}
