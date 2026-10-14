"""processing_mixin: offer repair invalid polygon."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QMessageBox,
    VectorGeometrySettings,
    polygons_needing_repair,
)


class OfferRepairInvalidPolygonMixin:
    def _offer_repair_invalid_polygon_descriptions(self: Any, expected_path: str = "") -> None:
        if bool(getattr(self, "_invalid_polygon_offer_active", False)):
            return
        if bool(getattr(self, "_closing", False)):
            return
        if bool(getattr(self, "_work_simulation_running", False)):
            return
        current = str(Path(self._workspace.current_image_path or ""))
        if expected_path and current != str(Path(expected_path)):
            return
        from os import environ

        if environ.get("PYTEST_CURRENT_TEST"):
            return
        if not self.isVisible():
            return
        polygons = self.polygon_editor.get_polygons()
        reasons_by_id = None
        if hasattr(self.polygon_editor, "polygons_needing_repair_map"):
            reasons_by_id = self.polygon_editor.polygons_needing_repair_map()
        if not reasons_by_id:
            reasons_by_id = self._repair_reasons_for_image_path(current) or {}
        invalid_count = len(reasons_by_id)
        if invalid_count <= 0:
            return
        reasons_text = self._format_invalid_polygon_description_reasons(
            polygons,
            reasons_by_id=reasons_by_id,
        )
        dismissed: set[str] = getattr(self, "_invalid_polygon_prompt_dismissed", set())
        if current in dismissed:
            return
        self._invalid_polygon_offer_active = True
        try:
            answer = QMessageBox.question(
                self,
                self._tr(
                    "invalid_polygon_description_title",
                    "Некорректное описание полигона" if self._ui_language == "ru" else "Incorrect polygon description",
                ),
                self._tr(
                    "invalid_polygon_description_prompt",
                    "Найдено полигонов с некорректным описанием: {count}. Причина: {reasons}. "
                    "Они выделены красным. Исправить топологию?"
                    if self._ui_language == "ru"
                    else "Found {count} incorrectly described polygon(s). Reason: {reasons}. "
                    "They are marked in red. Repair the topology?",
                    count=invalid_count,
                    reasons=reasons_text,
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
        finally:
            self._invalid_polygon_offer_active = False
        if answer != QMessageBox.StandardButton.Yes:
            dismissed.add(current)
            self._invalid_polygon_prompt_dismissed = dismissed
            self._append_log(
                self._tr(
                    "invalid_polygon_description_kept_log",
                    "Оставлены некорректные полигоны: {count}. Причина: {reasons}."
                    if self._ui_language == "ru"
                    else "Left {count} incorrectly described polygon(s). Reason: {reasons}.",
                    count=invalid_count,
                    reasons=reasons_text,
                )
            )
            return
        repaired = self.polygon_editor.repair_invalid_polygon_descriptions()
        if not repaired:
            self._append_log(
                self._tr(
                    "invalid_polygon_description_unrepaired_log",
                    "Не удалось автоматически исправить некорректные полигоны."
                    if self._ui_language == "ru"
                    else "Could not automatically repair incorrectly described polygons.",
                )
            )
            return
        self._sync_editor_polygons_to_current_workspace()
        settings = (
            self._vector_geometry_settings_from_widgets()
            if hasattr(self, "_vector_geometry_settings_from_widgets")
            else VectorGeometrySettings()
        )
        remaining = len(
            polygons_needing_repair(self.polygon_editor.get_polygons(), settings)
        )
        state = self._workspace.current_state
        if state is not None:
            state.polygons_needing_repair = {}
        if hasattr(self.polygon_editor, "apply_polygons_needing_repair"):
            self.polygon_editor.apply_polygons_needing_repair({})
        self._append_log(
            self._tr(
                "invalid_polygon_description_repaired_log",
                "Исправлено некорректных полигонов: {count}. Причина: {reasons}."
                if self._ui_language == "ru"
                else "Repaired incorrectly described polygons: {count}. Reason: {reasons}.",
                count=invalid_count - remaining,
                reasons=reasons_text,
            )
        )
