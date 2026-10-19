"""processing_mixin: save current result."""
from __future__ import annotations

import os
from typing import Any

from contour.widget_parts._imports import (
    Path,
    PolygonData,
    SaveOptions,
    save_result_bundle,
)


class SaveCurrentResultMixin:
    def save_current_result(
        self: Any,
        output_directory: str | None = None,
        save_options: SaveOptions | None = None,
        *,
        polygons: list[PolygonData] | None = None,
    ) -> dict[str, str]:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            self._append_log(self._tr("nothing_to_save_log"))
            return {}
        target_directory = output_directory or self.output_dir_edit.text().strip()
        if not target_directory:
            self._append_log(self._tr("output_directory_not_set_log"))
            return {}
        if polygons is not None:
            current_polygons = [polygon.clone() for polygon in polygons]
        elif self._editor_polygons_are_current_frame():
            current_polygons = self.get_polygons()
        else:
            current_polygons = [polygon.clone() for polygon in current_state.polygons]
        self._workspace.update_current_polygons(current_polygons)
        had_vector_edits = self._workspace.current_image_has_changes()
        contour_settings = self._current_contour_settings()
        polygons_to_save = current_polygons
        if contour_settings.object_type == "via":
            polygons_to_save = self._uniform_contact_polygons_for_save(
                current_polygons,
                diameter=int(contour_settings.via_output_diameter),
            )
        saved_files = save_result_bundle(
            output_directory=target_directory,
            image_path=current_image_path,
            polygons=polygons_to_save,
            source_image=current_state.source_image,
            display_settings=self._display_settings,
            save_options=save_options or self._current_save_options(),
            metadata={
                "contour_settings": contour_settings.to_dict(),
                "pipeline": self.get_pipeline(),
            },
        )
        if saved_files:
            saved_cif_path = saved_files.get("cif")
            linked_cif_path = current_state.loaded_cif_path or self._find_matching_cif_path(current_image_path)
            if (
                saved_cif_path
                and linked_cif_path
                and os.path.normcase(os.path.abspath(saved_cif_path))
                == os.path.normcase(os.path.abspath(linked_cif_path))
            ):
                self._workspace.refresh_cif_revision(current_image_path, saved_cif_path)
            self._append_log(self._tr("saved_result_log", saved_files=saved_files))
            self._handle_gamification_after_save(
                image_path=current_image_path,
                state=current_state,
                polygons=current_polygons,
                saved_files=saved_files,
                had_vector_edits=had_vector_edits,
            )
            saved_key = str(Path(current_image_path))
            if had_vector_edits:
                self._persisted_highlight_paths.add(saved_key)
            self._workspace.sync_polygon_reference_to_current(saved_key)
            self._update_frame_item_status(current_image_path)
            self._update_vector_edit_status_label()
        return saved_files
