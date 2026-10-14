"""processing_mixin: image list model row."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    FRAME_STATUS_ROLE,
    Path,
    QColor,
    Qt,
)


class ImageListModelRowMixin:
    def _image_list_model_row_paint_roles(self: Any, image_path: str) -> dict[int, object]:
        from ...application.frame_asset_sync import (
            background_hex_image_paint_status_for_theme,
            classify_image_side_paint_status,
            foreground_hex_image_paint_status_for_theme,
        )

        normalized = (
            image_path
            if image_path in self._image_path_to_index
            else str(Path(image_path))
        )
        cif_paths = getattr(self._workspace, "_cif_paths_by_stem", {})
        stem_lower = getattr(self, "_image_path_stem_lower", {}).get(normalized)
        if stem_lower is None:
            stem_lower = Path(normalized).stem.lower()
        has_vector = stem_lower in cif_paths
        extraction_enabled = self._is_extraction_mode_enabled()
        current_path = self._workspace.current_image_path
        if extraction_enabled or current_path == normalized:
            polygons_dirty = False if extraction_enabled else (
                self._workspace.image_has_changes(normalized)
                or normalized in self._restored_modified_image_paths
            )
        elif self._uses_large_frame_list():
            polygons_dirty = False
        else:
            polygons_dirty = (
                self._workspace.image_has_changes(normalized)
                or normalized in self._restored_modified_image_paths
            )
        painted = classify_image_side_paint_status(
            has_matching_cif=has_vector,
            vector_index_active=self._vector_index_active(),
            never_opened=True if extraction_enabled else normalized not in self._viewed_image_paths,
            polygons_dirty=polygons_dirty,
            persist_highlight=False if extraction_enabled else normalized in self._persisted_highlight_paths,
        )
        theme = getattr(self, "_ui_theme", "dark")
        hex_background = background_hex_image_paint_status_for_theme(painted, theme=theme)
        return {
            FRAME_STATUS_ROLE: painted.value,
            int(Qt.ItemDataRole.BackgroundRole): QColor(hex_background) if hex_background else None,
            int(Qt.ItemDataRole.ForegroundRole): QColor(
                foreground_hex_image_paint_status_for_theme(
                    painted,
                    has_matching_cif=has_vector,
                    theme=theme,
                )
            ),
        }

    def _image_list_model_item_data(self: Any, image_path: str, role: int):
        return self._image_list_model_row_paint_roles(image_path).get(int(role))

    def _set_image_list_paths(self: Any, normalized_paths: list[str]) -> None:
        paths = [str(Path(path)) for path in normalized_paths]
        if hasattr(self.image_list, "clear_manual_items"):
            self.image_list.clear_manual_items()
        self._image_paths_ordered = paths
        self._image_path_to_index = {path: index for index, path in enumerate(paths)}
        stem_lower_by_path: dict[str, str] = {}
        path_by_stem_lower: dict[str, str] = {}
        for path in paths:
            stem_lower = Path(path).stem.lower()
            stem_lower_by_path[path] = stem_lower
            path_by_stem_lower.setdefault(stem_lower, path)
        self._image_path_stem_lower = stem_lower_by_path
        self._image_path_by_stem_lower = path_by_stem_lower
        self._image_list_model.set_paths(paths)
