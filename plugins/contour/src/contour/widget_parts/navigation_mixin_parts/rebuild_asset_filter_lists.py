"""navigation_mixin: rebuild asset filter lists."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    ASSET_FILTER_LISTS_MAX_FRAMES,
    Path,
    QTimer,
    build_frame_asset_sets,
)


class RebuildAssetFilterListsMixin:
    def _rebuild_asset_filter_lists(self: Any) -> None:
        if not hasattr(self, "image_vector_list"):
            return
        if len(self._workspace.image_paths) > ASSET_FILTER_LISTS_MAX_FRAMES:
            self._cancel_asset_filter_list_build()
            return
        self._asset_list_build_generation += 1
        generation = self._asset_list_build_generation
        image_paths = [str(Path(path)) for path in self._workspace.image_paths]
        vector_items = sorted(self._workspace.cif_paths_by_stem.items(), key=lambda kv: kv[0].lower())
        sets = build_frame_asset_sets(image_paths, self._workspace.cif_paths_by_stem)
        list_widgets = (self.image_vector_list, self.image_only_list, self.vector_only_list)
        for list_widget in list_widgets:
            list_widget.clear()

        chunk_size = max(1, int(getattr(self, "_image_list_build_chunk_size", 250)))

        def _set_updates_enabled(enabled: bool) -> None:
            for widget in list_widgets:
                widget.setUpdatesEnabled(enabled)

        def _add_image_chunk(start: int) -> None:
            if generation != self._asset_list_build_generation:
                return
            end = min(len(image_paths), start + chunk_size)
            for list_widget in list_widgets:
                list_widget.blockSignals(True)
            _set_updates_enabled(False)
            try:
                for image_path in image_paths[start:end]:
                    stem = Path(image_path).stem.lower()
                    if stem in sets.image_and_vector_stems:
                        self.image_vector_list.addItem(self._make_asset_image_item(image_path))
                    elif stem in sets.image_only_stems:
                        self.image_only_list.addItem(self._make_asset_image_item(image_path))
            finally:
                _set_updates_enabled(True)
                for list_widget in list_widgets:
                    list_widget.blockSignals(False)
            if end < len(image_paths):
                QTimer.singleShot(0, lambda next_start=end: _add_image_chunk(next_start))
                return
            _add_vector_chunk(0)

        def _add_vector_chunk(start: int) -> None:
            if generation != self._asset_list_build_generation:
                return
            end = min(len(vector_items), start + chunk_size)
            self.vector_only_list.blockSignals(True)
            self.vector_only_list.setUpdatesEnabled(False)
            try:
                for stem, vector_path in vector_items[start:end]:
                    if stem.lower() in sets.vector_only_stems:
                        self.vector_only_list.addItem(self._make_asset_vector_item(stem, vector_path))
            finally:
                self.vector_only_list.setUpdatesEnabled(True)
                self.vector_only_list.blockSignals(False)
            if end < len(vector_items):
                QTimer.singleShot(0, lambda next_start=end: _add_vector_chunk(next_start))

        QTimer.singleShot(0, lambda: _add_image_chunk(0))
