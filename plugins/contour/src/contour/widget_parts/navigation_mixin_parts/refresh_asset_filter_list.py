"""navigation_mixin: refresh asset filter list."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    ASSET_FILTER_LISTS_MAX_FRAMES,
    LARGE_FRAME_COUNT_THRESHOLD,
    Path,
    QListWidgetItem,
    Qt,
    QTimer,
    classify_vector_side_status,
    paint_vector_row_item,
    time,
)


class RefreshAssetFilterListMixin:
    def _dialog_start_directory_from_value(self: Any, value: str | Path | None, fallback: str | Path | None = None) -> str:
        candidates = [value, fallback, Path.home()]
        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate).strip().strip("\"'")
            if not text:
                continue
            path = Path(text).expanduser()
            if path.is_dir():
                return str(path)
            if path.is_file() and path.parent.is_dir():
                return str(path.parent)
            if path.parent.is_dir():
                return str(path.parent)
        return str(Path.home())

    def _dialog_start_directory_from_line_edit(self: Any, line_edit, fallback: str | Path | None = None) -> str:
        text = line_edit.text() if hasattr(line_edit, "text") else ""
        return self._dialog_start_directory_from_value(text, fallback)

    def _on_sidebar_list_mode_changed(self: Any, index: int) -> None:
        if index < 0:
            return
        if hasattr(self, "sidebar_list_stack"):
            self.sidebar_list_stack.setCurrentIndex(0 if index == 0 else 1)
        if index == 1:
            # Defer arming: the mouse release that closes the combo popup is often
            # delivered to the vector list in the *next* event-loop tick.
            def _arm_suppress() -> None:
                self._vectors_list_ignore_navigate_until = time.monotonic() + 0.55

            QTimer.singleShot(0, _arm_suppress)
        else:
            self._vectors_list_ignore_navigate_until = 0.0

    def _image_path_for_cif_stem(self: Any, stem: str) -> str | None:
        target = stem.lower()
        indexed = getattr(self, "_image_path_by_stem_lower", None)
        if indexed:
            return indexed.get(target)
        stems = getattr(self, "_image_path_stem_lower", None)
        if stems:
            for path, path_stem in stems.items():
                if path_stem == target:
                    return path
            return None
        for path in self._workspace.image_paths:
            if Path(path).stem.lower() == target:
                return str(Path(path))
        return None

    def _paint_vector_list_item(self: Any, item: QListWidgetItem, stem: str) -> None:
        stem_lower = stem.lower()
        status = self._vector_status_enum_for_stem(stem_lower)
        paint_vector_row_item(item, stem, status, theme=getattr(self, "_ui_theme", "dark"))

    def _vector_status_enum_for_stem(self: Any, stem_lower: str):
        ipath = self._image_path_for_cif_stem(stem_lower)
        has_matching = ipath is not None
        cif_failed = stem_lower in self._cif_load_failure_stems
        normalized = ipath or ""
        never_opened = (not normalized) or (normalized not in self._viewed_image_paths)
        dirty = bool(
            ipath is not None
            and (
                self._workspace.image_has_changes(normalized)
                or normalized in self._restored_modified_image_paths
            )
        )
        persist = normalized in self._persisted_highlight_paths if normalized else False
        return classify_vector_side_status(
            has_matching_image=has_matching,
            cif_load_failed=cif_failed,
            image_never_viewed=never_opened,
            polygons_dirty=dirty,
            persist_highlight=persist,
        )

    def _rebuild_vector_list(self: Any) -> None:
        if not hasattr(self, "vector_list"):
            return
        self.vector_list.blockSignals(True)
        self.vector_list.clear()
        mapping = sorted(self._workspace.cif_paths_by_stem.items(), key=lambda kv: kv[0].lower())
        if len(mapping) > LARGE_FRAME_COUNT_THRESHOLD:
            count = len(mapping)
            summary = (
                f"Проиндексировано {count} векторов — используйте навигацию по кадрам."
                if self._ui_language == "ru"
                else f"{count} vectors indexed — use frame navigation."
            )
            item = QListWidgetItem(summary)
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.vector_list.addItem(item)
        else:
            for stem, cif_path in mapping:
                item = QListWidgetItem(Path(cif_path).stem)
                item.setToolTip(cif_path)
                item.setData(Qt.ItemDataRole.UserRole, cif_path)
                self._paint_vector_list_item(item, stem)
                self.vector_list.addItem(item)
        self.vector_list.blockSignals(False)
        if len(self._workspace.image_paths) <= ASSET_FILTER_LISTS_MAX_FRAMES:
            self._rebuild_asset_filter_lists()
            self._apply_asset_view_filter()

    def _vector_index_active(self: Any) -> bool:
        return bool(getattr(self._workspace, "_cif_paths_by_stem", {}))

    def _image_path_has_matching_vector(self: Any, image_path: str) -> bool:
        return bool(self._workspace.resolve_cif_path(image_path))

    def _make_asset_image_item(self: Any, image_path: str, *, show_text: bool = True) -> QListWidgetItem:
        item = QListWidgetItem(Path(image_path).stem)
        item.setToolTip(f"Путь к файлу: {image_path}" if self._ui_language == "ru" else f"File path: {image_path}")
        item.setData(Qt.ItemDataRole.UserRole, image_path)
        self._paint_image_row_item(item, image_path, show_text=show_text)
        return item

    def _make_asset_vector_item(self: Any, stem: str, vector_path: str) -> QListWidgetItem:
        item = QListWidgetItem(Path(vector_path).stem)
        item.setToolTip(vector_path)
        item.setData(Qt.ItemDataRole.UserRole, vector_path)
        self._paint_vector_list_item(item, stem)
        return item

    def _update_asset_items_for_image_path(self: Any, image_path: str) -> None:
        if not hasattr(self, "image_vector_list"):
            return
        normalized = str(Path(image_path))
        match_only = bool(getattr(self, "_asset_filter_match_only", False))
        has_vector = self._image_path_has_matching_vector(normalized)
        for list_widget in (self.image_vector_list, self.image_only_list):
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                if item is None:
                    continue
                if str(item.data(Qt.ItemDataRole.UserRole) or "") != normalized:
                    continue
                self._paint_image_row_item(item, normalized)
                item.setHidden(bool(match_only and not has_vector))
                return

    def _cancel_asset_filter_list_build(self: Any) -> None:
        self._asset_list_build_generation += 1
        if not hasattr(self, "image_vector_list"):
            return
        for list_widget in (self.image_vector_list, self.image_only_list, self.vector_only_list):
            list_widget.clear()

    def _refresh_asset_filter_list_item_states(self: Any) -> None:
        if not hasattr(self, "image_vector_list"):
            return
        if len(self._workspace.image_paths) > ASSET_FILTER_LISTS_MAX_FRAMES:
            if self.image_vector_list.count() or self.image_only_list.count() or self.vector_only_list.count():
                self._cancel_asset_filter_list_build()
            return
        match_only = bool(getattr(self, "_asset_filter_match_only", False))
        for list_widget in (self.image_vector_list, self.image_only_list):
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                if item is None:
                    continue
                image_path = str(item.data(Qt.ItemDataRole.UserRole) or "")
                if not image_path:
                    continue
                self._paint_image_row_item(item, image_path)
                item.setHidden(
                    bool(match_only and not self._image_path_has_matching_vector(image_path))
                )
        if not hasattr(self, "vector_only_list"):
            return
        for index in range(self.vector_only_list.count()):
            item = self.vector_only_list.item(index)
            if item is None:
                continue
            tip = item.toolTip()
            if not tip:
                continue
            self._paint_vector_list_item(item, Path(tip).stem.lower())
