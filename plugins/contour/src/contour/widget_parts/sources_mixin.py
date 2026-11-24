"""Loaded frame/vector sources: add, scan, remove and apply to the workspace.

The :class:`~contour.application.sources.SourceRegistry` is the source of
truth for what is loaded.  This mixin turns registry changes into the
existing workspace primitives (``_apply_image_paths_to_workspace`` and the
CIF index), runs folder scans one at a time through the existing scanners and
asks the user when a new source overlaps loaded frames or vectors.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QMessageBox

from contour.application.frame_asset_sync import VECTOR_FILE_SUFFIXES
from contour.application.sources import (
    SOURCE_IMAGES,
    SOURCE_VECTORS,
    FrameSource,
    SourceRegistry,
    files_source,
    folder_source,
    frame_stem,
    migrate_legacy_sources,
)
from contour.widget_parts._imports import is_image_path, is_visible_image_path

OVERLAP_REPLACE = "replace"
OVERLAP_KEEP = "keep"
OVERLAP_SKIP = "skip"
OVERLAP_CANCEL = "cancel"


class SourcesMixin:
    # ------------------------------------------------------------- registry
    @property
    def source_registry(self: Any) -> SourceRegistry:
        registry = self.__dict__.get("_sources")
        if registry is None:
            registry = SourceRegistry()
            self.__dict__["_sources"] = registry
        return registry

    def _scan_state(self: Any) -> dict[str, Any]:
        return self.__dict__.setdefault(
            "_source_scan_state",
            {"queue": {SOURCE_IMAGES: [], SOURCE_VECTORS: []}, "active": {SOURCE_IMAGES: None, SOURCE_VECTORS: None}},
        )

    def _vector_scan_pending(self: Any) -> bool:
        state = self._scan_state()
        return bool(state["active"][SOURCE_VECTORS] or state["queue"][SOURCE_VECTORS])

    def scanning_source_ids(self: Any) -> set[str]:
        state = self._scan_state()
        ids = {item for kind in state["queue"].values() for item in kind}
        ids |= {item for item in state["active"].values() if item}
        return ids

    # ---------------------------------------------------------- public add
    def add_image_folder(self: Any, path: str) -> None:
        self._add_source(folder_source(SOURCE_IMAGES, path))

    def add_image_files(self: Any, paths: list[str]) -> None:
        images = [str(Path(path)) for path in paths if is_visible_image_path(path)]
        if images:
            self._add_source(files_source(SOURCE_IMAGES, images))

    def add_vector_folder(self: Any, path: str) -> None:
        self._add_source(folder_source(SOURCE_VECTORS, path))

    def add_vector_files(self: Any, paths: list[str]) -> None:
        vectors = [str(Path(path)) for path in paths if Path(path).suffix.lower() in VECTOR_FILE_SUFFIXES]
        if vectors:
            self._add_source(files_source(SOURCE_VECTORS, vectors))

    def replace_sources(
        self: Any,
        kind: str,
        source: FrameSource | None,
        *,
        select_path: str | None = None,
    ) -> None:
        """Drop every source of ``kind`` and load ``source`` instead (CLI, Kraken)."""

        registry = self.source_registry
        if kind == SOURCE_IMAGES and self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self._cancel_source_scans(kind)
        for existing in registry.sources(kind):
            registry.remove(existing.id)
        if source is not None:
            registry.add(source)
            if not source.scanned:
                self._queue_source_scan(source)
        self._apply_sources(kind, select_path=select_path, force=True)

    def remove_source(self: Any, source_id: str) -> bool:
        registry = self.source_registry
        source = registry.get(source_id)
        if source is None or source.locked:
            return False
        affected = self._frames_affected_by_removal(source)
        if not self._confirm_discard_changes(affected):
            return False
        if source.kind == SOURCE_IMAGES:
            current = self._workspace.current_image_path
            if current and any(os.path.normcase(path) == os.path.normcase(str(current)) for path in affected):
                self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        self._cancel_source_scan(source)
        registry.remove(source_id)
        if self.__dict__.get("_source_filter_id") == source_id:
            self.set_source_filter(None)
        self._apply_sources(source.kind, force=True)
        return True

    def lock_sources_for_kraken(self: Any) -> None:
        registry = self.source_registry
        for source in registry.sources():
            registry.update(replace(source, locked=True))
        self._refresh_sources_panel()

    def _first_source_folder(self: Any, kind: str) -> str:
        return next((source.root for source in self.source_registry.sources(kind) if source.is_folder), "")

    def _sources_dialog_start_directory(self: Any, kind: str) -> str:
        """Start file dialogs in the most recently added source folder of ``kind``."""

        for source in reversed(self.source_registry.sources(kind)):
            if source.root and Path(source.root).is_dir():
                return source.root
        for source in reversed(self.source_registry.sources()):
            if source.root and Path(source.root).is_dir():
                return source.root
        return str(Path.home())

    def _ingest_dropped_source_paths(self: Any, paths: list[str]) -> None:
        """Folders dropped on the sources panel become sources; files are added as before."""

        files: list[str] = []
        for raw in paths:
            path = Path(raw)
            if not path.is_dir():
                files.append(str(path))
                continue
            entries = [entry for entry in path.iterdir() if entry.is_file()]
            if any(is_visible_image_path(str(entry)) for entry in entries):
                self.add_image_folder(str(path))
            if any(entry.suffix.lower() in VECTOR_FILE_SUFFIXES for entry in entries):
                self.add_vector_folder(str(path))
        if files:
            self._ingest_dropped_paths(files)

    # --------------------------------------------------------- source filter
    def set_source_filter(self: Any, source_id: str | None) -> None:
        """Show only the frames of one image source (or of frames a vector source covers)."""

        registry = self.source_registry
        source = registry.get(source_id) if source_id else None
        paths: set[str] = set()
        if source is not None and source.kind == SOURCE_IMAGES:
            paths = {os.path.normcase(path) for path in source.effective_paths()}
        elif source is not None:
            stems = set(source.stems())
            paths = {os.path.normcase(path) for path in registry.image_paths() if frame_stem(path) in stems}
        self.__dict__["_source_filter_id"] = source.id if source is not None else None
        self._source_filter_paths = paths
        proxy = getattr(self, "_image_list_proxy", None)
        if proxy is not None:
            proxy.invalidateFilter()

    # ------------------------------------------------------------ internals
    def _add_source(self: Any, source: FrameSource) -> None:
        if not source.scanned:
            self.source_registry.add(source)
            self._queue_source_scan(source)
            self._refresh_sources_panel()
            self._save_sources()
            return
        resolved, low_priority = self._resolve_source_overlap(source)
        if resolved is None:
            return
        if source.kind == SOURCE_IMAGES and self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self.source_registry.add(resolved, low_priority=low_priority)
        first_path = resolved.effective_paths()[0] if resolved.effective_paths() else None
        if self.__dict__.get("_suppress_source_selection"):
            first_path = None
        self._apply_sources(resolved.kind, select_path=first_path)

    def _resolve_source_overlap(self: Any, source: FrameSource) -> tuple[FrameSource | None, bool]:
        """Ask about frames/vectors already loaded; returns the source to add and its priority."""

        registry = self.source_registry
        if source.kind == SOURCE_IMAGES:
            duplicates = registry.image_duplicates(source)
            if not duplicates:
                return source, False
            if len(duplicates) == len(source.effective_paths()):
                self._append_log(self._tr("sources_all_frames_already_loaded_log"))
                return None, False
            if self._ask_source_overlap(SOURCE_IMAGES, len(duplicates)) != OVERLAP_SKIP:
                return None, False
            if source.is_folder:
                return replace(source, excluded=source.excluded | frozenset(duplicates)), False
            keep = [path for path in source.paths if path not in set(duplicates)]
            return replace(source, paths=tuple(keep)), False
        conflicts = registry.vector_conflicts(source)
        if not conflicts:
            return source, False
        answer = self._ask_source_overlap(SOURCE_VECTORS, len(conflicts))
        if answer == OVERLAP_CANCEL:
            return None, False
        return source, answer == OVERLAP_KEEP

    def _ask_source_overlap(self: Any, kind: str, count: int) -> str:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(self._tr("sources_overlap_title"))
        if kind == SOURCE_IMAGES:
            box.setText(self._tr("sources_frames_overlap_text", count=count))
            skip = box.addButton(self._tr("sources_skip_duplicates"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(self._tr("sources_cancel_add"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            return OVERLAP_SKIP if box.clickedButton() is skip else OVERLAP_CANCEL
        box.setText(self._tr("sources_vectors_overlap_text", count=count))
        newer = box.addButton(self._tr("sources_prefer_new"), QMessageBox.ButtonRole.AcceptRole)
        keep = box.addButton(self._tr("sources_keep_current"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(self._tr("sources_cancel_add"), QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is newer:
            return OVERLAP_REPLACE
        if clicked is keep:
            return OVERLAP_KEEP
        return OVERLAP_CANCEL

    def _frames_affected_by_removal(self: Any, source: FrameSource) -> list[str]:
        registry = self.source_registry
        if source.kind == SOURCE_IMAGES:
            return list(source.effective_paths())
        winning = {stem for stem, path in registry.cif_index().items() if source.contains(path)}
        return [path for path in registry.image_paths() if frame_stem(path) in winning]

    def _confirm_discard_changes(self: Any, paths: list[str]) -> bool:
        changed = [path for path in paths if self._workspace.image_has_changes(path)]
        if not changed:
            return True
        answer = QMessageBox.question(
            self,
            self._tr("sources_remove_title"),
            self._tr("sources_discard_changes_text", count=len(changed)),
        )
        return answer == QMessageBox.StandardButton.Yes

    def _apply_sources(self: Any, kind: str, *, select_path: str | None = None, force: bool = False) -> None:
        if kind == SOURCE_IMAGES:
            self._apply_image_sources(select_path=select_path, force=force)
        else:
            self._apply_vector_sources()
        self._save_sources()
        self._refresh_sources_panel()

    def _apply_image_sources(self: Any, *, select_path: str | None = None, force: bool = False) -> None:
        paths = [path for path in self.source_registry.image_paths() if is_image_path(path)]
        existing = [str(Path(path)) for path in self._workspace.image_paths]
        if paths == existing and not force:
            return
        current = self._workspace.current_image_path
        pending = self.__dict__.get("_pending_restore_current_image_path")
        candidates = [select_path, pending, str(Path(current)) if current else None]
        chosen = next((path for path in candidates if path and path in paths), None)
        if pending and pending in paths:
            self._pending_restore_current_image_path = None
        if chosen is None and current and str(Path(current)) in existing and paths:
            index = existing.index(str(Path(current)))
            chosen = paths[min(index, len(paths) - 1)]
        self._stop_work_simulation()
        self._apply_image_paths_to_workspace(
            paths,
            clear_extra_layers=not paths,
            select_path=chosen,
            fallback_to_first=chosen is None,
            clear_state_cache=False,
        )
        self._save_persisted_session_paths()
        if self.__dict__.get("_source_filter_id"):
            self.set_source_filter(self.__dict__.get("_source_filter_id"))

    def _apply_vector_sources(self: Any) -> None:
        if bool(getattr(self, "_image_list_rebuild_in_progress", False)):
            self.__dict__["_vector_sources_dirty"] = True
            return
        self.__dict__["_vector_sources_dirty"] = False
        old_index = dict(self._workspace.cif_paths_by_stem)
        new_index = self.source_registry.cif_index()
        changed = {stem for stem in set(old_index) | set(new_index) if old_index.get(stem) != new_index.get(stem)}
        self._workspace.set_cif_index(new_index)
        if changed:
            from contour.serializers import invalidate_cif_parse_cache

            invalidate_cif_parse_cache([old_index[stem] for stem in changed if stem in old_index])
            affected = [path for path in self._workspace.image_paths if frame_stem(path) in changed]
            self._workspace.forget_cleared_vectors(affected)
            self._workspace.invalidate_image_states(affected)
            neighbor_cache = getattr(self, "_neighbor_vector_cache", None)
            if neighbor_cache is not None:
                for path in affected:
                    neighbor_cache.pop(str(Path(path)), None)
        self._defer_vector_load_until_cif_index = False
        self._rebuild_vector_list()
        self._mark_thumbnail_grid_rebuild_pending()
        if changed:
            self._sync_after_cif_index_changed()
        else:
            self._refresh_image_list_item_states()

    def _apply_vector_sources_if_dirty(self: Any) -> bool:
        if not self.__dict__.get("_vector_sources_dirty"):
            return False
        self._apply_vector_sources()
        return True

    # ---------------------------------------------------------------- scans
    def _queue_source_scan(self: Any, source: FrameSource) -> None:
        state = self._scan_state()
        if source.id not in state["queue"][source.kind] and state["active"][source.kind] != source.id:
            state["queue"][source.kind].append(source.id)
        self._start_next_source_scan(source.kind)

    def _start_next_source_scan(self: Any, kind: str) -> None:
        state = self._scan_state()
        if state["active"][kind]:
            return
        queue = state["queue"][kind]
        while queue:
            source = self.source_registry.get(queue.pop(0))
            if source is None or source.scanned:
                continue
            state["active"][kind] = source.id
            self._refresh_sources_panel()
            if kind == SOURCE_IMAGES:
                self._directory_scanner.start(source.root)
            else:
                self._vector_indexer.start(source.root)
            return
        self._refresh_sources_panel()

    def _cancel_source_scan(self: Any, source: FrameSource) -> None:
        state = self._scan_state()
        if source.id in state["queue"][source.kind]:
            state["queue"][source.kind].remove(source.id)
        if state["active"][source.kind] == source.id:
            self._cancel_source_scans(source.kind, keep_queue=True)

    def _cancel_source_scans(self: Any, kind: str, *, keep_queue: bool = False) -> None:
        state = self._scan_state()
        if not keep_queue:
            state["queue"][kind].clear()
        if state["active"][kind]:
            state["active"][kind] = None
            scanner = self._directory_scanner if kind == SOURCE_IMAGES else self._vector_indexer
            scanner.invalidate_pending_results()
        self._start_next_source_scan(kind)

    def _on_source_scan_finished(self: Any, kind: str, paths: list[str] | None) -> bool:
        """Handle a scanner result for an active source scan; ``None`` means failure."""

        state = self._scan_state()
        source_id = state["active"][kind]
        if not source_id:
            return False
        state["active"][kind] = None
        registry = self.source_registry
        source = registry.get(source_id)
        if source is not None:
            found = [str(Path(path)) for path in paths or []]
            if kind == SOURCE_IMAGES:
                found = [path for path in found if is_image_path(path)]
            if paths is None:
                self._append_log(self._tr("sources_scan_failed_log", path=source.root))
            registry.remove(source_id)
            scanned = replace(source, paths=tuple(found), scanned=True)
            resolved, low_priority = self._resolve_source_overlap(scanned)
            if resolved is not None:
                registry.add(resolved, low_priority=low_priority)
            self._apply_sources(kind)
        self._start_next_source_scan(kind)
        return True

    # ----------------------------------------------------------- persistence
    def _save_sources(self: Any) -> None:
        store = getattr(self, "_session_settings_store", None)
        registry = self.source_registry
        if store is None or any(source.locked for source in registry.sources()):
            return
        store.save_sources(registry.to_payload())

    def _restore_sources_from_settings(self: Any) -> None:
        from ..infrastructure.settings_store import IMAGE_LIST_MODE_EXPLICIT

        store = self._session_settings_store
        payload = store.load_sources()
        if payload is None:
            legacy = self._path_settings.load()
            registry = migrate_legacy_sources(
                input_directory=legacy.input_directory,
                cif_directory=legacy.cif_directory,
                explicit_image_list=store.load_image_list_mode() == IMAGE_LIST_MODE_EXPLICIT,
                image_paths=store.load_image_paths(),
                vector_paths=store.load_vector_paths(),
            )
        else:
            registry = SourceRegistry.from_payload(payload)
        for source in registry.sources():
            if not source.is_folder:
                existing = tuple(path for path in source.paths if Path(path).is_file())
                if existing != source.paths:
                    registry.update(replace(source, paths=existing))
        self.__dict__["_sources"] = registry
        preferred = store.load_current_image_path()
        self._pending_restore_current_image_path = str(Path(preferred)) if preferred else None
        for source in registry.sources():
            if not source.scanned:
                self._queue_source_scan(source)
        self._apply_image_sources(force=True)
        self._apply_vector_sources()
        self._save_sources()
        self._refresh_sources_panel()

    def _refresh_sources_panel(self: Any) -> None:
        panel = getattr(self, "sources_panel", None)
        if panel is not None:
            panel.rebuild(self.source_registry, scanning=self.scanning_source_ids())
