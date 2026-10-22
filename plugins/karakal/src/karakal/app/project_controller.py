"""Projects: the computed cell check of a matrix tab saved as it is and opened again at once.

A tab gets its project when its cell check finishes: a new folder in the projects folder (beside
karakal.exe), or the folder the operator chose with «Save project as…». The results are written
again after every run of that tab (cells, dirt, memory map), the view (drop rules, sets, layer of
the matrix) also when the app closes. Opening a project puts back the layers, the settings and the
results of the run, with no folder scan, no cache lookup and no analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from ..core.project_store import (
    PROJECT_SUFFIX,
    RESULTS_FILE,
    VIEW_FILE,
    default_projects_dir,
    find_projects,
    is_project,
    project_folder,
    read_manifest,
    read_results,
    relative_path,
    relocate_state,
    resolve_folder,
    slim_results,
    write_manifest,
    write_results,
)
from ..ui.ui_constants import FOLDER_LABEL_ROLE

_LOGGER = logging.getLogger(__name__)

RECENT_LIMIT = 20
# Results of the run that a project keeps (fields of the matrix tab state).
RESULT_FIELDS = (
    "grid_inspection_results_ready",
    "grid_inspection_payload_by_key",
    "grid_inspection_payloads_by_layer",
    "grid_inspection_payloads_by_model",
    "grid_inspection_payloads_by_pair",
    "grid_inspection_config_payload",
    "grid_inspection_markup_signature",
    "grid_inspection_algorithm_version",
    "grid_inspection_calculation_record_keys",
    "grid_inspection_reference_record_keys",
    "grid_inspection_run_profiles",
    "memory_map",
    "memory_map_signature",
)
# How the run is looked at.
VIEW_FIELDS = (
    "grid_inspection_layer",
    "grid_inspection_model_id",
    "grid_inspection_matrix_selection",
    "excluded_record_keys",
    "frame_sets",
    "memory_map_roles",
    "gradient_name",
    "matrix_score_view_mode",
    "cell_size",
)
PATH_KEYS = ("path", "confidence_path", "original_path", "markup_path")


def _copy(value):
    """Containers copied two levels deep, so the save thread never iterates a dict the app changes."""

    if isinstance(value, dict):
        return {key: (dict(item) if isinstance(item, dict) else item) for key, item in value.items()}
    if isinstance(value, (set, list)):
        return type(value)(value)
    return value


class _SaveThread(QThread):
    saved = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, folder: Path, manifest: dict, results: dict | None, view: dict, parent=None) -> None:
        super().__init__(parent)
        self._folder = folder
        self._manifest = manifest
        self._results = results
        self._view = view

    def run(self) -> None:
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
            if self._results is not None:
                # Frame results keep only their defect cells: read back in seconds, not minutes.
                write_results(self._folder, slim_results(self._results))
            write_results(self._folder, self._view, VIEW_FILE)
            write_manifest(self._folder, self._manifest)
            self.saved.emit(str(self._folder))
        except Exception as error:  # shown to the operator
            _LOGGER.exception("Saving the project failed")
            self.failed.emit(str(error))


class ProjectController(QObject):
    recentChanged = pyqtSignal()
    statusChanged = pyqtSignal(str)
    # A project saved or opened: the home page makes its snapshot a recent selection.
    projectStored = pyqtSignal(str)

    def __init__(self, presenter) -> None:
        super().__init__(presenter)
        self._p = presenter
        self._settings = presenter._settings_service
        self._t = presenter._t
        self._folders: dict[object, Path] = {}
        self._thread: _SaveThread | None = None
        self._pending: tuple | None = None

    # Folders and the recent list --------------------------------------------------------------

    def projects_dir(self) -> Path:
        return default_projects_dir()

    def project_of(self, state) -> Path | None:
        return self._folders.get(getattr(state, "widget", None))

    def recent(self) -> list[Path]:
        return [Path(path) for path in self._settings.load_recent_projects()]

    def _remember(self, folder: Path) -> None:
        paths = [str(folder)] + [path for path in self._settings.load_recent_projects() if Path(path) != folder]
        self._settings.save_recent_projects(paths[:RECENT_LIMIT])
        self._settings.sync()
        self.recentChanged.emit()

    def forget(self, folder: Path | str) -> None:
        paths = [path for path in self._settings.load_recent_projects() if Path(path) != Path(folder)]
        self._settings.save_recent_projects(paths)
        self.recentChanged.emit()

    def entries(self) -> list[dict]:
        """Projects for the home page: the recent ones first, then the rest of the projects folder."""

        seen: set[str] = set()
        entries = []
        for folder in self.recent() + find_projects(self.projects_dir()):
            key = str(folder).lower()
            if key in seen:
                continue
            seen.add(key)
            entry = {"path": folder, "name": folder.name[: -len(PROJECT_SUFFIX)] if folder.suffix == PROJECT_SUFFIX else folder.name}
            try:
                manifest = read_manifest(folder)
            except (OSError, ValueError):
                entry["missing"] = True
                entries.append(entry)
                continue
            entry.update(
                saved_at=str(manifest.get("saved_at") or ""),
                frames=int(manifest.get("frames") or 0),
                layers=[str(layer.get("label") or "") for layer in manifest.get("layers") or ()],
                algorithm=str(manifest.get("algorithm_version") or ""),
                app_version=str(manifest.get("app_version") or ""),
            )
            entries.append(entry)
        return entries

    # Saving -------------------------------------------------------------------------------------

    def _default_name(self, state) -> str:
        labels = []
        p = self._p
        for row in range(p.folder_list.count()):
            label = str(p.folder_list.item(row).data(FOLDER_LABEL_ROLE) or "")
            if label:
                labels.append(label)
        stamp = datetime.now().strftime("%Y-%m-%d %H-%M")
        return f"{', '.join(labels[:3]) or 'Прогон'} {stamp}"

    def _manifest(self, state, folder: Path) -> dict:
        from ..core.grid_anomaly import GRID_DAMAGE_ALGORITHM_VERSION
        from ..version import APP_VERSION

        p = self._p
        folder_payload = p._build_folder_manager_payload()
        layers = []
        for row, entry in enumerate(folder_payload.get("folders") or ()):
            item = p.folder_list.item(row)
            layer = {"model_id": str(p._model_id_for_folder_item(item) or ""), "label": str(entry.get("label") or "")}
            for key in PATH_KEYS:
                value = str(entry.get(key) or "")
                layer[key] = value
                layer[f"{key}_relative"] = relative_path(value, folder)
            layers.append(layer)
        return {
            "name": folder.name[: -len(PROJECT_SUFFIX)],
            "app_version": APP_VERSION,
            "algorithm_version": str(getattr(state, "grid_inspection_algorithm_version", "") or GRID_DAMAGE_ALGORITHM_VERSION),
            "frames": len(getattr(state.build_result, "records", ()) or ()),
            "layers": layers,
            "folder_manager": folder_payload,
            "build_settings": p._build_build_settings_payload(),
        }

    def _results(self, state) -> dict:
        dirt = getattr(self._p, "_dirt_controller", None)
        return {
            "build_result": state.build_result,
            "tab": {name: _copy(getattr(state, name)) for name in RESULT_FIELDS},
            "dirt": dirt.export_state() if dirt is not None else {},
        }

    def _view(self, state) -> dict:
        dirt = getattr(self._p, "_dirt_controller", None)
        return {
            "tab": {name: _copy(getattr(state, name)) for name in VIEW_FIELDS},
            "dirt_settings": dirt.settings_payload() if dirt is not None else {},
        }

    def save(self, state=None, *, folder: Path | None = None, results: bool = True, wait: bool = False) -> Path | None:
        """Write the tab's project (its own folder, a new one in the projects folder, or ``folder``)."""

        state = state or self._p._current_tab_state()
        if state is None or not getattr(state.build_result, "records", None):
            return None
        target = folder or self.project_of(state)
        if target is None:
            target = project_folder(self.projects_dir(), self._default_name(state))
        self._folders[state.widget] = target
        if results is False and not (target / RESULTS_FILE).is_file():
            results = True
        job = (target, self._manifest(state, target), self._results(state) if results else None, self._view(state))
        if self._thread is not None:
            # One save at a time; a newer one replaces a waiting one (it carries everything).
            if self._pending is not None and self._pending[2] is not None and job[2] is None:
                job = (job[0], job[1], self._pending[2], job[3])
            self._pending = job
            if wait:
                # The running save first, then this one: both are on disk when save() returns.
                running = self._thread
                running.wait()
                self._thread_done(running)
                started = self._thread
                if started is not None:
                    started.wait()
                    self._thread_done(started)
            return target
        self._start(job, wait=wait)
        return target

    def _start(self, job: tuple, *, wait: bool) -> None:
        folder, manifest, results, view = job
        self.statusChanged.emit(self._t("project.saving", name=folder.name[: -len(PROJECT_SUFFIX)]))
        thread = _SaveThread(folder, manifest, results, view, self)
        thread.saved.connect(self._saved)
        thread.failed.connect(lambda message: self.statusChanged.emit(self._t("project.save_failed", error=message)))
        thread.finished.connect(lambda done=thread: self._thread_done(done))
        self._thread = thread
        thread.start()
        if wait:
            thread.wait()
            # Its «saved» signal reaches the recent list now, not after save() returned.
            QApplication.processEvents()
            self._thread_done(thread)

    def _saved(self, folder: str) -> None:
        path = Path(folder)
        self._remember(path)
        self.projectStored.emit(str(path))
        self.statusChanged.emit(
            self._t("project.saved", name=path.name[: -len(PROJECT_SUFFIX)], time=datetime.now().strftime("%H:%M"))
        )

    def _thread_done(self, thread: _SaveThread | None = None) -> None:
        """A save finished: the next one waiting starts. Called once per save (a save waited for
        is finished here at once; its own «finished» signal then finds it done)."""

        if thread is None or thread is not self._thread:
            return
        self._thread = None
        thread.deleteLater()
        pending, self._pending = self._pending, None
        if pending is not None:
            self._start(pending, wait=False)

    def results_changed(self) -> None:
        """A run of the shown tab finished (cells, dirt, memory map): its project follows."""

        state = self._p._current_tab_state()
        if state is not None and getattr(state, "grid_inspection_results_ready", False):
            self.save(state, results=True)

    def save_view(self, *, wait: bool = False) -> None:
        """Rules, sets and the layer of the matrix of every tab that has a project."""

        for state in list(getattr(self._p, "_tab_states", {}).values()):
            if self.project_of(state) is not None:
                self.save(state, results=False, wait=wait)

    def save_as(self) -> Path | None:
        state = self._p._current_tab_state()
        if state is None or not getattr(state.build_result, "records", None):
            QMessageBox.information(self._p._view, self._t("project.title"), self._t("project.nothing_to_save"))
            return None
        parent = QFileDialog.getExistingDirectory(self._p._view, self._t("project.save_as"), str(self.projects_dir()))
        if not parent:
            return None
        return self.save(state, folder=project_folder(parent, self._default_name(state)), wait=True)

    def save_current(self) -> Path | None:
        state = self._p._current_tab_state()
        if state is None or not getattr(state.build_result, "records", None):
            QMessageBox.information(self._p._view, self._t("project.title"), self._t("project.nothing_to_save"))
            return None
        return self.save(state, wait=True)

    def shutdown(self) -> None:
        try:
            self.save_view(wait=True)
        except Exception:  # closing must not fail on a save
            _LOGGER.exception("Saving project views on close failed")
        # Every save still running or waiting is on disk before the app goes.
        while self._thread is not None:
            running = self._thread
            running.wait()
            self._thread_done(running)

    def open_projects_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.projects_dir())))

    # Opening ------------------------------------------------------------------------------------

    def open_dialog(self) -> bool:
        chosen = QFileDialog.getExistingDirectory(self._p._view, self._t("project.open"), str(self.projects_dir()))
        return bool(chosen) and self.open(Path(chosen))

    def open(self, folder: Path | str) -> bool:
        folder = Path(folder)
        p = self._p
        if p._worker_thread is not None:
            QMessageBox.information(p._view, self._t("project.title"), self._t("project.busy"))
            return False
        if not is_project(folder):
            QMessageBox.warning(p._view, self._t("project.title"), self._t("project.not_a_project", path=str(folder)))
            return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            try:
                manifest = read_manifest(folder)
                results = read_results(folder)
                view = read_results(folder, VIEW_FILE)
            except Exception as error:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(p._view, self._t("project.title"), self._t("project.open_failed", error=str(error)))
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                return False
            self._apply(folder, manifest, results, view)
        finally:
            QApplication.restoreOverrideCursor()
        self._remember(folder)
        self.projectStored.emit(str(folder))
        self.statusChanged.emit(self._t("project.opened", name=folder.name[: -len(PROJECT_SUFFIX)]))
        return True

    def _apply(self, folder: Path, manifest: dict, results: dict, view: dict) -> None:
        p = self._p
        # Folders: as they were, else beside the project as they were relative to it.
        path_moves: list[tuple[str, str]] = []
        for layer in manifest.get("layers") or ():
            for key in PATH_KEYS:
                stored = str(layer.get(key) or "")
                found = resolve_folder(stored, str(layer.get(f"{key}_relative") or ""), folder)
                if stored and found != stored:
                    path_moves.append((stored, found))
        folder_payload = relocate_state(dict(manifest.get("folder_manager") or {}), path_moves, {})
        p._clear_folders()
        p._apply_folder_manager_payload(folder_payload, keep_missing=True)
        # Layer ids follow the folders: rename them everywhere when the folders moved.
        id_moves = {}
        for row, layer in enumerate(manifest.get("layers") or ()):
            item = p.folder_list.item(row)
            if item is None:
                continue
            old_id = str(layer.get("model_id") or "")
            new_id = str(p._model_id_for_folder_item(item) or "")
            if old_id and new_id and old_id != new_id:
                id_moves[old_id] = new_id
        results = relocate_state(results, path_moves, id_moves)
        view = relocate_state(view, path_moves, id_moves)
        build_settings = dict(manifest.get("build_settings") or {})
        if build_settings:
            p._apply_build_settings_payload(build_settings)
        dirt = getattr(p, "_dirt_controller", None)
        if dirt is not None and view.get("dirt_settings"):
            dirt.apply_settings(view["dirt_settings"])
        combo = getattr(p, "app_mode_combo", None)
        if combo is not None:
            index = combo.findData("grid_inspection")
            if index >= 0 and combo.currentIndex() != index:
                combo.setCurrentIndex(index)
        p._refresh_folder_rows()
        p._refresh_pair_matrix()
        p._on_build_finished(results["build_result"])
        state = p._current_tab_state()
        if state is None:
            return
        for name, value in dict(results.get("tab") or {}).items():
            if name in RESULT_FIELDS:
                setattr(state, name, value)
        for name, value in dict(view.get("tab") or {}).items():
            if name in VIEW_FIELDS:
                setattr(state, name, value)
        self._folders[state.widget] = folder
        if not getattr(state, "grid_inspection_markup_signature", ()):
            # A project made by «Compute and export» has no markup: the layers' signature as they are.
            state.grid_inspection_markup_signature = p._grid_markup_signature()
        p._refresh_grid_inspection_mode_view()
        if dirt is not None:
            dirt.import_state(dict(results.get("dirt") or {}))
        memory_map = getattr(p, "_memory_map_controller", None)
        if memory_map is not None:
            try:
                memory_map.refresh()
            except Exception:
                _LOGGER.exception("Showing the memory map of the project failed")
        frame_sets = getattr(p, "_frame_sets_controller", None)
        if frame_sets is not None:
            frame_sets.refresh()
        p._sync_action_buttons()
