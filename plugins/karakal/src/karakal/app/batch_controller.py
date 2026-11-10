"""Compute and export from the home page: a snapshot checked and its error maps written, with no
matrix; the matrix of it opened from the same page.

The snapshot folder is read for its layers (core.snapshot_layout); the frames are indexed, every
layer is checked by the same worker the matrix uses (the run's bank of normal cells, the conflicts
of a ones/zeros pair) with the app's settings, and the error map of every frame is written in the
chosen format. The run is saved as a project too: the matrix button opens it without recomputing.
A run stopped part way leaves its frames in the cell cache: the matrix computes only the rest.

Every snapshot chosen is a recent selection (core.recent_selections) with its format, its export
folder and what its last run left; a project saved from the matrix makes its snapshot one too.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..core import recent_selections as selections
from ..core.batch_progress import RUNNING, BatchProgress
from ..core.domain import BuildOptions, ModelSpec
from ..core.frame_set_export import (
    ITEM_ERRORS,
    FrameSetExportFrame,
    FrameSetExportModel,
    FrameSetExportPlan,
    export_frame_set,
)
from ..core.grid_anomaly import GRID_DAMAGE_ALGORITHM_VERSION, RunProfileRequest
from ..core.grid_narrowing import GridNarrowing, narrow_grid_frame_result
from ..core.pair_conflict import PAIR_CONFLICT_LAYER, filter_class_conflicts, pair_key
from ..core.project_store import (
    VIEW_FILE,
    default_projects_dir,
    is_project,
    project_folder,
    read_manifest,
    relative_path,
    slim_results,
    stable_model_id,
    write_manifest,
    write_results,
)
from ..core.recent_selections import Selection
from ..core.snapshot_layout import SnapshotLayout, discover_snapshot

_LOGGER = logging.getLogger(__name__)

ERROR_FORMATS = ("png", "jpg", "bmp", "tif")
SET_NAME = "Весь прогон"


@dataclass(slots=True)
class BatchJob:
    layout: SnapshotLayout
    output_dir: Path
    errors_format: str
    specs: tuple[ModelSpec, ...]
    cell_classes: dict[str, str]
    options: BuildOptions
    config: object
    requested_layers: tuple[str, ...]
    error_types: tuple[str, ...]
    debris_min_area_px: int
    conflict_thresholds: object
    with_conflicts: bool
    defect_labels: dict[str, str]
    folder_payload: dict
    build_settings: dict
    config_payload: dict
    performance_config: object = None
    # Frames per row of the snapshot (0: frames have no neighbours) and the sparse cell distance.
    frames_per_row: int = 0
    sparse_reach_sites: float = 4.0


@dataclass(slots=True)
class BatchReport:
    output_dir: Path | None = None
    project: Path | None = None
    frames: int = 0
    written: int = 0
    missing: tuple[str, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)
    # Frames that could not be checked or written: (frame, layer, why).
    problems: tuple[tuple[str, str, str], ...] = ()
    # Defect cells of every layer, as the error maps show them.
    defects: int = 0


class BatchCancelled(Exception):
    pass


class _BatchThread(QThread):
    """The run in its own thread: every stage reported as it goes, pausable, stoppable."""

    progressed = pyqtSignal(object)
    problems = pyqtSignal(object)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, job: BatchJob, titles, problem_text, parent=None) -> None:
        super().__init__(parent)
        self._job = job
        self._titles = titles
        self._problem_text = problem_text
        self.cancel = False
        self._worker = None
        self._resume = threading.Event()
        self._resume.set()
        self._progress: BatchProgress | None = None
        self._last_emit = 0.0
        self._problems: list[tuple[str, str, str]] = []

    # Control ------------------------------------------------------------------------------------

    def request_cancel(self) -> None:
        self.cancel = True
        self._resume.set()
        worker = self._worker
        if worker is not None:
            worker.request_cancel()

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def set_paused(self, paused: bool) -> None:
        worker = self._worker
        progress = self._progress
        if paused:
            self._resume.clear()
            if worker is not None:
                worker.request_pause()
            if progress is not None:
                progress.pause()
        else:
            if progress is not None:
                progress.resume()
            self._resume.set()
            if worker is not None:
                worker.request_resume()
        self._emit(force=True)

    def _checkpoint(self) -> bool:
        """Waits here while paused; True when the run is stopped."""

        while not self._resume.wait(0.1):
            if self.cancel:
                break
        return self.cancel

    def _check(self) -> None:
        if self._checkpoint():
            raise BatchCancelled()

    # Reporting ----------------------------------------------------------------------------------

    def _emit(self, *, force: bool = False) -> None:
        progress = self._progress
        if progress is None:
            return
        now = time.monotonic()
        if force or now - self._last_emit >= 0.25:
            self._last_emit = now
            self.progressed.emit(progress.view())

    def _problem(self, frame: str, layer: str, reason: str) -> None:
        self._problems.append((str(frame), str(layer), str(reason)))
        self.problems.emit(list(self._problems))

    # The run ------------------------------------------------------------------------------------

    def run(self) -> None:
        try:
            self.done.emit(self._run())
        except BatchCancelled:
            self.stopped.emit()
        except Exception as error:  # shown to the operator
            _LOGGER.exception("Compute and export failed")
            self.failed.emit(str(error))

    def _run(self) -> BatchReport:
        from ..core.analytics import collect_frame_records
        from ..core.workers import MultiModelGridInspectionWorker

        job = self._job
        names = {spec.model_id: spec.display_name for spec in job.specs}
        ones = next((spec.model_id for spec in job.specs if job.cell_classes.get(spec.model_id) == "ones"), "")
        zeros = next((spec.model_id for spec in job.specs if job.cell_classes.get(spec.model_id) == "zeros"), "")
        pairs = ((pair_key(ones, zeros), ones, zeros),) if ones and zeros else ()
        counts = {spec.model_id: layer.mask_count for spec, layer in zip(job.specs, job.layout.layers)}
        progress = BatchProgress.plan(
            [(spec.model_id, spec.display_name, counts.get(spec.model_id, 0)) for spec in job.specs],
            [(key, f"{names[a]} ↔ {names[b]}", min(counts.get(a, 0), counts.get(b, 0))) for key, a, b in pairs],
            titles=self._titles,
        )
        self._progress = progress
        # 1. Frames of the snapshot, matched across the layers.
        progress.start("index")
        self._emit(force=True)

        def index_progress(done, total, key) -> None:
            progress.update("index", int(done), int(total), Path(str(key).split(":", 1)[-1]).name)
            self._emit()

        build = collect_frame_records(
            job.specs,
            job.options,
            cancel_check=self._checkpoint,
            progress_callback=index_progress,
        )
        self._check()
        records = tuple(build.records)
        progress.finish("index")
        model_ids = tuple(spec.model_id for spec in job.specs)
        frames_of = {model_id: sum(1 for record in records if record.model_mask_paths.get(model_id)) for model_id in model_ids}
        for model_id in model_ids:
            stage = progress.stage(f"cells:{model_id}")
            if stage is not None:
                stage.total = frames_of[model_id]
        for key, a, b in pairs:
            stage = progress.stage(f"pair:{key}")
            if stage is not None:
                stage.total = sum(1 for record in records if record.model_mask_paths.get(a) and record.model_mask_paths.get(b))
        # A pair's frame with no mask of one network is a frame the check misses.
        for key, a, b in pairs:
            for record in records:
                for model_id in (a, b):
                    if not record.model_mask_paths.get(model_id):
                        self._problem(record.display_name, names[model_id], self._problem_text("no_mask"))
        self._emit(force=True)
        # 2. Cells of every layer, then the conflicts of the pair: the matrix's own worker.
        requests = {}
        for model_id in model_ids:
            paths = tuple(str(record.model_mask_paths.get(model_id)) for record in records if record.model_mask_paths.get(model_id))
            if paths:
                requests[model_id] = RunProfileRequest(paths, sample_limit=32, frame_id=f"run:{model_id}", min_frames=1)
        worker = MultiModelGridInspectionWorker(
            records,
            model_ids,
            job.config,
            reference_profiles=requests,
            performance_config=job.performance_config,
            requested_layers=job.requested_layers,
            conflict_pairs=pairs,
            frames_per_row=int(job.frames_per_row),
            sparse_reach_sites=float(job.sparse_reach_sites),
        )
        if self.paused:
            worker.request_pause()
        self._worker = worker
        # The worker counts every (layer, frame) and then (pair, frame) item in one run.
        offsets: dict[str, int] = {}
        running_total = 0
        for model_id in model_ids:
            offsets[model_id] = running_total
            running_total += frames_of[model_id]
        for key, _a, _b in pairs:
            offsets[key] = running_total
            stage = progress.stage(f"pair:{key}")
            running_total += stage.total if stage is not None else 0
        current = {"key": ""}

        def started(key: str, _step: int, _total: int) -> None:
            current["key"] = str(key)
            progress.start(f"pair:{key}" if str(key) in offsets and str(key) not in frames_of else f"bank:{key}")
            self._emit(force=True)

        def worker_progress(done: int, _total: int, key: str) -> None:
            stage_key = current["key"]
            if not stage_key:
                return
            local = int(done) - offsets.get(stage_key, 0)
            if local <= 0:
                return
            name = Path(str(key)).name if key else ""
            if stage_key in frames_of:
                progress.update(f"cells:{stage_key}", local, frames_of[stage_key], name)
            else:
                progress.update(f"pair:{stage_key}", local, None, name)
            self._emit()

        def errors(found: dict) -> None:
            layer = names.get(current["key"], current["key"])
            for frame, message in dict(found or {}).items():
                self._problem(Path(str(frame)).name, layer, str(message))

        payloads: dict = {}
        profiles: dict = {}
        outcome = {"failed": "", "cancelled": False}
        worker.modelStarted.connect(started)
        worker.progress.connect(worker_progress)
        worker.analysisErrors.connect(errors)
        worker.finished.connect(lambda result: payloads.update(dict(result or {})))
        worker.referenceProfileReady.connect(lambda model_id, profile: profiles.__setitem__(str(model_id), profile))
        worker.failed.connect(lambda message: outcome.__setitem__("failed", str(message)))
        worker.cancelled.connect(lambda: outcome.__setitem__("cancelled", True))
        worker.run()
        self._worker = None
        if outcome["cancelled"] or self.cancel:
            raise BatchCancelled()
        if outcome["failed"]:
            raise RuntimeError(outcome["failed"])
        for model_id in model_ids:
            progress.finish(f"bank:{model_id}")
            progress.finish(f"cells:{model_id}")
        for key, _a, _b in pairs:
            progress.finish(f"pair:{key}")
        by_model, by_pair = split_payloads(payloads)
        # 3. Error maps of every frame and layer, as the matrix shows them.
        progress.start("export")
        self._emit(force=True)
        layer_results = {
            model_id: export_results(by_model.get(model_id, {}), by_pair, job, model_id, pairs) for model_id in model_ids
        }
        defects = sum(
            1
            for results in layer_results.values()
            for result in results.values()
            for cell in getattr(result, "per_cell_results", ()) or ()
            if str(getattr(cell, "status", "")) != "normal"
        )
        frames = tuple(
            FrameSetExportFrame(
                key=str(record.key),
                name=Path(str(record.display_name or record.key)).name,
                original_path=str(record.original_path or ""),
                mask_paths={model_id: str(record.model_mask_paths.get(model_id) or "") for model_id in model_ids},
                confidence_paths={model_id: str(record.model_prob_paths.get(model_id) or "") for model_id in model_ids},
                results={model_id: layer_results[model_id].get(str(record.key)) for model_id in model_ids},
            )
            for record in records
        )
        plan = FrameSetExportPlan(
            set_name=SET_NAME,
            layer_title=", ".join(spec.display_name for spec in job.specs),
            frames=frames,
            models=tuple(FrameSetExportModel(spec.model_id, spec.display_name) for spec in job.specs),
            items=frozenset({ITEM_ERRORS}),
            error_types=job.error_types,
            defect_labels=job.defect_labels,
            errors_format=job.errors_format,
        )
        self._check()
        written_per_frame = max(1, len(model_ids))

        def export_progress(done: int, total: int, name: str) -> None:
            progress.update("export", int(done) * written_per_frame, int(total) * written_per_frame, str(name))
            self._emit()

        export = export_frame_set(plan, job.output_dir, progress=export_progress, cancelled=self._checkpoint)
        if export.cancelled:
            raise BatchCancelled()
        known = {frame for frame, _layer, _why in self._problems}
        for name in export.missing:
            # A frame already told (e.g. no mask of one network) is not told again.
            if name not in known:
                self._problem(name, "", self._problem_text("no_result"))
        progress.finish("export")
        # 4. The run as a project, to open in the matrix without recomputing.
        progress.start("project")
        self._emit(force=True)
        project = write_batch_project(job, build, by_model, by_pair, profiles)
        progress.finish_all()
        self._emit(force=True)
        return BatchReport(
            output_dir=export.run_dir,
            project=project,
            frames=len(records),
            written=export.written,
            missing=tuple(export.missing),
            problems=tuple(self._problems),
            defects=defects,
        )


def split_payloads(payloads: dict) -> tuple[dict, dict]:
    """The worker's results by layer (model -> layer -> frame -> result) and by pair (raw conflicts)."""

    by_model: dict[str, dict[str, dict[str, object]]] = {}
    by_pair: dict[str, dict[str, object]] = {}
    for key, frame_map in dict(payloads).items():
        if not isinstance(frame_map, dict):
            continue
        if str(key).startswith("pair:"):
            by_pair[str(key)] = {
                str(frame): layers.get(PAIR_CONFLICT_LAYER)
                for frame, layers in frame_map.items()
                if isinstance(layers, dict) and layers.get(PAIR_CONFLICT_LAYER) is not None
            }
            continue
        layers: dict[str, dict[str, object]] = {}
        for frame, layer_payload in frame_map.items():
            if not isinstance(layer_payload, dict):
                continue
            for layer_key, result in layer_payload.items():
                if result is not None:
                    layers.setdefault(str(layer_key), {})[str(frame)] = result
        by_model[str(key)] = layers
    return by_model, by_pair


def export_results(layers: dict, by_pair: dict, job: BatchJob, model_id: str, pairs) -> dict[str, object]:
    """Frame results of a layer the way the matrix shows them: the binary layer, debris from the
    operator's size, with the conflicts of its pair when class conflicts are on."""

    from .presenter import KarakalPresenter

    results = dict(layers.get("binary") or layers.get("confidence") or {})
    narrowing = GridNarrowing(debris_min_area_px=int(job.debris_min_area_px))
    results = {key: narrow_grid_frame_result(result, narrowing) for key, result in results.items()}
    if job.with_conflicts:
        for key, _ones, _zeros in pairs:
            if model_id not in (_ones, _zeros):
                continue
            for frame, raw in (by_pair.get(key) or {}).items():
                if frame in results:
                    results[frame] = KarakalPresenter._with_conflict_cells(
                        results[frame], filter_class_conflicts(raw, job.conflict_thresholds)
                    )
    return results


def write_batch_project(job: BatchJob, build, by_model: dict, by_pair: dict, profiles: dict) -> Path:
    """A project of the run, as the matrix saves one after its cell check."""

    first = job.specs[0].model_id
    folder = project_folder(default_projects_dir(), f"{job.layout.root.name} {datetime.now().strftime('%Y-%m-%d %H-%M')}")
    from ..version import APP_VERSION

    layers = []
    for entry, spec in zip(job.folder_payload.get("folders") or (), job.specs):
        layer = {"model_id": spec.model_id, "label": spec.display_name}
        for key in ("path", "confidence_path", "original_path", "markup_path"):
            value = str(entry.get(key) or "")
            layer[key] = value
            layer[f"{key}_relative"] = relative_path(value, folder)
        layers.append(layer)
    shown = dict(by_model.get(first) or {})
    results = {
        "build_result": build,
        "tab": {
            "grid_inspection_results_ready": True,
            "grid_inspection_payload_by_key": dict(shown.get("binary") or shown.get("confidence") or {}),
            "grid_inspection_payloads_by_layer": shown,
            "grid_inspection_payloads_by_model": by_model,
            "grid_inspection_payloads_by_pair": by_pair,
            "grid_inspection_config_payload": dict(job.config_payload),
            "grid_inspection_algorithm_version": GRID_DAMAGE_ALGORITHM_VERSION,
            "grid_inspection_run_profiles": dict(profiles),
        },
        "dirt": {},
    }
    view = {
        "tab": {
            "grid_inspection_matrix_selection": first,
            "grid_inspection_model_id": first,
            "grid_inspection_layer": "binary",
        },
        "dirt_settings": {},
    }
    folder.mkdir(parents=True, exist_ok=True)
    write_results(folder, slim_results(results))
    write_results(folder, view, VIEW_FILE)
    write_manifest(
        folder,
        {
            "name": folder.name[: -len(".karakal")],
            "app_version": APP_VERSION,
            "algorithm_version": GRID_DAMAGE_ALGORITHM_VERSION,
            "frames": len(build.records),
            "layers": layers,
            "folder_manager": job.folder_payload,
            "build_settings": job.build_settings,
        },
    )
    return folder


class BatchController(QObject):
    """The home page: snapshot, layers found, format, run, the matrix button, recent selections."""

    finished = pyqtSignal(object)

    def __init__(self, presenter, page) -> None:
        super().__init__(presenter)
        self._p = presenter
        self._page = page
        self._settings = presenter._settings_service
        self._layout: SnapshotLayout | None = None
        self._thread: _BatchThread | None = None
        self._report: BatchReport | None = None
        self._last_view = None
        self._run_snapshot = ""
        payload = self._settings.load_recent_selections()
        self._selections: list[Selection] = selections.load(payload)
        self._pace = float(payload.get("seconds_per_frame") or 0.0) if isinstance(payload, dict) else 0.0
        page.snapshotBrowseRequested.connect(self.browse_snapshot)
        page.snapshotChosen.connect(self.choose_snapshot)
        page.outputBrowseRequested.connect(self.browse_output)
        page.formatChanged.connect(lambda _value: self._keep_inputs())
        page.output_edit.editingFinished.connect(self._keep_inputs)
        page.runRequested.connect(self.run)
        page.cancelRequested.connect(self.cancel)
        page.pauseRequested.connect(self.toggle_pause)
        page.openFolderRequested.connect(self.open_output)
        page.matrixRequested.connect(self.open_matrix)
        page.selectionPicked.connect(self.pick)
        page.forgetRequested.connect(self.forget)
        project = self._projects()
        if project is not None:
            project.projectStored.connect(self._project_stored)
            if not self._selections:
                # The projects of earlier versions become selections once.
                for entry in reversed(project.entries()[: selections.RECENT_LIMIT]):
                    if not entry.get("missing"):
                        self._project_stored(str(entry["path"]), save=False)
                if self._selections:
                    self._save()
        self._show()

    def _projects(self):
        return getattr(self._p, "_project_controller", None)

    def is_running(self) -> bool:
        return self._thread is not None

    # Selections ---------------------------------------------------------------------------------

    def selections(self) -> list[Selection]:
        return list(self._selections)

    def current(self) -> Selection | None:
        return selections.find(self._selections, self._layout.root) if self._layout is not None else None

    def _save(self) -> None:
        self._settings.save_recent_selections(selections.dump(self._selections, self._pace))
        self._settings.sync()

    def _put(self, selection: Selection) -> None:
        self._selections = selections.remember(self._selections, selection)
        self._save()

    def _keep_inputs(self) -> None:
        """The format and the export folder are the chosen selection's."""

        selection = self.current()
        if selection is None or self._thread is not None:
            return
        selection.errors_format = self._page.errors_format()
        selection.output_dir = self._page.output_dir()
        self._save()
        self._page.set_selections(self._selections, selection.snapshot)

    def _show(self) -> None:
        page = self._page
        page.set_selections(self._selections, str(self._layout.root) if self._layout is not None else "")
        selection = self.current()
        project = Path(selection.project) if selection is not None and selection.project else None
        state = selections.matrix_state(
            selection if self._layout is not None else None,
            running=self._thread is not None,
            project_exists=project is not None and is_project(project),
        )
        if self._layout is not None and self._layout.layers and state == selections.MATRIX_EMPTY:
            state = selections.MATRIX_NONE
        per_row = 0
        spin = getattr(self._p._view, "frames_per_row_spin", None)
        if spin is not None:
            per_row = int(spin.value())
        page.set_matrix(state, selection, self._layout, self._pace, per_row)

    def _project_stored(self, folder: str, *, save: bool = True) -> None:
        """A project saved or opened (in the matrix too): its snapshot is a done selection."""

        try:
            manifest = read_manifest(folder)
        except (OSError, ValueError):
            return
        root = selections.snapshot_root([layer.get("path") or "" for layer in manifest.get("layers") or ()])
        if root is None:
            return
        selection = selections.find(self._selections, root) or Selection(snapshot=str(root))
        if not selections.same_folder(selection.project, folder):
            # Another project: what the earlier one counted is not this one's.
            selection.written = 0
            selection.problems = 0
            selection.defects = -1
            selection.seconds = 0.0
        selection.project = str(folder)
        selection.run = selections.RUN_DONE
        selection.share = 1.0
        selection.stage = ""
        selection.frames = int(manifest.get("frames") or 0)
        selection.at = str(manifest.get("saved_at") or selections.now_text())
        self._selections = selections.remember(self._selections, selection)
        if save:
            self._save()
            self._show()

    def pick(self, snapshot: str) -> None:
        """A recent selection: its snapshot, format and export folder back on the page."""

        selection = selections.find(self._selections, snapshot)
        if selection is None or self._thread is not None:
            return
        self.choose_snapshot(selection.snapshot)

    def forget(self, snapshot: str) -> None:
        self._selections = selections.forget(self._selections, snapshot)
        self._save()
        self._show()

    # Inputs -------------------------------------------------------------------------------------

    def browse_snapshot(self) -> None:
        start = str(self._layout.root) if self._layout is not None else ""
        chosen = QFileDialog.getExistingDirectory(self._page, self._p._t("batch.choose_snapshot"), start)
        if chosen:
            self.choose_snapshot(chosen)

    def browse_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self._page, self._p._t("batch.choose_output"), self._page.output_dir())
        if chosen:
            self._page.set_output_dir(chosen)
            self._keep_inputs()

    def choose_snapshot(self, path: str) -> SnapshotLayout | None:
        if self._thread is not None:
            return self._layout
        layout = discover_snapshot(path)
        self._layout = layout
        self._report = None
        page = self._page
        page.set_layout(layout)
        if layout.layers:
            selection = selections.find(self._selections, layout.root)
            if selection is None:
                previous = self._selections[0] if self._selections else None
                selection = Selection(
                    snapshot=str(layout.root),
                    errors_format=previous.errors_format if previous is not None else page.errors_format(),
                    output_dir=str(layout.root),
                )
            if not selection.output_dir:
                selection.output_dir = str(layout.root)
            page.set_errors_format(selection.errors_format)
            page.set_output_dir(selection.output_dir)
            self._put(selection)
        self._show()
        return layout

    def _job(self) -> BatchJob | None:
        layout = self._layout
        if layout is None or not layout.layers:
            return None
        p = self._p
        specs = tuple(
            ModelSpec(
                model_id=stable_model_id(layer.mask_folder, layer.confidence_folder),
                display_name=layer.label,
                mask_folder=layer.mask_folder,
                prob_folder=layer.confidence_folder,
                original_folder=layer.source_folder,
            )
            for layer in layout.layers
        )
        naming = p._frame_naming()
        options = BuildOptions(
            recursive=False,
            mask_suffix=naming.mask_suffix,
            confidence_suffix=naming.confidence_suffix,
            union_layer_frames=True,
        )
        config_payload = p._grid_inspection_config_payload()
        from ..ui.ui_constants import GRID_INSPECTION_ERROR_TYPE_OPTIONS

        skip = {"conductor_zone", "normal"}
        return BatchJob(
            layout=layout,
            output_dir=Path(self._page.output_dir() or layout.root),
            errors_format=self._page.errors_format(),
            specs=specs,
            cell_classes={spec.model_id: layer.cell_class for spec, layer in zip(specs, layout.layers)},
            options=options,
            config=p._grid_damage_config_from_payload(config_payload),
            requested_layers=tuple(p._selected_grid_compute_layers()) or ("binary",),
            error_types=tuple(value for value in p._selected_grid_error_types() if value not in skip),
            debris_min_area_px=int(getattr(p, "_grid_debris_min_area_px", 0) or 0),
            conflict_thresholds=p._pair_conflict_thresholds(),
            with_conflicts=bool(p._class_conflict_selected()),
            defect_labels={value: p._t(label_key) for label_key, value in GRID_INSPECTION_ERROR_TYPE_OPTIONS},
            folder_payload=self._folder_payload(layout),
            build_settings=p._build_build_settings_payload(),
            config_payload=dict(config_payload),
            performance_config=getattr(p._view, "performance_config", None),
            frames_per_row=int(spin.value()) if (spin := getattr(p._view, "frames_per_row_spin", None)) is not None else 0,
            sparse_reach_sites=float(getattr(p, "_grid_sparse_reach_sites", 4)),
        )

    @staticmethod
    def _folder_payload(layout: SnapshotLayout) -> dict:
        return {
            "folders": [
                {
                    "path": str(layer.mask_folder),
                    "checked": True,
                    "label": layer.label,
                    "confidence_path": str(layer.confidence_folder or ""),
                    "confidence_expanded": False,
                    "original_path": str(layer.source_folder or ""),
                    "frames_per_row": 0,
                    "cell_class": layer.cell_class,
                    "markup_path": "",
                    "manual_cells": [],
                }
                for layer in layout.layers
            ],
            "export_folder": None,
            "comparison_pairs": [],
            "comparison_pair_defaults_initialized": True,
        }

    # The run ------------------------------------------------------------------------------------

    def run(self) -> bool:
        if self._thread is not None:
            return False
        job = self._job()
        if job is None:
            self._page.show_message("batch.no_layers")
            return False
        self._keep_inputs()
        self._report = None
        self._last_view = None
        self._run_snapshot = str(job.layout.root)
        self._page.set_running(True)
        t = self._p._t
        thread = _BatchThread(
            job,
            lambda kind, name: t(f"batch.stage_title.{kind}", name=name),
            lambda kind: t(f"batch.problem.{kind}"),
            self,
        )
        thread.progressed.connect(self._progressed)
        thread.problems.connect(self._page.set_problems)
        thread.done.connect(self._done)
        thread.failed.connect(self._failed)
        thread.stopped.connect(self._stopped)
        thread.finished.connect(self._thread_done)
        self._thread = thread
        self._show()
        thread.start()
        return True

    def _progressed(self, view) -> None:
        self._last_view = view
        self._page.set_progress(view)

    def cancel(self) -> None:
        if self._thread is not None:
            self._thread.request_cancel()

    def toggle_pause(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.set_paused(not thread.paused)
        self._page.set_paused(thread.paused)

    def _thread_done(self) -> None:
        thread = self._thread
        self._thread = None
        self._page.set_running(False)
        if thread is not None:
            thread.deleteLater()
        self._show()

    def _ended(self, run: str, **values) -> Selection:
        selection = selections.find(self._selections, self._run_snapshot) or Selection(snapshot=self._run_snapshot)
        selection.run = run
        selection.at = selections.now_text()
        for name, value in values.items():
            setattr(selection, name, value)
        self._put(selection)
        return selection

    def _where(self) -> tuple[float, str]:
        """How far the run got and the stage it was at."""

        view = self._last_view
        if view is None:
            return 0.0, ""
        stage = next((item for item in view.stages if item.state == RUNNING), None)
        return float(view.share), stage.title if stage is not None else ""

    def _stopped(self) -> None:
        share, stage = self._where()
        self._ended(selections.RUN_STOPPED, share=share, stage=stage)
        self._page.show_message("batch.stopped")

    def _failed(self, message: str) -> None:
        share, stage = self._where()
        self._ended(selections.RUN_FAILED, share=share, stage=stage)
        self._page.show_message("batch.failed", error=message)

    def _done(self, report: BatchReport) -> None:
        self._report = report
        seconds = float(self._last_view.elapsed) if self._last_view is not None else 0.0
        if report.frames and seconds > 0:
            self._pace = seconds / report.frames
        self._ended(
            selections.RUN_DONE,
            share=1.0,
            stage="",
            project=str(report.project or ""),
            frames=report.frames,
            written=report.written,
            problems=len({frame for frame, _layer, _why in report.problems}),
            defects=report.defects,
            seconds=seconds,
        )
        self._page.set_done(report, seconds)
        project = self._projects()
        if project is not None and report.project is not None:
            project._remember(report.project)
        self.finished.emit(report)

    def open_output(self) -> None:
        if self._report is not None and self._report.output_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._report.output_dir)))

    # The matrix ---------------------------------------------------------------------------------

    def _show_workspace(self) -> None:
        show = getattr(self._p._view, "show_workspace", None)
        if callable(show):
            show()

    def open_matrix(self) -> bool:
        """The matrix of the chosen snapshot: its project at once, else computed there (the frames a
        stopped run checked come from the cell cache). A run going on stops and goes on there."""

        thread = self._thread
        if thread is not None:
            answer = QMessageBox.question(
                self._page,
                self._p._t("home.matrix"),
                self._p._t("home.m.move_run"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
            thread.request_cancel()
            thread.wait()
        layout = self._layout
        if layout is None or not layout.layers:
            self._show_workspace()
            return True
        selection = self.current()
        if (
            thread is None
            and selection is not None
            and selection.run == selections.RUN_DONE
            and selection.project
            and is_project(selection.project)
        ):
            project = self._projects()
            if project is not None and project.open(selection.project):
                self._show_workspace()
                return True
            return False
        return self.compute_in_matrix(layout)

    def compute_in_matrix(self, layout: SnapshotLayout) -> bool:
        """The snapshot's layers put in the matrix and its run started there."""

        p = self._p
        if getattr(p, "_worker_thread", None) is not None:
            QMessageBox.information(p._view, p._t("project.title"), p._t("project.busy"))
            return False
        p._clear_folders()
        p._apply_folder_manager_payload(self._folder_payload(layout))
        combo = getattr(p, "app_mode_combo", None)
        if combo is not None:
            index = combo.findData("grid_inspection")
            if index >= 0 and combo.currentIndex() != index:
                combo.setCurrentIndex(index)
        p._refresh_folder_rows()
        p._refresh_pair_matrix()
        self._show_workspace()
        p._on_primary_run_requested()
        return True

    def shutdown(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.request_cancel()
            thread.wait(10000)
