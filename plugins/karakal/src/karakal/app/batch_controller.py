"""«Compute and export»: a snapshot checked and its error maps written, with no matrix.

The snapshot folder is read for its layers (core.snapshot_layout); the frames are indexed, every
layer is checked by the same worker the matrix uses (the run's bank of normal cells, the conflicts
of a ones/zeros pair) with the app's settings, and the error map of every frame is written in the
chosen format. The run is saved as a project too: «Open in the matrix» shows it without
recomputing.
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
from PyQt6.QtWidgets import QFileDialog

from ..core.batch_progress import BatchProgress
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
    project_folder,
    relative_path,
    slim_results,
    stable_model_id,
    write_manifest,
    write_results,
)
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
    """The «Compute and export» page: snapshot, layers found, format, run, result."""

    finished = pyqtSignal(object)

    def __init__(self, presenter, page) -> None:
        super().__init__(presenter)
        self._p = presenter
        self._page = page
        self._layout: SnapshotLayout | None = None
        self._thread: _BatchThread | None = None
        self._report: BatchReport | None = None
        page.browseRequested.connect(self.browse_snapshot)
        page.snapshotChosen.connect(self.choose_snapshot)
        page.outputBrowseRequested.connect(self.browse_output)
        page.runRequested.connect(self.run)
        page.cancelRequested.connect(self.cancel)
        page.pauseRequested.connect(self.toggle_pause)
        page.openFolderRequested.connect(self.open_output)
        page.openProjectRequested.connect(self.open_project)

    def is_running(self) -> bool:
        return self._thread is not None

    def browse_snapshot(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self._page, self._p._t("batch.choose_snapshot"))
        if chosen:
            self.choose_snapshot(chosen)

    def browse_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self._page, self._p._t("batch.choose_output"), self._page.output_dir())
        if chosen:
            self._page.set_output_dir(chosen)

    def choose_snapshot(self, path: str) -> SnapshotLayout:
        layout = discover_snapshot(path)
        self._layout = layout
        self._page.set_layout(layout)
        if not self._page.output_dir():
            self._page.set_output_dir(str(layout.root))
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

    def run(self) -> bool:
        if self._thread is not None:
            return False
        job = self._job()
        if job is None:
            self._page.set_status("batch.no_layers")
            return False
        self._report = None
        self._page.set_running(True)
        self._page.set_status("batch.running")
        t = self._p._t
        thread = _BatchThread(
            job,
            lambda kind, name: t(f"batch.stage_title.{kind}", name=name),
            lambda kind: t(f"batch.problem.{kind}"),
            self,
        )
        thread.progressed.connect(self._page.set_progress)
        thread.problems.connect(self._page.set_problems)
        thread.done.connect(self._done)
        thread.failed.connect(lambda message: self._page.set_status("batch.failed", error=message))
        thread.stopped.connect(lambda: self._page.set_status("batch.stopped"))
        thread.finished.connect(self._thread_done)
        self._thread = thread
        thread.start()
        return True

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

    def _done(self, report: BatchReport) -> None:
        self._report = report
        self._page.set_done(report)
        project = getattr(self._p, "_project_controller", None)
        if project is not None and report.project is not None:
            project._remember(report.project)
        self.finished.emit(report)

    def open_output(self) -> None:
        if self._report is not None and self._report.output_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._report.output_dir)))

    def open_project(self) -> bool:
        report = self._report
        project = getattr(self._p, "_project_controller", None)
        if report is None or report.project is None or project is None:
            return False
        opened = project.open(report.project)
        if opened:
            show = getattr(self._p._view, "show_workspace", None)
            if callable(show):
                show()
        return opened

    def shutdown(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.request_cancel()
            thread.wait(10000)
