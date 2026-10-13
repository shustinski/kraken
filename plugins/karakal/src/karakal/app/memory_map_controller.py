"""Memory map tab: which networks, building in a thread, block roles, frame window, export, and
«Full memory analysis»: frames, cell defects and the map in one click.

The map is built only when the operator asks for it («Build map»). It is made from the layer shown
in the matrix: with its ones/zeros partner when the layer is in such a pair (both networks), else
from that layer alone (its «Cells» class tells whether it paints ones or zeros).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox

from ..core.memory_map import (
    MODE_ONES,
    MODE_PAIR,
    MODE_ZEROS,
    BuildCancelled,
    MapFrame,
    MemoryMap,
    build_memory_map,
    with_block_roles,
)
from ..core.memory_map_export import write_memory_map_files
from ..ui.ui_constants import FOLDER_LABEL_ROLE

_LOGGER = logging.getLogger(__name__)


class _BuildThread(QThread):
    progressed = pyqtSignal(str, int, int)
    built = pyqtSignal(object)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, frames, per_row: int, use_processes: bool, parent=None) -> None:
        super().__init__(parent)
        self._frames = frames
        self._per_row = per_row
        self._use_processes = use_processes
        self.cancel = False

    def run(self) -> None:
        try:
            mmap = build_memory_map(
                self._frames,
                per_row=self._per_row,
                progress=lambda stage, done, total: self.progressed.emit(stage, int(done), int(total)),
                cancelled=lambda: self.cancel,
                use_processes=self._use_processes,
                # Inside the app a process pool that does not start soon is given up for threads.
                startup_timeout=45.0,
            )
            self.built.emit(mmap)
        except BuildCancelled:
            self.stopped.emit()
        except Exception as error:  # shown to the operator
            _LOGGER.exception("Memory map build failed")
            self.failed.emit(str(error))


class MemoryMapController(QObject):
    def __init__(self, presenter, panel) -> None:
        super().__init__(presenter)
        self._presenter = presenter
        self._panel = panel
        self._thread: _BuildThread | None = None
        self._thread_state = None
        self._last_export = None
        self.use_processes = True
        self._pending_full = False
        panel.buildRequested.connect(self.build)
        panel.fullAnalysisRequested.connect(self.full_analysis)
        panel.cancelRequested.connect(self.cancel)
        panel.exportRequested.connect(self.export)
        panel.frameRequested.connect(self.open_frame)
        panel.blockRoleChanged.connect(self.set_block_role)

    def _t(self, key: str, **kwargs) -> str:
        return self._presenter._t(key, **kwargs)

    # --- What the map is made of ---------------------------------------------------------------

    def sources(self, state) -> dict | None:
        """Networks, frames and frames per row for the layer shown in the matrix."""

        presenter = self._presenter
        if state is None or not getattr(getattr(state, "build_result", None), "records", ()):
            return None
        shown = str(presenter._matrix_layer_model_id(state) or presenter._grid_inspection_model_id_for_state(state) or "")
        if not shown:
            return None
        pair = presenter._shown_grid_pair(state)
        if pair is None:
            pair = next((p for p in presenter._grid_conflict_pairs() if shown in (p.ones_id, p.zeros_id)), None)
        if pair is not None:
            mode, ones_id, zeros_id = MODE_PAIR, pair.ones_id, pair.zeros_id
        elif presenter._folder_cell_class(shown) == "zeros":
            mode, ones_id, zeros_id = MODE_ZEROS, "", shown
        else:
            mode, ones_id, zeros_id = MODE_ONES, shown, ""
        records = tuple(state.build_result.records)

        def path(record, model_id: str) -> str:
            return str((record.model_mask_paths or {}).get(model_id) or "") if model_id else ""

        defects = self._defect_finder(state)
        frames = [
            MapFrame(
                str(record.key),
                path(record, ones_id),
                path(record, zeros_id),
                defect_boxes=defects(ones_id, str(record.key)) + defects(zeros_id, str(record.key)),
            )
            for record in records
        ]
        # The layer's own frames per row when it is set, else «auto» (found from the bits).
        per_row = presenter._layer_frames_per_row(ones_id or zeros_id)
        return {
            "mode": mode,
            "ones_id": ones_id,
            "zeros_id": zeros_id,
            "frames": frames,
            "records": records,
            "per_row": max(0, int(per_row)),
            "with_masks": sum(1 for f in frames if f.ones_path or f.zeros_path),
        }

    def _defect_finder(self, state):
        """Boxes of the cells the defect search marked broken in a layer's frame (as the matrix
        shows them now: with its defect types and debris size); empty before any analysis."""

        from ..core.grid_anomaly import GridFrameAnalysisResult
        from ..core.grid_narrowing import narrow_grid_frame_result

        by_model = dict(getattr(state, "grid_inspection_payloads_by_model", {}) or {})
        shown_model = str(getattr(state, "grid_inspection_model_id", "") or "")
        by_layer = dict(getattr(state, "grid_inspection_payloads_by_layer", {}) or {})
        try:
            narrowing = self._presenter._grid_narrowing_for_state(state)
        except Exception:
            narrowing = None

        def boxes(model_id: str, key: str) -> tuple:
            if not model_id:
                return ()
            results = (by_model.get(model_id) or {}).get("binary") or {}
            if not results and model_id == shown_model:
                results = by_layer.get("binary") or {}
            result = results.get(key)
            if not isinstance(result, GridFrameAnalysisResult):
                return ()
            if narrowing is not None:
                result = narrow_grid_frame_result(result, narrowing)
            return tuple(
                tuple(int(v) for v in cell.bbox)
                for cell in result.per_cell_results
                if cell.status == "broken" and "conductor_zone" not in tuple(cell.reasons or ())
            )

        return boxes

    def _layer_name(self, model_id: str) -> str:
        item = self._presenter._folder_item_for_model_id(model_id) if model_id else None
        return str(item.data(FOLDER_LABEL_ROLE) or model_id) if item is not None else model_id

    def _signature(self, sources: dict, per_row: int) -> tuple:
        return (sources["mode"], sources["ones_id"], sources["zeros_id"], int(per_row), len(sources["frames"]))

    def refresh(self) -> None:
        """Show the tab's map (if built) and what a new one would be made of."""

        state = self._presenter._current_tab_state()
        sources = self.sources(state)
        if sources is None:
            self._panel.set_sources(self._t("memory_map.sources_none"), False, self._t("memory_map.sources_none"))
            self._panel.set_map(getattr(state, "memory_map", None) if state is not None else None)
            return
        names = {
            MODE_PAIR: self._t(
                "memory_map.sources_pair", ones=self._layer_name(sources["ones_id"]), zeros=self._layer_name(sources["zeros_id"])
            ),
            MODE_ONES: self._t("memory_map.sources_ones", name=self._layer_name(sources["ones_id"])),
            MODE_ZEROS: self._t("memory_map.sources_zeros", name=self._layer_name(sources["zeros_id"])),
        }
        text = names[sources["mode"]] + " " + self._t("memory_map.sources_frames", count=sources["with_masks"])
        busy = self._thread is not None
        self._panel.set_sources(text, sources["with_masks"] > 0 and not busy)
        mmap = getattr(state, "memory_map", None)
        if self._panel._map is not mmap:
            self._panel.set_map(mmap)
        if mmap is None and not busy:
            self._panel.set_per_row(sources["per_row"])
        signature = getattr(state, "memory_map_signature", None)
        if mmap is not None and not busy:
            self._panel.set_stale(signature != self._signature(sources, int(self._panel.per_row_spin.value())))

    # --- Building ------------------------------------------------------------------------------

    def build(self, per_row: int) -> None:
        if self._thread is not None:
            return
        state = self._presenter._current_tab_state()
        sources = self.sources(state)
        if sources is None or not sources["with_masks"]:
            return
        thread = _BuildThread(sources["frames"], int(per_row), self.use_processes, self)
        thread.progressed.connect(self._panel.set_busy)
        thread.built.connect(lambda mmap, state=state, sources=sources, per_row=per_row: self._built(state, sources, per_row, mmap))
        thread.failed.connect(lambda text: self._finish(self._t("memory_map.failed", error=text)))
        thread.stopped.connect(lambda: self._finish(self._t("memory_map.stopped")))
        thread.finished.connect(self._thread_done)
        self._thread = thread
        self._thread_state = state
        self._panel.set_busy("pitch", 0, 1)
        thread.start()

    def cancel(self) -> None:
        self._pending_full = False
        if self._thread is not None:
            self._thread.cancel = True

    # --- Full memory analysis ------------------------------------------------------------------

    def full_analysis(self) -> None:
        """Whatever is missing, in order: gather the frames and find the cell defects (the main run),
        then build the map and show it. With an up-to-date analysis the map is built at once."""

        presenter = self._presenter
        action, enabled, reason = presenter._run_action()
        if action == presenter.RUN_ACTION_STOP or self._thread is not None:
            return
        if not enabled:
            if reason:
                QMessageBox.information(self._panel, self._t("dialog.info_title"), reason)
            return
        if action == presenter.RUN_ACTION_RECOMPUTE:
            self._build_and_show()
            return
        self._pending_full = True
        self._panel.set_idle(self._t("memory_map.full_waiting"))
        if action == presenter.RUN_ACTION_RUN:
            presenter._on_primary_run_requested()
        else:
            presenter._on_compute_requested()
        presenter._sync_run_buttons()

    def analysis_finished(self) -> None:
        if not self._pending_full:
            return
        self._pending_full = False
        self._build_and_show()

    def cancel_pending(self) -> None:
        if self._pending_full:
            self._pending_full = False
            self._panel.set_idle(self._t("memory_map.stopped"))

    def _build_and_show(self) -> None:
        tabs = getattr(self._presenter._view, "grid_inspection_content_tabs", None)
        if tabs is not None:
            tabs.setCurrentWidget(self._panel)
        self.refresh()
        self.build(int(self._panel.per_row_spin.value()))

    def _built(self, state, sources: dict, per_row: int, mmap: MemoryMap) -> None:
        # The operator's roles of blocks of an earlier map carry over to blocks at the same place.
        roles = dict(getattr(state, "memory_map_roles", {}) or {})
        if roles:
            service = [b.index for b in mmap.blocks if roles.get(_block_key(b), b.service)]
            mmap = with_block_roles(mmap, service)
        state.memory_map = mmap
        state.memory_map_signature = self._signature(sources, per_row)
        counts = mmap.seam_counts()
        bad = sum(v for k, v in counts.items() if k not in ("bits", "empty"))
        message = self._t("memory_map.built", blocks=len(mmap.blocks), bad=bad)
        if int(per_row) <= 0:
            message += " " + self._t("memory_map.per_row_found", count=mmap.per_row)
        self._panel.set_map(mmap)
        self._finish(message)

    def _finish(self, message: str) -> None:
        self._panel.set_idle(message)

    def _thread_done(self) -> None:
        thread = self._thread
        self._thread = None
        self._thread_state = None
        if thread is not None:
            thread.deleteLater()
        self.refresh()

    def shutdown(self) -> None:
        if self._thread is not None:
            self._thread.cancel = True
            self._thread.wait(60000)

    # --- Operator actions ----------------------------------------------------------------------

    def set_block_role(self, index: int, service: bool) -> None:
        state = self._presenter._current_tab_state()
        mmap = getattr(state, "memory_map", None) if state is not None else None
        if mmap is None or not 0 <= index < len(mmap.blocks):
            return
        chosen = {b.index for b in mmap.blocks if b.service}
        (chosen.add if service else chosen.discard)(index)
        state.memory_map = with_block_roles(mmap, chosen)
        roles = dict(getattr(state, "memory_map_roles", {}) or {})
        roles[_block_key(mmap.blocks[index])] = bool(service)
        state.memory_map_roles = roles
        self._panel.set_map(state.memory_map, fit=False)

    def open_frame(self, row: int, col: int) -> None:
        state = self._presenter._current_tab_state()
        mmap = getattr(state, "memory_map", None) if state is not None else None
        if mmap is None:
            return
        point = mmap.frame_point(row, col)
        if point is None:
            return
        index, px, py = point
        records = tuple(state.build_result.records)
        if not 0 <= index < len(records):
            return
        w, h = 0.6 * mmap.pitch.x, 0.45 * mmap.pitch.y
        focus = {"bbox": [px - w / 2, py - h / 2, w, h]}
        self._presenter._open_record_details(records[index], state, grid_focus=focus)

    def export(self) -> None:
        from ..ui.memory_map_export_dialog import MemoryMapExportDialog

        state = self._presenter._current_tab_state()
        mmap = getattr(state, "memory_map", None) if state is not None else None
        if mmap is None:
            return
        previous = self._last_export
        if previous is None and getattr(self._presenter, "_export_folder", None):
            from ..core.memory_map_export import MemoryMapExport, TxtFormat, default_symbols

            previous = MemoryMapExport(folder=str(self._presenter._export_folder), custom=TxtFormat(symbols=default_symbols(mmap.mode)))
        dialog = MemoryMapExportDialog(self._t, mmap, previous, self._panel)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        choice = dialog.choice()
        self._last_export = choice
        try:
            written = write_memory_map_files(mmap, choice)
        except OSError as error:
            QMessageBox.warning(self._panel, self._t("dialog.warning_title"), self._t("memory_map.export_failed", error=str(error)))
            return
        box = QMessageBox(self._panel)
        box.setWindowTitle(self._t("memory_map.export_title"))
        box.setText(self._t("memory_map.export_done", count=len(written), folder=choice.folder))
        open_button = box.addButton(self._t("memory_map.open_folder"), QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(choice.folder))))


def _block_key(block) -> tuple[int, int, int, int]:
    return (int(block.row), int(block.col), int(block.rows), int(block.cols))
