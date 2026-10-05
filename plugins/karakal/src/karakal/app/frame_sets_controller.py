"""Drop rules, frame sets and set export for the grid inspection matrix.

Owns nothing but glue: the rules and sets live in ``state.frame_sets``
(``core/frame_sets.py``), the controls in ``ui/frame_sets_panel.py``. Every
operation works on the matrix layer chosen in the main window.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFileDialog, QMenu, QProgressDialog

from ..core.frame_set_export import (
    ITEM_ERRORS,
    ITEM_MASK,
    FrameSetExportFrame,
    FrameSetExportModel,
    FrameSetExportPlan,
    export_frame_set,
    format_size,
    preview_frame_set_export,
)
from ..core.frame_sets import (
    USER_SET_PREFIX,
    VIEW_ALL,
    VIEW_DROPPED,
    VIEW_KEPT,
    VIEW_SELECTED,
    FrameSetModel,
    frames_below,
)
from ..ui.frame_sets_panel import SLIDER_STEPS, FrameSetsPanel, FrameSetsSlider
from ..ui.ui_constants import GRID_INSPECTION_ERROR_TYPE_OPTIONS

_LOGGER = logging.getLogger(__name__)
REFRESH_DEBOUNCE_MS = 120
# Defect types drawn and counted in the export (zones are not defects of a cell).
_DEFECT_FILTER_SKIP = {"conductor_zone"}


def _percent(value: float) -> str:
    return f"{float(value) * 100.0:.0f}%"


class _ExportRunner(QObject):
    progressed = pyqtSignal(int, int)
    finished_report = pyqtSignal(object)

    def __init__(self, plan: FrameSetExportPlan, output_dir: Path) -> None:
        super().__init__()
        self._plan = plan
        self._output_dir = output_dir
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            report = export_frame_set(
                self._plan,
                self._output_dir,
                progress=lambda current, total, _name: self.progressed.emit(int(current), int(total)),
                cancelled=lambda: self._cancel,
            )
        except Exception as error:  # reported to the user, the app keeps running
            _LOGGER.exception("Frame set export failed")
            report = error
        self.finished_report.emit(report)


class FrameSetsController(QObject):
    """Connect the gradient panel, the matrix and the presenter."""

    def __init__(self, presenter) -> None:
        super().__init__(presenter)
        self._p = presenter
        self._t = presenter._t
        view = presenter._view
        self.legend = view.grid_inspection_legend
        self.matrix = view.grid_inspection_matrix_views["unified"]
        self.slider = FrameSetsSlider(self.legend)
        self.panel = FrameSetsPanel(self.legend)
        self.badge = getattr(view, "grid_inspection_view_badge", None)
        self.legend.attach_expansion(self.slider, self.panel)
        self._export_items = frozenset({ITEM_ERRORS, ITEM_MASK})
        self._export_canvas = False
        self._export_table = True
        self._export_fill_black = False
        self._export_set_id = ""
        self._last_report = None
        self._export_thread: QThread | None = None
        self._defect_cache: tuple[int, int, dict[str, dict[str, int]]] | None = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(REFRESH_DEBOUNCE_MS)
        self._refresh_timer.timeout.connect(self.refresh)
        self.panel.set_export_items(
            self._export_items,
            canvas=self._export_canvas,
            table=self._export_table,
            fill_black=self._export_fill_black,
        )
        self._connect()

    # Wiring ------------------------------------------------------------------

    def _connect(self) -> None:
        panel = self.panel
        self.matrix.colorScaleChanged.connect(lambda _info: self.schedule_refresh())
        self.matrix.rangeSelectionChanged.connect(lambda _count: self.schedule_refresh())
        self.legend.expandToggled.connect(lambda _expanded: self.refresh())
        self.slider.valueChanged.connect(lambda _value: self._refresh_threshold())
        panel.applyRequested.connect(self._apply_rule)
        panel.invertRequested.connect(self._toggle_invert)
        panel.ruleToggled.connect(self._on_rule_toggled)
        panel.ruleRemoved.connect(self._on_rule_removed)
        panel.manualCleared.connect(self._on_manual_cleared)
        panel.viewChosen.connect(self._on_view_chosen)
        panel.setRemoved.connect(self._on_set_removed)
        panel.addSelectedRequested.connect(self._add_selected)
        panel.selectShownRequested.connect(self._select_shown)
        panel.clearSelectionRequested.connect(lambda: self.matrix.set_range_selected_keys(()))
        panel.exportOpened.connect(self._on_export_opened)
        panel.exportChoicesChanged.connect(self._on_export_choices_changed)
        panel.exportBrowseRequested.connect(self._browse_output_dir)
        panel.exportRequested.connect(self._run_export)
        panel.openFolderRequested.connect(self._open_export_folder)
        panel.showReportRequested.connect(self._open_export_report)
        if self.badge is not None:
            self.badge.resetRequested.connect(lambda: self._on_view_chosen(VIEW_ALL))

    def schedule_refresh(self) -> None:
        self._refresh_timer.start()

    # State helpers -------------------------------------------------------------

    def _state(self):
        state = self._p._current_tab_state()
        if state is None or not hasattr(state, "frame_sets"):
            return None
        return state

    def _model(self, state) -> FrameSetModel:
        return state.frame_sets

    def _all_keys(self, state) -> set[str]:
        return {str(record.key) for record in getattr(state.build_result, "records", ()) or ()}

    def _qualities(self) -> dict[str, float]:
        return self.matrix.grid_quality_by_key()

    def _grid_mode(self) -> bool:
        try:
            return self._p._current_app_mode() == "grid_inspection"
        except Exception:
            return False

    def _layer_title(self) -> str:
        combo = getattr(self._p._view, "grid_matrix_layer_combo", None)
        try:
            return str(combo.currentText() or "") if combo is not None and combo.isVisible() else ""
        except RuntimeError:
            return ""

    # Refresh -------------------------------------------------------------------

    def refresh(self) -> None:
        """Apply the shown set to the matrix and analysis, then update the panel."""

        state = self._state()
        if state is None or not self._grid_mode():
            self.matrix.set_participating_keys(None)
            self.matrix.set_pending_drop_keys(())
            self.legend.set_threshold_marker(None)
            if self.badge is not None:
                self.badge.clear()
            return
        model = self._model(state)
        qualities = self._qualities()
        if not model.has_filter():
            model.frozen_window = None
        self.matrix.set_frozen_color_window(model.frozen_window)
        all_keys = self._all_keys(state)
        participating = model.participating_keys(all_keys, qualities, self.matrix.range_selected_keys())
        self.matrix.set_participating_keys(participating)
        self._refresh_badge(state, participating, all_keys)
        if participating != state.frame_set_participating:
            state.frame_set_participating = None if participating is None else set(participating)
            state.frame_set_version += 1
            self._p._refresh_grid_inspection_errors_panel(state)
            self._p._schedule_metric_histogram_update(state)
        self._refresh_panel()
        self._refresh_cards()

    def _view_name(self, model: FrameSetModel, view_id: str) -> str:
        names = {
            VIEW_ALL: self._t("frame_sets.sets.all"),
            VIEW_KEPT: self._t("frame_sets.sets.kept_inverted" if model.inverted else "frame_sets.sets.kept"),
            VIEW_DROPPED: self._t("frame_sets.sets.dropped"),
            VIEW_SELECTED: self._t("frame_sets.sets.selected"),
        }
        if view_id in names:
            return names[view_id]
        for item in model.user_sets:
            if view_id == f"{USER_SET_PREFIX}{item.set_id}":
                return item.name
        return ""

    def _refresh_badge(self, state, participating, all_keys) -> None:
        """Strip above the matrix, shown whenever only a part of the run is bright."""

        if self.badge is None:
            return
        if participating is None:
            self.badge.clear()
            return
        model = self._model(state)
        self.badge.show_view(self._view_name(model, model.view_id), len(participating), len(all_keys))

    def _view_note(self, participating, all_keys) -> str:
        """What the chosen set does to the matrix, in words."""

        total = len(all_keys)
        if participating is None:
            return self._t("frame_sets.view_note.all", total=total)
        count = len(participating & all_keys)
        if count == 0:
            return self._t("frame_sets.view_note.empty")
        return self._t("frame_sets.view_note.part", count=count, rest=total - count)

    def _cutoff(self, qualities: dict[str, float]) -> float | None:
        value = self.slider.value()
        if value <= 0 or not qualities:
            return None
        return self.matrix.quality_for_palette_position(value / SLIDER_STEPS)

    def _refresh_panel(self) -> None:
        state = self._state()
        if state is None or not self.legend.is_expanded():
            self.matrix.set_pending_drop_keys(())
            self.legend.set_threshold_marker(None)
            return
        model = self._model(state)
        qualities = self._qualities()
        all_keys = self._all_keys(state)
        selected = self.matrix.range_selected_keys()
        dropped = self._refresh_threshold(state, qualities)
        self.panel.set_inverted(model.inverted, can_invert=bool(dropped))
        self.panel.set_rules(
            [
                (rule.rule_id, self._rule_label(rule), len(frames_below(qualities, rule.cutoff)), rule.enabled)
                for rule in model.rules
            ],
            len(model.manual_dropped),
        )
        self.panel.set_sets(self._set_rows(state, qualities, all_keys, selected, removable=True), model.view_id)
        self.panel.set_view_note(self._view_note(model.participating_keys(all_keys, qualities, selected), all_keys))
        self.panel.set_targets([(f"{USER_SET_PREFIX}{item.set_id}", item.name) for item in model.user_sets])
        self.panel.set_selected_count(len(selected))
        if self.panel.export_open():
            self._refresh_export(state, qualities, all_keys, selected)

    def _refresh_threshold(self, state=None, qualities: dict[str, float] | None = None) -> set[str]:
        """Marker, blinking frames and the threshold text; returns frames dropped already."""

        state = state or self._state()
        if state is None or not self.legend.is_expanded():
            return set()
        model = self._model(state)
        qualities = self._qualities() if qualities is None else qualities
        dropped = model.dropped_keys(qualities)
        cutoff = self._cutoff(qualities)
        if cutoff is None:
            pending: set[str] = set()
            marker = 0.0
            text = self._t("frame_sets.threshold.none" if qualities else "frame_sets.threshold.no_results")
        else:
            below = frames_below(qualities, cutoff)
            pending = below - dropped
            share = len(below) / max(1, len(qualities))
            marker = self.slider.value() / SLIDER_STEPS
            text = self._t(
                "frame_sets.threshold.scale",
                quality=_percent(cutoff),
                count=len(below),
                share=f"{share * 100.0:.1f}%",
            )
        self.legend.set_threshold_marker(marker)
        self.matrix.set_pending_drop_keys(pending)
        self.panel.set_threshold_text(text)
        self.panel.set_pending_count(len(pending))
        return dropped

    def _rule_label(self, rule) -> str:
        label = self._t("frame_sets.rule.scale", quality=_percent(rule.cutoff))
        return f"{label} · {rule.layer_label}" if rule.layer_label else label

    def _set_rows(self, state, qualities, all_keys, selected, *, removable: bool):
        model = self._model(state)
        kept_name = self._t("frame_sets.sets.kept_inverted" if model.inverted else "frame_sets.sets.kept")
        rows = [
            (VIEW_ALL, self._t("frame_sets.sets.all"), len(all_keys), False),
            (VIEW_KEPT, kept_name, len(model.set_keys(VIEW_KEPT, all_keys, qualities)), False),
            (VIEW_DROPPED, self._t("frame_sets.sets.dropped"), len(model.set_keys(VIEW_DROPPED, all_keys, qualities)), False),
            (VIEW_SELECTED, self._t("frame_sets.sets.selected"), len(selected & all_keys), False),
        ]
        for item in model.user_sets:
            view_id = f"{USER_SET_PREFIX}{item.set_id}"
            rows.append((view_id, item.name, len(set(item.keys) & all_keys), removable))
        return rows

    # Drop rules ----------------------------------------------------------------

    def _ensure_frozen(self, model: FrameSetModel) -> None:
        if model.frozen_window is None:
            model.frozen_window = self.matrix.color_window()

    def _show_kept_after_drop(self, model: FrameSetModel) -> None:
        if model.view_id == VIEW_ALL:
            model.view_id = VIEW_KEPT

    def _apply_rule(self) -> None:
        state = self._state()
        if state is None:
            return
        qualities = self._qualities()
        cutoff = self._cutoff(qualities)
        if cutoff is None:
            return
        model = self._model(state)
        self._ensure_frozen(model)
        model.add_rule(float(self.slider.value()), float(cutoff), self._layer_title())
        self._show_kept_after_drop(model)
        self.refresh()

    def _toggle_invert(self) -> None:
        state = self._state()
        if state is None:
            return
        model = self._model(state)
        model.inverted = not model.inverted
        self._show_kept_after_drop(model)
        self.refresh()

    def _on_rule_toggled(self, rule_id: int, enabled: bool) -> None:
        state = self._state()
        if state is None:
            return
        self._model(state).set_rule_enabled(rule_id, enabled)
        QTimer.singleShot(0, self.refresh)

    def _on_rule_removed(self, rule_id: int) -> None:
        state = self._state()
        if state is None:
            return
        model = self._model(state)
        model.remove_rule(rule_id)
        if not model.has_filter():
            model.inverted = False
        QTimer.singleShot(0, self.refresh)

    def _on_manual_cleared(self) -> None:
        state = self._state()
        if state is None:
            return
        model = self._model(state)
        model.clear_manual()
        if not model.has_filter():
            model.inverted = False
        QTimer.singleShot(0, self.refresh)

    # Sets ------------------------------------------------------------------------

    def _on_view_chosen(self, view_id: str) -> None:
        state = self._state()
        if state is None:
            return
        model = self._model(state)
        if model.view_id == view_id:
            return
        model.view_id = str(view_id)
        QTimer.singleShot(0, self.refresh)

    def _on_set_removed(self, view_id: str) -> None:
        state = self._state()
        if state is None or not str(view_id).startswith(USER_SET_PREFIX):
            return
        self._model(state).remove_set(int(str(view_id)[len(USER_SET_PREFIX):]))
        QTimer.singleShot(0, self.refresh)

    def _add_keys_to_target(self, state, target: str, keys) -> None:
        model = self._model(state)
        keys = {str(key) for key in keys}
        if not keys:
            return
        if str(target).startswith(USER_SET_PREFIX):
            model.extend_set(int(str(target)[len(USER_SET_PREFIX):]), keys)
        else:
            model.add_set(model.next_set_name(self._t("frame_sets.sets.base_name")), keys)

    def _add_selected(self, target: str) -> None:
        state = self._state()
        if state is None:
            return
        self._add_keys_to_target(state, target, self.matrix.range_selected_keys())
        self.refresh()

    def _select_shown(self) -> None:
        state = self._state()
        if state is None:
            return
        shown = state.frame_set_participating
        self.matrix.set_range_selected_keys(self._all_keys(state) if shown is None else shown)

    # Defect counts (results table of the export) -----------------------------

    def _defect_counts(self, state) -> dict[str, dict[str, int]]:
        payloads = getattr(state, "grid_inspection_payload_by_key", {}) or {}
        signature = (id(payloads), len(payloads))
        if self._defect_cache is not None and self._defect_cache[:2] == signature:
            return self._defect_cache[2]
        counts: dict[str, dict[str, int]] = {}
        for key, result in payloads.items():
            frame_counts: dict[str, int] = {}
            for cell in getattr(result, "per_cell_results", getattr(result, "cells", ())) or ():
                status = str(getattr(cell, "status", "") or "")
                if status == "normal":
                    continue
                reasons = {str(reason) for reason in (getattr(cell, "reasons", ()) or ())}
                reasons.add(self._p._grid_inspection_error_type(status, tuple(sorted(reasons))))
                for reason in reasons:
                    frame_counts[reason] = frame_counts.get(reason, 0) + 1
            counts[str(key)] = frame_counts
        self._defect_cache = (signature[0], signature[1], counts)
        return counts

    # Frame card ----------------------------------------------------------------

    def card_frame_dropped(self, key: str) -> bool:
        state = self._state()
        if state is None:
            return False
        return str(key) in self._model(state).dropped_keys(self._qualities())

    def card_targets(self) -> list[tuple[str, str]]:
        state = self._state()
        if state is None:
            return []
        return [("new", self._t("frame_sets.sets.new"))] + [
            (f"{USER_SET_PREFIX}{item.set_id}", item.name) for item in self._model(state).user_sets
        ]

    def card_add_to_set(self, key: str, target: str) -> None:
        state = self._state()
        if state is None:
            return
        self._add_keys_to_target(state, target, {str(key)})
        self.refresh()

    def card_toggle_drop(self, key: str) -> None:
        state = self._state()
        if state is None:
            return
        model = self._model(state)
        qualities = self._qualities()
        if str(key) in model.dropped_keys(qualities):
            model.return_frame(str(key), qualities)
            if not model.has_filter():
                model.inverted = False
        else:
            self._ensure_frozen(model)
            model.drop_frame(str(key))
            self._show_kept_after_drop(model)
        self.refresh()

    def install_card_actions(self, dialog) -> None:
        install = getattr(dialog, "install_frame_set_actions", None)
        if callable(install):
            install(self)

    def _refresh_cards(self) -> None:
        for dialog in list(getattr(self._p, "_details_dialogs", ()) or ()):
            refresh = getattr(dialog, "refresh_frame_set_actions", None)
            if callable(refresh):
                try:
                    refresh()
                except RuntimeError:
                    continue

    def build_card_menu(self, key: str, parent) -> QMenu:
        menu = QMenu(parent)
        for target, name in self.card_targets():
            menu.addAction(name, lambda value=target: self.card_add_to_set(key, value))
        return menu

    # Export ----------------------------------------------------------------------

    def _output_dir(self) -> Path:
        return Path(getattr(self._p, "_export_folder", None) or Path.cwd())

    def _on_export_opened(self) -> None:
        state = self._state()
        if state is not None:
            self._export_set_id = self._model(state).view_id
        self._refresh_panel()

    def _on_export_choices_changed(self) -> None:
        self._export_set_id = self.panel.export_set_id() or self._export_set_id
        self._export_items = self.panel.export_items()
        self._export_canvas = self.panel.export_canvas()
        self._export_table = self.panel.export_table()
        self._export_fill_black = self.panel.export_fill_black()
        QTimer.singleShot(0, self._refresh_panel)

    def _refresh_export(self, state, qualities, all_keys, selected) -> None:
        rows = [(view_id, name, count) for view_id, name, count, _removable in self._set_rows(
            state, qualities, all_keys, selected, removable=False
        )]
        if self._export_set_id not in {row[0] for row in rows}:
            self._export_set_id = VIEW_KEPT
        self.panel.set_export_title(
            self._t(
                "frame_sets.export.title",
                layer=self._layer_title() or ", ".join(title for _id, title in self._export_models(state)),
            )
        )
        self.panel.set_export_sets(rows, self._export_set_id)
        self.panel.set_output_dir(str(self._output_dir()))
        plan = self._build_plan(state, qualities, all_keys, selected)
        set_count = len(self._model(state).set_keys(self._export_set_id, all_keys, qualities, selected))
        self.panel.set_fill_black_available(ITEM_ERRORS in plan.items and 0 < set_count < len(all_keys))
        has_items = bool(plan.items) or plan.canvas or plan.table
        if not plan.frames or not has_items:
            self.panel.set_export_preview("", self._t("frame_sets.export.nothing"), can_run=False)
            return
        preview = preview_frame_set_export(plan, self._output_dir())
        self.panel.set_export_preview(
            "\n".join(preview.lines),
            self._t("frame_sets.export.summary", files=preview.file_count, size=format_size(preview.approx_bytes)),
            can_run=True,
        )

    def _export_models(self, state) -> list[tuple[str, str]]:
        specs = [
            (str(spec.model_id), str(spec.display_name or spec.model_id))
            for spec in (getattr(state.build_result, "model_specs", ()) or ())
            if str(getattr(spec, "model_id", "") or "")
        ]
        selection = self._p._grid_inspection_matrix_selection(state)
        chosen = [item for item in specs if item[0] == selection]
        return chosen or specs

    def _results_for_model(self, state, model_id: str, single: bool) -> dict[str, object]:
        layer = self._p._grid_export_analysis_layer(state)
        model_layers = dict((getattr(state, "grid_inspection_payloads_by_model", {}) or {}).get(model_id) or {})
        results = dict(model_layers.get(layer) or model_layers.get("binary") or {})
        if not results and single:
            results = dict(getattr(state, "grid_inspection_payload_by_key", {}) or {})
        return results

    def _build_plan(self, state, qualities, all_keys, selected) -> FrameSetExportPlan:
        model = self._model(state)
        keys = model.set_keys(self._export_set_id, all_keys, qualities, selected)
        set_name = next(
            (name for view_id, name, _count, _removable in self._set_rows(state, qualities, all_keys, selected, removable=False)
             if view_id == self._export_set_id),
            self._t("frame_sets.sets.all"),
        )
        models = self._export_models(state)
        results_by_model = {
            model_id: self._results_for_model(state, model_id, len(models) == 1) for model_id, _title in models
        }
        counts = self._defect_counts(state)
        kept = model.kept_keys(all_keys, qualities)
        fill_black = self._export_fill_black and ITEM_ERRORS in self._export_items
        frames = []
        fill_frames = []
        for record in getattr(state.build_result, "records", ()) or ():
            key = str(record.key)
            if key not in keys:
                if fill_black:
                    fill_frames.append(
                        FrameSetExportFrame(
                            key=key,
                            name=Path(str(record.display_name or key)).name,
                            results={model_id: results_by_model[model_id].get(key) for model_id, _ in models},
                        )
                    )
                continue
            position = self.matrix.record_position(key) or (-1, -1)
            frames.append(
                FrameSetExportFrame(
                    key=key,
                    name=Path(str(record.display_name or key)).name,
                    original_path=str(record.original_path or record.base_path or ""),
                    mask_paths={model_id: str((record.model_mask_paths or {}).get(model_id) or "") for model_id, _ in models},
                    confidence_paths={
                        model_id: str((record.model_prob_paths or {}).get(model_id) or "") for model_id, _ in models
                    },
                    results={model_id: results_by_model[model_id].get(key) for model_id, _ in models},
                    quality=qualities.get(key),
                    row=int(position[0]),
                    column=int(position[1]),
                    participating=key in kept,
                    defect_counts=dict(counts.get(key, {})),
                )
            )
        error_types = tuple(
            value for value in self._p._selected_grid_error_types() if value not in _DEFECT_FILTER_SKIP
        )
        canvas_colors = {}
        if self._export_canvas:
            for frame in frames:
                color = self.matrix.frame_color_rgb(frame.key)
                if color is not None:
                    canvas_colors[frame.key] = color
        return FrameSetExportPlan(
            set_name=set_name,
            layer_title=self._layer_title() or ", ".join(title for _id, title in models),
            frames=tuple(frames),
            models=tuple(FrameSetExportModel(model_id, title) for model_id, title in models),
            items=self._export_items,
            canvas=self._export_canvas,
            table=self._export_table,
            error_types=error_types,
            defect_labels={value: self._t(label_key) for label_key, value in GRID_INSPECTION_ERROR_TYPE_OPTIONS},
            canvas_colors=canvas_colors,
            matrix_size=self.matrix.matrix_shape(),
            rules_text=tuple(self._rule_label(rule) for rule in model.rules if rule.enabled)
            + ((self._t("frame_sets.rules.manual") + f": {len(model.manual_dropped)}",) if model.manual_dropped else ())
            + ((self._t("frame_sets.inverted_note"),) if model.inverted else ()),
            fill_frames=tuple(fill_frames) if frames else (),
        )

    def _browse_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self._p._view, self._t("frame_sets.export.choose_dir"), str(self._output_dir())
        )
        if folder:
            self._p._export_folder = Path(folder)
            self._refresh_panel()

    def _run_export(self) -> None:
        state = self._state()
        if state is None or self._export_thread is not None:
            return
        plan = self._build_plan(
            state, self._qualities(), self._all_keys(state), self.matrix.range_selected_keys()
        )
        if not plan.frames:
            return
        output_dir = self._output_dir()
        progress = QProgressDialog(
            self._t("frame_sets.export.running"),
            self._t("common.cancel"),
            0,
            max(1, len(plan.frames) + len(plan.fill_frames)),
            self._p._view,
        )
        progress.setMinimumDuration(300)
        progress.setValue(0)
        thread = QThread(self)
        runner = _ExportRunner(plan, output_dir)
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.progressed.connect(lambda current, total: (progress.setMaximum(max(1, total)), progress.setValue(current)))
        progress.canceled.connect(runner.cancel)

        def finished(report) -> None:
            progress.reset()
            progress.deleteLater()
            thread.quit()
            thread.wait()
            self._export_thread = None
            self._export_runner = None
            self._on_export_finished(report)

        runner.finished_report.connect(finished)
        self._export_thread = thread
        self._export_runner = runner
        thread.start()

    def _on_export_finished(self, report) -> None:
        if isinstance(report, Exception):
            self._last_report = None
            self.panel.set_export_done(str(report))
            return
        self._last_report = report
        if report.cancelled:
            text = self._t("frame_sets.export.cancelled", count=report.written, path=str(report.set_dir))
        elif report.missing:
            text = self._t(
                "frame_sets.export.done_missing",
                count=report.written,
                path=str(report.set_dir),
                missing=len(report.missing),
            )
        else:
            text = self._t("frame_sets.export.done", count=report.written, path=str(report.set_dir))
        self.panel.set_export_done(text)

    def _open_export_folder(self) -> None:
        if self._last_report is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_report.set_dir)))

    def _open_export_report(self) -> None:
        if self._last_report is not None and self._last_report.report_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_report.report_path)))
