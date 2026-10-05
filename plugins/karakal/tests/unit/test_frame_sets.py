"""Drop rules, frame sets and set export of the grid inspection matrix."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings

from karakal.app.main_window import KarakalWidget
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
from karakal.core.frame_set_export import (
    CSV_FILE,
    ITEM_ERRORS,
    ITEM_ERRORS_OVER_MASK,
    ITEM_MASK,
    ITEM_SOURCE,
    REPORT_FILE,
    FrameSetExportFrame,
    FrameSetExportModel,
    FrameSetExportPlan,
    export_frame_set,
    preview_frame_set_export,
)
from karakal.core.frame_sets import (
    VIEW_ALL,
    VIEW_DROPPED,
    VIEW_KEPT,
    VIEW_SELECTED,
    FrameSetModel,
)
from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.ui.frame_sets_panel import SLIDER_STEPS
from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE

QUALITIES = {f"f{index}": index / 10.0 for index in range(10)}


def test_rule_drops_frames_below_cutoff_and_invert_keeps_only_them() -> None:
    model = FrameSetModel()
    keys = set(QUALITIES)
    model.add_rule(300.0, 0.3)
    assert model.dropped_keys(QUALITIES) == {"f0", "f1", "f2"}
    assert model.kept_keys(keys, QUALITIES) == keys - {"f0", "f1", "f2"}
    model.inverted = True
    assert model.kept_keys(keys, QUALITIES) == {"f0", "f1", "f2"}
    assert model.set_keys(VIEW_DROPPED, keys, QUALITIES) == keys - {"f0", "f1", "f2"}


def test_disabled_rule_and_manual_drop_and_return() -> None:
    model = FrameSetModel()
    keys = set(QUALITIES)
    rule = model.add_rule(300.0, 0.3)
    model.set_rule_enabled(rule.rule_id, False)
    assert model.dropped_keys(QUALITIES) == set()
    model.set_rule_enabled(rule.rule_id, True)
    model.drop_frame("f9")
    model.return_frame("f1", QUALITIES)
    assert model.dropped_keys(QUALITIES) == {"f0", "f2", "f9"}
    model.return_frame("f9", QUALITIES)
    assert "f9" in model.kept_keys(keys, QUALITIES)


def test_user_sets_and_selected_view() -> None:
    model = FrameSetModel()
    keys = set(QUALITIES)
    frame_set = model.add_set("A", {"f1", "f2"})
    model.extend_set(frame_set.set_id, {"f3"})
    model.view_id = f"set:{frame_set.set_id}"
    assert model.participating_keys(keys, QUALITIES) == {"f1", "f2", "f3"}
    model.view_id = VIEW_SELECTED
    assert model.participating_keys(keys, QUALITIES, {"f5", "nope"}) == {"f5"}
    model.remove_set(frame_set.set_id)
    assert model.next_set_name("Выборка") == "Выборка 1"


def _mask(path: Path) -> None:
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:30, 10:30] = 255
    assert _grayscale_array_to_qimage(image).save(str(path))


def _result(key: str, defects: int, *, size: int = 40) -> GridFrameAnalysisResult:
    cells = tuple(
        GridCellAnalysisResult(0, index, (index * 4, 0, 3, 3), (index * 4 + 1.5, 1.5), None, "broken", 0.9, ("merged_contour",))
        for index in range(defects)
    )
    return GridFrameAnalysisResult(
        frame_id=key, frame_path=f"{key}.png", image_width=size, image_height=size, grid_rows=1, grid_cols=9,
        total_expected_cells=9, detected_cells=9, normal_cells=9 - defects, suspicious_cells=0,
        broken_cells=defects, missing_cells=0, artifact_cells=0, damage_score=min(1.0, defects / 9.0),
        severity_level="", grid_detected=True, per_cell_results=cells,
    )


def test_export_writes_chosen_items_table_and_report(tmp_path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    frames = []
    for index in range(3):
        mask = masks / f"frame.{index:04d}.png"
        _mask(mask)
        frames.append(
            FrameSetExportFrame(
                key=f"k{index}", name=mask.name, original_path=str(mask), mask_paths={"m": str(mask)},
                results={"m": _result(f"k{index}", index + 1)}, quality=0.5, row=0, column=index,
                defect_counts={"merged_contour": index + 1},
            )
        )
    plan = FrameSetExportPlan(
        set_name="Участвуют", layer_title="Модель", frames=tuple(frames), models=(FrameSetExportModel("m", "Модель"),),
        items=frozenset({ITEM_SOURCE, ITEM_MASK, ITEM_ERRORS, ITEM_ERRORS_OVER_MASK}), canvas=True, table=True,
        error_types=("merged_contour",), matrix_size=(1, 3),
    )
    preview = preview_frame_set_export(plan, tmp_path / "out")
    assert any("Маски" in line and "3 файлов" in line for line in preview.lines)
    report = export_frame_set(plan, tmp_path / "out")
    assert not report.missing
    set_dir = report.set_dir
    assert len(list((set_dir / "Ошибки").glob("*.png"))) == 3
    assert sorted(path.name for path in (set_dir / "Ошибки на маске").glob("*.png")) == [
        "frame.0000.png", "frame.0001.png", "frame.0002.png",
    ]
    assert len(list((set_dir / "Маски").iterdir())) == 3
    assert (set_dir / "холст.png").is_file()
    with (set_dir / CSV_FILE).open(encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    assert len(rows) == 4
    assert (set_dir / REPORT_FILE).read_text(encoding="utf-8").startswith("Экспорт Karakal")


def test_export_fills_the_rest_of_the_run_with_black_error_frames(tmp_path) -> None:
    chosen = [
        FrameSetExportFrame(key=f"k{index}", name=f"frame.{index:04d}.png", results={"m": _result(f"k{index}", 2)})
        for index in range(2)
    ]
    rest = [
        FrameSetExportFrame(key=f"k{index}", name=f"frame.{index:04d}.png", results={"m": _result(f"k{index}", 3)})
        for index in range(2, 5)
    ]
    plan = FrameSetExportPlan(
        set_name="Выделено", layer_title="Модель", frames=tuple(chosen), models=(FrameSetExportModel("m", "Модель"),),
        items=frozenset({ITEM_ERRORS}), error_types=("merged_contour",), fill_frames=tuple(rest),
    )
    preview = preview_frame_set_export(plan, tmp_path / "out")
    errors_line = next(line for line in preview.lines if "Ошибки" in line)
    assert "5 файлов" in errors_line and "чёрных: 3" in errors_line
    # Every number of the tree starts in the same column.
    starts = {match.end() - 1 for line in preview.lines if (match := re.search(r"\\\s+\d", line))}
    assert len(starts) == 1
    report = export_frame_set(plan, tmp_path / "out")
    files = sorted((report.set_dir / "Ошибки").glob("*.png"))
    assert [path.name for path in files] == [f"frame.{index:04d}.png" for index in range(5)]
    for path in files[2:]:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert image.shape[:2] == (40, 40) and not image.any()
    first = cv2.imdecode(np.fromfile(str(files[0]), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert first.any()
    assert "чёрные: 3" in (report.set_dir / REPORT_FILE).read_text(encoding="utf-8")


def _grid_widget(tmp_path: Path, qtbot):
    folder = tmp_path / "masks"
    folder.mkdir()
    names = [f"frame_{index:04d}.png" for index in range(10)]
    for name in names:
        _mask(folder / name)
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    item = presenter._append_folder_item(folder, checked=True)
    item.setData(FOLDER_CHECKED_ROLE, True)
    specs = presenter._checked_model_specs()
    model_id = str(specs[0].model_id)
    records = tuple(
        FrameRecord(Path(name).stem, name, model_mask_paths={model_id: str(folder / name)}) for name in names
    )
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(BuildResult(records=records, options=BuildOptions(), model_specs=specs))
    state = presenter._current_tab_state()
    state.grid_inspection_payloads_by_model = {
        model_id: {"binary": {record.key: _result(record.key, index) for index, record in enumerate(records)}}
    }
    state.grid_inspection_matrix_selection = model_id
    state.grid_inspection_results_ready = True
    presenter._apply_grid_inspection_active_layer_view(state, streaming=False)
    controller = presenter._frame_sets_controller
    controller.refresh()
    return widget, presenter, state, controller, records


def test_panel_drops_worst_frames_inverts_and_limits_errors(tmp_path, qtbot) -> None:
    widget, presenter, state, controller, records = _grid_widget(tmp_path, qtbot)
    legend = widget.grid_inspection_legend
    legend.set_expanded(True)
    qualities = controller._qualities()
    worst = {records[index].key for index in (7, 8, 9)}
    cutoff = min(qualities[key] for key in set(qualities) - worst)
    position = controller.matrix.palette_position_for_quality(cutoff)
    controller.slider.slider.setValue(int(round(position * SLIDER_STEPS)))
    assert controller.matrix._pending_drop_keys == worst
    assert controller.panel.apply_button.isEnabled()

    controller.panel.apply_button.click()
    model = state.frame_sets
    assert len(model.rules) == 1
    assert model.view_id == VIEW_KEPT
    assert model.frozen_window is not None
    assert state.frame_set_participating == {record.key for record in records} - worst
    assert controller.matrix._pending_drop_keys == set()
    listed = {payload["record_key"] for payload in presenter._iter_grid_inspection_error_payloads(state)}
    assert listed and not (listed & worst)
    badge = widget.grid_inspection_view_badge
    assert not badge.isHidden() and "7" in badge.text_label.text() and "10" in badge.text_label.text()
    assert controller.panel.view_note.text() == controller._t("frame_sets.view_note.part", count=7, rest=3)
    # The test payloads live only in the matrix, so refresh by hand instead of running the event loop.
    badge.reset_button.click()
    controller.refresh()
    assert model.view_id == VIEW_ALL and badge.isHidden()
    controller._on_view_chosen(VIEW_KEPT)
    controller.refresh()

    controller.panel.invert_button.click()
    assert state.frame_set_participating == worst

    controller.card_toggle_drop(records[0].key)
    assert controller.card_frame_dropped(records[0].key)
    assert records[0].key in state.frame_set_participating


def test_selection_set(tmp_path, qtbot) -> None:
    widget, presenter, state, controller, records = _grid_widget(tmp_path, qtbot)
    widget.grid_inspection_legend.set_expanded(True)
    model = state.frame_sets

    controller.matrix.set_range_selected_keys({records[1].key, records[2].key})
    controller.refresh()
    controller._add_selected("new")
    assert set(model.user_sets[-1].keys) == {records[1].key, records[2].key}


def test_panel_export_runs_for_the_chosen_set(tmp_path, qtbot) -> None:
    widget, presenter, state, controller, records = _grid_widget(tmp_path, qtbot)
    presenter._export_folder = tmp_path / "out"
    widget.grid_inspection_legend.set_expanded(True)
    controller.panel.open_export_button.click()
    assert controller.panel.export_open()
    assert "Маски" in controller.panel.tree_view.toPlainText()
    controller.panel.run_export_button.click()
    qtbot.waitUntil(lambda: controller._last_report is not None, timeout=10000)
    assert len(list((controller._last_report.set_dir / "Маски").iterdir())) == len(records)
    assert controller.panel.done_label.text()
