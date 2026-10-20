"""Dirty regions in the app: run from the cell check settings, marks on the matrix, the frame window
layer and the frame set export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import pytest
from PyQt6.QtCore import QRectF, QSettings
from PyQt6.QtWidgets import QGraphicsScene

import karakal.core.dirt_batch as dirt_batch
from karakal.core.dirt_regions import DirtConfig, analyze_dirt_frame

from test_dirt_regions import lattice, smear, sure


@pytest.fixture(autouse=True)
def _own_dirt_cache(tmp_path_factory, monkeypatch):
    """Dirt found by the tests is cached in a folder of the test run, not in the user's cache."""

    folder = tmp_path_factory.mktemp("dirt_cache")
    monkeypatch.setattr(dirt_batch, "default_cache_dir", lambda: folder)


def _write(path: Path, image) -> str:
    cv2.imencode(".png", image)[1].tofile(str(path))
    return str(path)


def _widget_with_run(tmp_path: Path, qtbot):
    from karakal.app.main_window import KarakalWidget
    from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
    from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE

    folder = tmp_path / "zeros"
    folder.mkdir()
    dirty = smear(lattice(), 380, 350, 70, 90)
    clean = lattice()
    paths = {}
    for name, mask in (("F_001", dirty), ("F_002", clean)):
        paths[name] = (_write(folder / f"{name}.png", mask), _write(folder / f"{name}_confidence.png", sure(mask)))
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    item = presenter._append_folder_item(folder, checked=True)
    item.setData(FOLDER_CHECKED_ROLE, True)
    specs = presenter._checked_model_specs()
    model_id = presenter._model_id_for_folder_item(item)
    records = tuple(
        FrameRecord(name, f"{name}.png", model_mask_paths={model_id: mask}, model_prob_paths={model_id: confidence})
        for name, (mask, confidence) in paths.items()
    )
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(BuildResult(records=records, options=BuildOptions(), model_specs=specs))
    # The settings sit in a collapsible group; collapsed, everything in it is switched off.
    widget.analysis_settings_group.setChecked(True)
    controller = presenter._dirt_controller
    controller.use_processes = False
    return widget, presenter, controller, model_id


def test_find_dirt_marks_the_dirty_frame(tmp_path, qtbot) -> None:
    widget, presenter, controller, model_id = _widget_with_run(tmp_path, qtbot)
    controls = widget.dirt_controls
    assert controls.status.text() == widget._t("dirt.summary_empty")
    controls.run_button.click()
    qtbot.waitUntil(lambda: controller.has_results() and not controller.is_running(), timeout=30000)

    results = controller.results_for(model_id)
    assert set(results) == {"F_001", "F_002"}
    assert len(results["F_001"].regions) == 1 and results["F_001"].used_confidence
    assert results["F_002"].regions == ()
    assert "1" in controls.status.text() and "2" in controls.status.text()
    assert set(controller.frame_marks(model_id)) == {"F_001"}
    marked = [getattr(view, "_dirt_marks", {}) for view in presenter._grid_inspection_views().values()]
    assert marked and all(set(marks) == {"F_001"} for marks in marked)


def test_dirt_after_the_cell_check_only_when_asked(tmp_path, qtbot) -> None:
    widget, _presenter, controller, _model_id = _widget_with_run(tmp_path, qtbot)
    widget.dirt_controls.enabled_check.setChecked(False)
    controller.analysis_finished()
    assert not controller.is_running() and not controller.has_results()
    widget.dirt_controls.enabled_check.setChecked(True)
    controller.analysis_finished()
    qtbot.waitUntil(lambda: controller.has_results() and not controller.is_running(), timeout=30000)


def test_dirt_settings_are_kept_with_the_build_settings(tmp_path, qtbot) -> None:
    widget, presenter, controller, _model_id = _widget_with_run(tmp_path, qtbot)
    controls = widget.dirt_controls
    controls.merge_gap_spin.setValue(5.0)
    controls.sensitivity_spin.setValue(1.5)
    payload = presenter._build_build_settings_payload()["dirt_settings"]
    assert payload == {"enabled": True, "merge_gap_cells": 5.0, "sensitivity": 1.5, "filter_level": "light"}
    assert controller.config() == DirtConfig(merge_gap_cells=5.0, sensitivity=1.5)
    controls.apply_settings({"enabled": False, "merge_gap_cells": 1.0, "sensitivity": 2.0})
    assert not controls.enabled() and controller.config().merge_gap_cells == 1.0


def _result(tmp_path: Path):
    mask = smear(smear(lattice(), 240, 350, 55, 80, 1), 520, 350, 55, 80, 2)
    mask_path = _write(tmp_path / "F_001.png", mask)
    confidence_path = _write(tmp_path / "F_001_confidence.png", sure(mask))
    return analyze_dirt_frame(mask_path, frame_id="F_001", confidence_path=confidence_path)


def test_frame_window_layer_draws_box_outline_and_label(tmp_path, qtbot) -> None:
    from karakal.ui.dirt_overlay import DirtOverlay
    from karakal.ui.i18n import Translator

    result = _result(tmp_path)
    (region,) = result.regions
    scene = QGraphicsScene()
    # The window shows the frame at twice the mask size: the layer follows the scene.
    scene.setSceneRect(QRectF(0, 0, 2 * result.width, 2 * result.height))
    translate = Translator("ru").tr
    overlay = DirtOverlay(scene, translate)
    qtbot.addWidget(overlay.widget)
    overlay.set_result(result)
    items = overlay.items()
    assert len(items) == 3
    box = items[0]
    x, y, w, h = region.bbox
    assert box.rect() == QRectF(2 * x, 2 * y, 2 * w, 2 * h)
    assert "#1" in box.toolTip() and f"x {x}, y {y}" in box.toolTip()
    assert "Областей: 1" in overlay.status.text()

    overlay.box_check.setChecked(False)
    assert len(overlay.items()) == 2
    overlay.heat_check.setChecked(True)
    assert not overlay._heat_item.pixmap().isNull() and overlay._heat_item.isVisible()
    overlay.set_active(False)
    assert overlay.items() == () and not overlay._heat_item.isVisible()
    overlay.set_active(True)
    overlay.set_result(None, known=False)
    assert overlay.items() == () and overlay.status.text() == translate("dirt.frame_unknown")


def test_frame_set_export_writes_the_dirt_folder(tmp_path) -> None:
    from karakal.core.frame_set_export import (
        ITEM_DIRT,
        FrameSetExportFrame,
        FrameSetExportModel,
        FrameSetExportPlan,
        export_frame_set,
        preview_frame_set_export,
    )

    result = _result(tmp_path)
    plan = FrameSetExportPlan(
        set_name="Выборка",
        layer_title="Нули",
        frames=(FrameSetExportFrame(key="F_001", name="F_001.png"), FrameSetExportFrame(key="F_009", name="F_009.png")),
        models=(FrameSetExportModel("zeros", "Нули"),),
        items=frozenset({ITEM_DIRT}),
        dirt_results={"zeros": {"F_001": result}},
    )
    preview = "\n".join(preview_frame_set_export(plan, tmp_path / "out").lines)
    assert "Грязь" in preview and "dirt_regions.csv" in preview
    report = export_frame_set(plan, tmp_path / "out", run_name="run")
    folder = report.set_dir / "Грязь"
    assert (folder / "F_001.dirt.json").is_file() and (folder / "F_001.dirt.png").is_file()
    assert (folder / "F_001.dirt_view.jpg").is_file()
    assert report.missing == ("F_009.png",)
    payload = json.loads((folder / "F_001.dirt.json").read_text(encoding="utf-8"))
    assert payload["layer"] == "Нули" and len(payload["regions"]) == 1
    with (folder / "dirt_regions.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["frame"], row["id"]) for row in rows] == [("F_001", "1")]


def test_frame_window_shows_the_dirt_of_its_frame(tmp_path, qtbot) -> None:
    widget, presenter, controller, _model_id = _widget_with_run(tmp_path, qtbot)
    widget.dirt_controls.run_button.click()
    qtbot.waitUntil(lambda: controller.has_results() and not controller.is_running(), timeout=30000)
    state = presenter._current_tab_state()
    presenter._open_record_details(state.build_result.records[0], state)
    dialog = presenter._details_dialogs[-1]
    qtbot.addWidget(dialog)
    # The window turns to the cell defects view once its frame is loaded in the background.
    qtbot.waitUntil(lambda: dialog._selected_result_kind() == "grid_cell_defects", timeout=30000)
    overlay = dialog.dirt_overlay
    assert not overlay.widget.isHidden()
    assert overlay.result is not None and overlay.result.frame_id == "F_001"
    assert len(overlay.items()) == 3
    # Another frame in the same window: its own dirt (none here).
    dialog.set_dirt_result(controller.result_for(_model_id, "F_002"))
    assert overlay.items() == () and overlay.status.text() == widget._t("dirt.frame_none")


def test_dirt_rule_of_the_frame_set_model() -> None:
    from karakal.core.frame_sets import RULE_DIRT, VIEW_DROPPED, FrameSetModel

    model = FrameSetModel(dirt_levels={"A": "heavy", "B": "medium", "C": "light"})
    keys = {"A", "B", "C", "D"}
    qualities = {key: 0.9 for key in keys}
    assert model.dirty_keys("medium") == {"A", "B"}
    rule = model.add_dirt_rule("medium", "Нули")
    assert rule.kind == RULE_DIRT and model.has_filter()
    assert model.kept_keys(keys, qualities) == {"C", "D"}
    assert model.set_keys(VIEW_DROPPED, keys, qualities) == {"A", "B"}
    # Inverted, only the dirty frames take part.
    model.inverted = True
    assert model.kept_keys(keys, qualities) == {"A", "B"}
    model.inverted = False
    model.set_rule_enabled(rule.rule_id, False)
    assert model.kept_keys(keys, qualities) == keys
    model.set_rule_enabled(rule.rule_id, True)
    # The rule follows the dirt found last.
    model.dirt_levels = {"D": "heavy"}
    assert model.kept_keys(keys, qualities) == {"A", "B", "C"}
    model.remove_rule(rule.rule_id)
    assert not model.has_filter()


def test_filter_list_and_drop_find_the_dirty_frames(tmp_path, qtbot) -> None:
    widget, presenter, controller, _model_id = _widget_with_run(tmp_path, qtbot)
    controls = widget.dirt_controls
    assert not controls.frames_list.isVisibleTo(widget)
    controls.run_button.click()
    qtbot.waitUntil(lambda: controller.has_results() and not controller.is_running(), timeout=30000)

    # The list: only the dirty frame, with its level and dirtiness.
    assert controls.frames_list.count() == 1
    item = controls.frames_list.item(0)
    assert item.text().startswith("F_001 — ")
    views = presenter._grid_inspection_views().values()
    assert all(getattr(view, "_dirt_filter_keys", None) is None for view in views)

    # The matrix filter fades every other frame.
    controls.filter_check.setChecked(True)
    assert all(view._dirt_filter_keys == {"F_001"} for view in views)
    level = controller.frame_levels()["F_001"]
    stricter = {"light": "medium", "medium": "heavy"}.get(level)
    if stricter is not None:
        controls.level_combo.setCurrentIndex(controls.level_combo.findData(stricter))
        assert all(view._dirt_filter_keys == set() for view in views) and controls.frames_list.count() == 0
        controls.level_combo.setCurrentIndex(controls.level_combo.findData("light"))
    controls.filter_check.setChecked(False)
    assert all(view._dirt_filter_keys is None for view in views)

    # Next / click: the frame is chosen on the matrix.
    chosen = []
    controls.frameChosen.connect(chosen.append)
    controls.next_button.click()
    assert chosen == ["F_001"]
    assert str(presenter.grid_inspection_matrix_view.current_record().key) == "F_001"

    # The drop panel offers «Drop dirty» and the rule drops the dirty frame.
    frame_sets = presenter._frame_sets_controller
    frame_sets.refresh()
    state = presenter._current_tab_state()
    model = state.frame_sets
    assert model.dirt_levels == {"F_001": level}
    frame_sets.panel.dirtDropRequested.emit("light")
    assert any(rule.kind == "dirt" for rule in model.rules)
    assert model.kept_keys({"F_001", "F_002"}, {}) == {"F_002"}
