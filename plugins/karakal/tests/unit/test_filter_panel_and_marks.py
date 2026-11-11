"""The filter panel (settings under their filters, sliders beside numbers, live hints) and the marks
of the defects for a person (frame window, «errors over the mask»)."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QDoubleSpinBox, QSpinBox

from test_grid_debris_min_area import _cell, _frame_dialog


def test_a_slider_drives_its_number_and_a_drag_sends_one_change(qtbot) -> None:
    from karakal.ui.filter_settings import SpinSlider

    spin = QSpinBox()
    spin.setRange(1, 2000)
    spin.setValue(6)
    pair = SpinSlider(spin, log=True)
    qtbot.addWidget(pair)
    sent: list[int] = []
    spin.valueChanged.connect(sent.append)
    # The log scale keeps small values easy to reach and comes back to the same number.
    for value in (1, 6, 40, 300, 2000):
        assert abs(pair._to_value(pair._to_position(value)) - value) <= max(1, value * 0.01)
    assert pair._to_position(20) > 0.3 * 1000
    pair.slider.setValue(pair._to_position(300))
    assert spin.value() == pair._to_value(pair._to_position(300)) and sent
    sent.clear()
    pair.slider.setSliderDown(True)
    for position in (100, 200, 300):
        pair.slider.setValue(position)
    assert sent == [] and spin.value() == int(round(pair._to_value(300)))
    pair.slider.setSliderDown(False)  # Qt sends sliderReleased: the one change goes out
    assert sent == [spin.value()]
    # A number typed in moves the slider; a box switched off switches its slider off.
    spin.setValue(6)
    assert pair.slider.value() == pair._to_position(6)
    spin.setEnabled(False)
    assert not pair.slider.isEnabled()


def test_choice_buttons_follow_their_number(qtbot) -> None:
    from karakal.ui.filter_settings import ChoiceButtons

    spin = QSpinBox()
    spin.setRange(2, 4)
    spin.setValue(3)
    choice = ChoiceButtons(spin, (2, 3, 4))
    qtbot.addWidget(choice)
    assert choice.buttons[3].isChecked() and not choice.buttons[2].isChecked()
    choice.buttons[4].click()
    assert spin.value() == 4 and choice.buttons[4].isChecked() and not choice.buttons[3].isChecked()


def test_settings_show_under_their_filters_by_one_switch(tmp_path, qtbot) -> None:
    from karakal.app.main_window import KarakalWidget

    ini = str(tmp_path / "k.ini")
    widget = KarakalWidget(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    sections = widget._grid_filter_sections
    assert set(sections) == {"small_artifact", "sparse_cell", "class_conflict", "network_only"}
    assert not widget.grid_filter_settings_check.isChecked()
    assert not any(section.isVisibleTo(widget._grid_inspection_tuning_group) for section in sections.values())
    widget.grid_filter_settings_check.setChecked(True)
    group = widget._grid_inspection_tuning_group
    assert sections["small_artifact"].isVisibleTo(group) and sections["sparse_cell"].isVisibleTo(group)
    # No ones/zeros pair, no pair of one class: their filters and settings stay folded.
    assert not sections["class_conflict"].isVisibleTo(group) and not sections["network_only"].isVisibleTo(group)
    widget.grid_error_type_checks["small_artifact"].setChecked(False)
    assert not sections["small_artifact"].isVisibleTo(group)
    # The settings of a filter sit right under it in the panel.
    layout = group.layout()
    order = [layout.itemAt(index).widget() for index in range(layout.count())]
    assert order.index(sections["sparse_cell"]) == order.index(widget.grid_error_type_checks["sparse_cell"]) + 1
    # The switch is remembered.
    again = KarakalWidget(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(again)
    assert again.grid_filter_settings_check.isChecked()
    # Settings titles carry the mark of their live hint.
    assert widget._grid_debris_min_area_row._title_label.text().endswith("ⓘ")


def test_the_frame_legend_singles_a_type_out_and_steps_through_defects(qtbot) -> None:
    cells = (
        _cell((100, 100, 6, 6), ("small_artifact",), area=37),
        _cell((200, 200, 30, 30), ("broken_geometry",), area=800),
        _cell((40, 300, 20, 40), ("broken_geometry", "sparse_cell")),
        _cell((300, 40, 20, 40), ("sparse_cell",)),
    )
    dialog = _frame_dialog(qtbot, cells)
    dialog._rebuild_grid_mark_items()
    assert dialog._grid_type_counts == {"small_artifact": 1, "broken_geometry": 2, "sparse_cell": 2}
    assert set(dialog._grid_legend_buttons) == {"small_artifact", "broken_geometry", "sparse_cell"}
    assert dialog._grid_legend_buttons["sparse_cell"].text().endswith("· 2")
    assert dialog.grid_defect_position_label.text() == "— / 4"
    # Marked once: no filled picture under the scene items any more.
    assert dialog._grid_mark_items
    # Through the defects, top to bottom: Ctrl+→.
    assert dialog._step_grid_defect(+1)
    assert dialog._grid_defect_index == 0 and dialog._grid_hot_cell is dialog._grid_marked_cells[0]
    assert dialog.grid_defect_position_label.text() == "1 / 4"
    assert dialog._step_grid_defect(-1) and dialog._grid_defect_index == 3
    # A type singled out: only its defects are stepped through.
    dialog._toggle_grid_solo("sparse_cell")
    assert dialog._grid_legend_buttons["sparse_cell"].isChecked()
    assert len(dialog._grid_navigable_cells()) == 2
    assert dialog.grid_defect_position_label.text() == "— / 2"
    dialog._toggle_grid_solo("sparse_cell")
    assert len(dialog._grid_navigable_cells()) == 4


def test_errors_over_the_mask_mark_each_type_by_its_shape() -> None:
    from karakal.core.grid_anomaly import GridFrameAnalysisResult
    from karakal.ui.defect_marks import render_marked_errors
    from karakal.ui.ui_constants import GRID_INSPECTION_ERROR_TYPE_COLORS

    mask = np.zeros((120, 160), np.uint8)
    mask[20:60, 20:40] = 255
    cells = (_cell((20, 20, 20, 40), ("broken_geometry",)), _cell((100, 80, 4, 4), ("small_artifact",)))
    result = GridFrameAnalysisResult(
        frame_id="f", frame_path="", image_width=160, image_height=120, grid_rows=0, grid_cols=0,
        total_expected_cells=2, detected_cells=2, normal_cells=0, suspicious_cells=0, broken_cells=2,
        missing_cells=0, artifact_cells=0, damage_score=0.5, severity_level="HIGH", per_cell_results=cells,
    )
    image = render_marked_errors(mask, result, ("broken_geometry", "small_artifact"))

    def bgr(kind: str) -> list[int]:
        value = GRID_INSPECTION_ERROR_TYPE_COLORS[kind].lstrip("#")
        return [int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16)]

    # The cell keeps its own pixels inside; the outline is around it.
    assert image[40, 30].tolist() == [114, 114, 114]
    assert image[19, 30].tolist() == bgr("broken_geometry")
    # Debris: a ring at a distance, the piece itself left as it is.
    ring = image[60:100, 80:125].reshape(-1, 3).tolist()
    assert bgr("small_artifact") in ring
    assert image[82, 102].tolist() == [0, 0, 0]


def test_every_live_hint_draws_and_tells_what_its_value_does(qtbot) -> None:
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtCore import QRectF

    from karakal.ui.i18n import Translator
    from karakal.ui.live_hints import SCENES

    translate = Translator().tr
    image = QImage(300, 170, QImage.Format.Format_ARGB32)
    values = {
        "debris": {"min_area": 40},
        "sparse": {"reach_x": 3, "reach_y": 2, "free_sides": 3},
        "conflict": {"min_area": 6, "min_share": 10},
        "agreement": {"cover": 30},
    }
    seen: dict[str, set[str]] = {}
    for key, scene in SCENES.items():
        for step in range(0, 40):
            painter = QPainter(image)
            caption_key, args = scene(painter, QRectF(0, 0, 300, 170), step * 0.25, values[key])
            painter.end()
            text = translate(caption_key, **args)
            assert text and "{" not in text
            seen.setdefault(key, set()).add(caption_key)
    # Over time the walking and sliding examples cross their thresholds both ways.
    assert seen["sparse"] == {"live_hint.sparse_yes", "live_hint.sparse_no"}
    assert seen["conflict"] == {"live_hint.conflict_yes", "live_hint.conflict_no"}
    assert seen["agreement"] == {"live_hint.agreement_both", "live_hint.agreement_one"}


def test_a_matrix_tile_knows_the_types_of_its_defects() -> None:
    from karakal.core.grid_anomaly import GridFrameAnalysisResult
    from karakal.ui.matrix_view import _MatrixCellItem

    cells = (
        _cell((0, 0, 5, 5), ("small_artifact",)),
        _cell((10, 10, 5, 5), ("broken_geometry", "sparse_cell")),
        _cell((20, 20, 5, 5), ()),
    )
    result = GridFrameAnalysisResult(
        frame_id="f", frame_path="", image_width=40, image_height=40, grid_rows=0, grid_cols=0,
        total_expected_cells=3, detected_cells=3, normal_cells=1, suspicious_cells=0, broken_cells=2,
        missing_cells=0, artifact_cells=0, damage_score=0.5, severity_level="HIGH", per_cell_results=cells,
    )
    item = _MatrixCellItem.__new__(_MatrixCellItem)
    item.grid_result = result
    assert set(item._defect_types()) == {"small_artifact", "broken_geometry", "sparse_cell"}
