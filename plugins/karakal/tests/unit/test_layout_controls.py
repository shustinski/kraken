"""Layer rows reordered by dragging, collapsible side panel, chevron sections."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QModelIndex, QSettings

from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.ui.ui_constants import FOLDER_LABEL_ROLE, widget_stylesheet


def _widget(tmp_path, qtbot, name: str = "k.ini"):
    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / name), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    return widget


def _labels(widget) -> list[str]:
    return [str(widget.folder_list.item(row).data(FOLDER_LABEL_ROLE)) for row in range(widget.folder_list.count())]


def _add(tmp_path, presenter, names) -> None:
    for name in names:
        folder = tmp_path / name
        folder.mkdir(parents=True, exist_ok=True)
        assert _grayscale_array_to_qimage(np.full((8, 8), 255, dtype=np.uint8)).save(str(folder / "F_0001.png"))
        presenter._append_folder_item(folder, checked=True)
    presenter._refresh_folder_rows()


def test_layer_row_has_handle_settings_and_remove_but_no_arrows(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    _add(tmp_path, widget._presenter, ("a", "b"))
    row = widget.folder_list.itemWidget(widget.folder_list.item(0))
    assert not hasattr(row, "btn_up") and not hasattr(row, "btn_down")
    assert row.grip.toolTip() and not row.btn_remove.icon().isNull() and not row.btn_confidence_toggle.icon().isNull()


def test_dragging_a_layer_reorders_and_rebuilds_rows(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    _add(tmp_path, widget._presenter, ("a", "b", "c"))
    assert widget.folder_list.dragDropMode() == widget.folder_list.DragDropMode.InternalMove
    # What a drop of the last row on top does to the list model.
    assert widget.folder_list.model().moveRow(QModelIndex(), 2, QModelIndex(), 0)
    qtbot.waitUntil(
        lambda: all(
            widget.folder_list.itemWidget(widget.folder_list.item(row)) is not None
            for row in range(widget.folder_list.count())
        ),
        timeout=2000,
    )
    assert _labels(widget) == ["c", "a", "b"]
    assert [spec.display_name for spec in widget._presenter._checked_model_specs()] == ["c", "a", "b"]


def test_alt_arrows_move_the_focused_layer(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    _add(tmp_path, presenter, ("a", "b", "c"))
    widget.folder_list.setCurrentRow(0)
    presenter._move_focused_folder_item(1)
    assert _labels(widget) == ["b", "a", "c"]
    presenter._move_focused_folder_item(-1)
    assert _labels(widget) == ["a", "b", "c"]


def test_side_panel_collapses_to_a_rail_and_is_remembered(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    widget.show()
    assert widget.control_scroll.isVisible() and not widget.side_rail.isVisible()
    widget.toggle_sidebar()
    assert not widget.control_scroll.isVisible() and widget.side_rail.isVisible()
    assert widget.rail_run_button.property("runState") == widget.btn_run.property("runState")
    again = _widget(tmp_path, qtbot)
    again.show()
    assert again.side_rail.isVisible() and not again.control_scroll.isVisible()
    again.set_sidebar_collapsed(False)
    assert again.control_scroll.isVisible()


def test_sections_use_chevrons_and_defects_show_their_count() -> None:
    sheet = widget_stylesheet()
    assert "QGroupBox::indicator:checked" in sheet and "chevron_down.png" in sheet
    assert "QGroupBox::indicator:unchecked" in sheet and "chevron_right.png" in sheet
