"""Memory map tab of the memory work mode: build on request, look, change block roles, export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QSettings

from karakal.core.memory_map import MODE_ONES, MODE_PAIR, Pitch, assemble_memory_map
from karakal.core.memory_map_export import MemoryMapExport, TxtFormat, write_memory_map_files

from test_memory_map import CFG, PX, PY, Memory, frames_of, read_all


@pytest.fixture(autouse=True)
def _no_modal_windows(monkeypatch):
    """A message box in these tests is a failure to look at, not a window on the desktop."""

    from PyQt6.QtWidgets import QMessageBox

    shown = []
    for name in ("warning", "information", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, lambda *args, name=name, **kwargs: shown.append((name, args[2] if len(args) > 2 else "")) or QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(("exec", self.text())) or 0)
    yield shown
    assert not [entry for entry in shown if entry[0] != "exec"], shown


def _widget_with_run(tmp_path: Path, qtbot, *, zeros: bool):
    from karakal.app.main_window import KarakalWidget
    from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
    from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE, FOLDER_FRAMES_PER_ROW_ROLE

    memory = Memory(26, 64, seed=21, gap_col=30)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    folders = {"ones": tmp_path / "ones", "zeros": tmp_path / "zeros"}
    # Frame numbers come after the last underscore of a name.
    names = [f"F_{index:03d}.png" for index in range(len(positions))]
    for name_key, value in (("ones", 1), ("zeros", 0)):
        if name_key == "zeros" and not zeros:
            continue
        folders[name_key].mkdir()
        for name, (x, y) in zip(names, positions):
            cv2.imwrite(str(folders[name_key] / name), memory.render(x, y, value))
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    items = {}
    for name_key in ("ones", "zeros") if zeros else ("ones",):
        item = presenter._append_folder_item(folders[name_key], checked=True)
        item.setData(FOLDER_CHECKED_ROLE, True)
        item.setData(FOLDER_FRAMES_PER_ROW_ROLE, 4)
        presenter._set_folder_item_cell_class(item, name_key)
        items[name_key] = item
    specs = presenter._checked_model_specs()
    ids = {name_key: presenter._model_id_for_folder_item(item) for name_key, item in items.items()}
    records = tuple(
        FrameRecord(
            Path(name).stem,
            name,
            model_mask_paths={ids[key]: str(folders[key] / name) for key in ids},
        )
        for name in names
    )
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(BuildResult(records=records, options=BuildOptions(), model_specs=specs))
    controller = presenter._memory_map_controller
    controller.use_processes = False
    return widget, presenter, controller, memory, positions, records


def _build(qtbot, widget, controller) -> None:
    panel = widget.memory_map_panel
    tabs = widget.grid_inspection_content_tabs
    tabs.setCurrentWidget(panel)
    assert panel.build_button.isEnabled()
    panel.build_button.click()
    qtbot.waitUntil(lambda: controller._thread is None and panel._map is not None, timeout=60000)


def test_the_tab_builds_the_map_of_the_shown_ones_layer(tmp_path, qtbot, monkeypatch) -> None:
    import karakal.core.memory_map as mm

    # Small synthetic frames compare fewer sites at a seam than real ones.
    monkeypatch.setattr(mm, "MemoryMapConfig", lambda: CFG)
    widget, presenter, controller, memory, positions, records = _widget_with_run(tmp_path, qtbot, zeros=False)
    panel = widget.memory_map_panel
    assert widget.grid_inspection_content_tabs.tabText(2) == presenter._t("tab.memory_map")
    assert presenter._t("profile.grid_defects.title") in ("Memory work", "Работа с памятью")
    sources = controller.sources(presenter._current_tab_state())
    assert sources["mode"] == MODE_ONES and sources["per_row"] == 4 and sources["with_masks"] == 12

    _build(qtbot, widget, controller)
    state = presenter._current_tab_state()
    mmap = state.memory_map
    assert mmap is panel._map and mmap.mode == MODE_ONES
    assert {seam.status for seam in mmap.seams} == {"bits"}
    assert panel.blocks_table.rowCount() == len(mmap.blocks) >= 1
    assert panel.export_button.isEnabled()

    # The operator makes block 0 a service block; it shows and stays with the tab state.
    panel.blocks_table.cellWidget(0, 3).setCurrentIndex(1)
    assert state.memory_map.blocks[0].service
    assert list(state.memory_map_roles.values()) == [True]

    # Double click on a site opens its frame centred on the cell.
    opened = []
    monkeypatch.setattr(presenter, "_open_record_details", lambda record, st, grid_focus=None: opened.append((record, grid_focus)))
    row, col = (int(v) for v in np.argwhere(mmap.field == 2)[0])
    panel.frameRequested.emit(row, col)
    record, focus = opened[0]
    index = [r.key for r in records].index(record.key)
    x0, y0, w, h = focus["bbox"]
    x, y = positions[index]
    assert memory.render(x, y, 1)[int(round(y0 + h / 2)), int(round(x0 + w / 2))] == 255


def test_a_ones_zeros_pair_gives_the_full_map(tmp_path, qtbot, monkeypatch) -> None:
    import karakal.core.memory_map as mm

    monkeypatch.setattr(mm, "MemoryMapConfig", lambda: CFG)
    widget, presenter, controller, _memory, _positions, _records = _widget_with_run(tmp_path, qtbot, zeros=True)
    sources = controller.sources(presenter._current_tab_state())
    assert sources["mode"] == MODE_PAIR
    _build(qtbot, widget, controller)
    mmap = presenter._current_tab_state().memory_map
    assert mmap.mode == MODE_PAIR and len(mmap.blocks) == 2  # the gap column is found with both networks
    assert widget.memory_map_panel.legend_checks[3].isVisibleTo(widget.memory_map_panel)  # zeros in the legend


def test_export_writes_the_chosen_files_and_leaves_service_blocks_out(tmp_path) -> None:
    from karakal.core.memory_map import with_block_roles

    memory = Memory(26, 64, seed=22, gap_col=30)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    mmap = assemble_memory_map(read_all(memory, positions), Pitch(PX, PY), per_row=4, config=CFG)
    mmap = with_block_roles(mmap, [1])
    custom = TxtFormat(symbols={"one": "#", "zero": "-", "empty": "?", "conflict": "X", "defect": "!", "partial": "~", "outside": " "}, cell_separator=",")
    written = write_memory_map_files(
        mmap,
        MemoryMapExport(folder=str(tmp_path / "out"), name="m", txt_field=True, txt_blocks=True, txt_custom=True, custom=custom, png=True, csv=True, summary=True, main_only=True),
    )
    assert sorted(p.name for p in written) == ["m.png", "m_blocks.txt", "m_cells.csv", "m_custom.txt", "m_field.txt", "m_summary.json"]
    blocks_text = (tmp_path / "out" / "m_blocks.txt").read_text(encoding="utf-8")
    assert "# block 0" in blocks_text and "# block 1" not in blocks_text  # service block 1 left out
    field_rows = [line for line in (tmp_path / "out" / "m_field.txt").read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    block1 = mmap.blocks[1]
    assert all(line[block1.col : block1.col + block1.cols].strip() == "" for line in field_rows)
    assert "#" in (tmp_path / "out" / "m_custom.txt").read_text(encoding="utf-8")


def test_export_dialog_asks_for_a_folder_and_returns_the_choice(tmp_path, qtbot) -> None:
    from karakal.ui.i18n import Translator
    from karakal.ui.memory_map_export_dialog import MemoryMapExportDialog

    memory = Memory(26, 64, seed=23)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    mmap = assemble_memory_map(read_all(memory, positions, zeros=False), Pitch(PX, PY), per_row=4, mode=MODE_ONES, config=CFG)
    dialog = MemoryMapExportDialog(Translator("en").tr, mmap)
    qtbot.addWidget(dialog)
    dialog._accept()
    assert dialog.result() != dialog.DialogCode.Accepted and "folder" in dialog.error_label.text().lower()
    dialog.folder_edit.setText(str(tmp_path))
    dialog.txt_custom.setChecked(True)
    dialog.symbol_edits["one"].setText("#")
    dialog.separator_edit.setText(" ")
    dialog.main_only.setChecked(True)
    choice = dialog.choice()
    assert choice.txt_custom and choice.custom.symbols["one"] == "#" and choice.custom.cell_separator == " " and choice.main_only
    assert choice.custom.symbols["empty"] == "."  # one network: «not painted»


def test_export_dialog_turns_the_bits_and_previews_them(tmp_path, qtbot) -> None:
    from karakal.ui.i18n import Translator
    from karakal.ui.memory_map_export_dialog import MemoryMapExportDialog

    memory = Memory(26, 64, seed=24, gap_col=30)
    positions = frames_of(memory, frame_rows=3, per_row=4)
    mmap = assemble_memory_map(read_all(memory, positions), Pitch(PX, PY), per_row=4, config=CFG)
    dialog = MemoryMapExportDialog(Translator("en").tr, mmap)
    qtbot.addWidget(dialog)
    body = [line for line in dialog.preview.toPlainText().splitlines() if not line.startswith("#")]
    symbols = {2: "1", 3: "0", 1: "?", 0: " ", 5: "!", 4: "X"}
    assert body[0].rstrip() == "".join(symbols[int(v)] for v in mmap.field[0][:120]).rstrip()

    dialog.field_orientation.setCurrentIndex(dialog.field_orientation.findData("flip_h"))
    dialog.number_from.setCurrentIndex(1)
    dialog._update_preview()
    body = [line for line in dialog.preview.toPlainText().splitlines() if not line.startswith("#")]
    assert body[0].rstrip() == "".join(symbols[int(v)] for v in mmap.field[0][::-1][:120]).rstrip()
    choice = dialog.choice()
    assert choice.field_orientation == "flip_h" and choice.number_from == 1 and choice.block_orientation == "none"
    assert "numbered 1" in dialog.preview.toPlainText()


def test_the_map_takes_the_defects_of_the_cell_analysis(tmp_path, qtbot, monkeypatch) -> None:
    import cv2 as _cv2

    from karakal.core.grid_anomaly import analyze_grid_frame_path
    from karakal.core.synthetic_masks import synthetic_frame

    widget, presenter, controller, _memory, _positions, records = _widget_with_run(tmp_path, qtbot, zeros=False)
    state = presenter._current_tab_state()
    model_id = controller.sources(state)["ones_id"]
    frame = synthetic_frame({"cut_corner": 3}, seed=5)
    path = tmp_path / "bitten.png"
    _cv2.imwrite(str(path), frame.mask)
    result = analyze_grid_frame_path(path, frame_id="x", use_cache=False, read_cache=False, write_cache=False)
    broken = [cell for cell in result.per_cell_results if cell.status == "broken"]
    assert broken
    key = str(records[0].key)
    state.grid_inspection_payloads_by_model = {model_id: {"binary": {key: result}}}
    sources = controller.sources(state)
    boxes = sources["frames"][0].defect_boxes
    assert {tuple(int(v) for v in cell.bbox) for cell in broken} <= set(boxes)
    assert sources["frames"][1].defect_boxes == ()


def test_full_memory_analysis_runs_the_defect_search_then_builds_and_shows_the_map(tmp_path, qtbot, monkeypatch) -> None:
    import karakal.core.memory_map as mm
    from karakal.ui.ui_constants import FOLDER_FRAMES_PER_ROW_ROLE

    monkeypatch.setattr(mm, "MemoryMapConfig", lambda: CFG)
    widget, presenter, controller, _memory, _positions, _records = _widget_with_run(tmp_path, qtbot, zeros=True)
    # No frames per row on the layers: the map finds them from the bits.
    for row in range(presenter.folder_list.count()):
        presenter.folder_list.item(row).setData(FOLDER_FRAMES_PER_ROW_ROLE, 0)
    widget.grid_inspection_content_tabs.setCurrentIndex(0)
    state = presenter._current_tab_state()
    assert not state.grid_inspection_results_ready
    controller.refresh()
    assert widget.memory_map_panel.per_row_spin.value() == 0  # «auto»

    widget.memory_map_panel.full_button.click()
    qtbot.waitUntil(lambda: state.memory_map is not None and controller._thread is None, timeout=120000)
    assert state.grid_inspection_results_ready  # the defect search ran first
    assert state.grid_inspection_payloads_by_model  # both layers analysed
    assert state.memory_map.per_row == 4 and state.memory_map.mode == "pair"
    assert widget.grid_inspection_content_tabs.currentWidget() is widget.memory_map_panel
    assert "4" in widget.memory_map_panel.status_label.text()

    # With an up-to-date analysis the button builds the map at once.
    state.memory_map = None
    widget.memory_map_panel.full_button.click()
    qtbot.waitUntil(lambda: state.memory_map is not None and controller._thread is None, timeout=60000)
