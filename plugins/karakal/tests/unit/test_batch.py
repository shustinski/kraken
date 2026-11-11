"""«Compute and export»: layers found in a snapshot folder, cells checked and error maps written
without the matrix, the run saved as a project."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from PyQt6.QtCore import QSettings

from karakal.core.snapshot_layout import discover_snapshot

from test_dirt_regions import lattice, smear, sure


def _write(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, image)[1].tofile(str(path))


def _snapshot(root: Path, *, pair: bool = True, frames: int = 3) -> Path:
    for index in range(1, frames + 1):
        name = f"TEST_{index:05d}"
        mask = smear(lattice(), 380, 350, 70, 90) if index == 1 else lattice()
        _write(root / "binary_cif" / f"{name}.png", mask)
        _write(root / "confidence" / f"{name}_confidence.png", sure(mask))
        _write(root / "jpg" / f"{name}.jpg", lattice())
        if pair:
            inverted = 255 - lattice()
            _write(root / "binary_cif_inv" / f"TEST_inv_{index:05d}.png", inverted)
            _write(root / "confidence_inv" / f"TEST_inv_{index:05d}_confidence.png", sure(inverted))
    return root


def test_snapshot_layout_finds_the_pair_its_confidence_and_sources(tmp_path) -> None:
    layout = discover_snapshot(_snapshot(tmp_path / "snap"))
    assert [layer.label for layer in layout.layers] == ["binary_cif", "binary_cif_inv"]
    ones, zeros = layout.layers
    assert (ones.cell_class, zeros.cell_class) == ("ones", "zeros") and layout.is_pair
    assert ones.confidence_folder.name == "confidence" and zeros.confidence_folder.name == "confidence_inv"
    assert ones.source_folder.name == "jpg" and zeros.source_folder.name == "jpg"
    assert (ones.mask_count, ones.confidence_count) == (3, 3)
    assert layout.notes == ()


def test_snapshot_layout_single_network_and_shared_confidence(tmp_path) -> None:
    single = discover_snapshot(_snapshot(tmp_path / "single", pair=False))
    assert len(single.layers) == 1 and single.layers[0].cell_class == "" and not single.is_pair
    # Both networks' confidence maps in one folder: the _inv masks take it too.
    shared = tmp_path / "shared"
    _snapshot(shared)
    for item in (shared / "confidence_inv").iterdir():
        item.rename(shared / "confidence" / item.name)
    (shared / "confidence_inv").rmdir()
    zeros = discover_snapshot(shared).layers[1]
    assert zeros.confidence_folder.name == "confidence"
    # Images straight in the folder: one layer; no confidence noted.
    flat = tmp_path / "flat"
    _write(flat / "TEST_00001.png", lattice())
    layout = discover_snapshot(flat)
    assert len(layout.layers) == 1 and layout.layers[0].mask_folder == flat
    assert discover_snapshot(tmp_path / "nothing").notes == ("missing",)




def _home(tmp_path, qtbot, settings: str = "k"):
    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / f"{settings}.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    return widget, widget._presenter._batch_controller, widget.home_page


def _run_home(qtbot, controller, page) -> list:
    finished = []
    controller.finished.connect(finished.append)
    page.run_button.click()
    qtbot.waitUntil(lambda: bool(finished) and not controller.is_running(), timeout=120000)
    return finished


@pytest.mark.parametrize("fmt", ["jpg", "png"])
def test_compute_and_export_from_the_home_page_and_open_the_matrix(tmp_path, qtbot, fmt) -> None:
    from karakal.core.recent_selections import MATRIX_NONE, MATRIX_READY, RUN_DONE

    snapshot = _snapshot(tmp_path / "snapshot_4")
    widget, controller, page = _home(tmp_path, qtbot)
    presenter = widget._presenter
    # Dropped on the page: the layers are read, the run can start.
    page.drop_zone.folderDropped.emit(str(snapshot))
    layout = controller._layout
    assert page.snapshot_edit.text() == str(snapshot) and page.layers_table.rowCount() == 2
    assert page.run_button.isEnabled() and not page.layers_table.isVisibleTo(page)
    assert widget._t("batch.class.ones") in page.summary.text()
    page.details_button.click()
    assert page.layers_table.isVisibleTo(page)
    assert page.matrix_button.state == MATRIX_NONE
    page.format_buttons[fmt].click()
    page.set_output_dir(str(tmp_path / "out"))
    finished = _run_home(qtbot, controller, page)
    report = finished[0]
    assert report.frames == 3 and report.missing == ()
    errors = report.output_dir / "Весь прогон" / "Ошибки" / "Геометрия"
    for layer in layout.layers:
        written = sorted(path.name for path in (errors / layer.label).iterdir())
        assert len(written) == 3 and all(name.endswith(f".{fmt}") for name in written)
    assert page.open_folder_button.isVisibleTo(page)
    qtbot.wait(50)
    # The matrix button: green, what the run found, opens the project at once.
    assert page.matrix_button.state == MATRIX_READY and page.matrix_button.share == 1.0
    assert any(widget._t("home.m.frames", count=3) in line.text() for line in page.matrix_button.lines)
    selection = controller.current()
    assert selection.run == RUN_DONE and selection.errors_format == fmt and selection.defects >= 1
    assert page.selections.count() == 1 and page.selections.item(0).text().startswith("snapshot_4")
    presenter._start_compute_grid_inspection = lambda *_args, **_kwargs: pytest.fail("recomputed")
    page.matrix_button.click()
    assert widget.root_stack.currentWidget() is widget.workspace
    state = presenter._current_tab_state()
    assert state is not None and state.grid_inspection_results_ready
    assert len(state.build_result.records) == 3
    # Ones TEST_00001 and zeros TEST_inv_00001 are one frame, each with its own confidence map.
    for record in state.build_result.records:
        assert len(record.model_mask_paths) == 2 and all(record.model_prob_paths.values())
        assert any("_inv_" in path for path in record.model_prob_paths.values())
    assert len(presenter._grid_inspection_views()["unified"]._grid_inspection_payload_by_key) == 3
    assert presenter._project_controller.recent()[0] == report.project


def test_the_run_fills_its_button_and_the_matrix_button_tells_the_stage(tmp_path, qtbot) -> None:
    from karakal.core.recent_selections import MATRIX_RUNNING

    snapshot = _snapshot(tmp_path / "snap")
    # The zeros network has no mask of frame 3: a frame with a problem.
    (snapshot / "binary_cif_inv" / "TEST_inv_00003.png").unlink()
    widget, controller, page = _home(tmp_path, qtbot)
    controller.choose_snapshot(str(snapshot))
    page.set_output_dir(str(tmp_path / "out"))
    seen = []
    page.run_button.click()
    assert page.run_button.property("runState") == "stop" and page.pause_button.isVisibleTo(page)
    assert page.matrix_button.state == MATRIX_RUNNING
    original = page.set_progress

    def watch(view) -> None:
        original(view)
        seen.append((page.run_button.styleSheet(), [line.text() for line in page.matrix_button.lines]))

    controller._thread.progressed.disconnect()
    controller._thread.progressed.connect(lambda view: (controller.__setattr__("_last_view", view), watch(view)))
    finished = []
    controller.finished.connect(finished.append)
    qtbot.waitUntil(lambda: bool(finished) and not controller.is_running(), timeout=120000)
    assert seen
    # Part way the run button fills from the left; the matrix button tells the stage.
    assert any("qlineargradient" in style for style, _lines in seen)
    stage_line = widget._t("home.m.stage", index=1, count=99, title="")[:4]
    assert any(lines and lines[0].startswith(stage_line) for _style, lines in seen)
    assert page.run_button.property("runState") == "go" and not page.pause_button.isVisibleTo(page)
    # Frame 3: no zeros mask.
    assert page.problems_button.isVisibleTo(page)
    assert page.problems_button.text() == widget._t("batch.problems", count=1)
    page.problems_button.click()
    assert page.problems_list.count() == 1 and "TEST_00003" in page.problems_list.item(0).text()
    assert finished[0].problems and finished[0].problems[0][1] == "binary_cif_inv"
    assert controller.current().problems == 1
    assert any(widget._t("home.m.problems", count=1) == line.text() for line in page.matrix_button.lines)


def test_pause_holds_the_run_and_resume_finishes_it(tmp_path, qtbot) -> None:
    widget, controller, page = _home(tmp_path, qtbot)
    controller.choose_snapshot(str(_snapshot(tmp_path / "snap")))
    page.set_output_dir(str(tmp_path / "out"))
    finished = []
    controller.finished.connect(finished.append)
    page.run_button.click()
    page.pause_button.click()
    assert controller._thread.paused and page.pause_button.text() == widget._t("run.resume")
    qtbot.wait(400)
    assert not finished
    page.pause_button.click()
    assert not controller._thread.paused
    qtbot.waitUntil(lambda: bool(finished) and not controller.is_running(), timeout=120000)
    assert finished[0].frames == 3


def test_a_stopped_run_is_part_computed_and_the_matrix_computes_the_rest(tmp_path, qtbot) -> None:
    from karakal.core.recent_selections import MATRIX_PARTIAL, RUN_STOPPED

    widget, controller, page = _home(tmp_path, qtbot)
    controller.choose_snapshot(str(_snapshot(tmp_path / "snap", frames=4)))
    page.set_output_dir(str(tmp_path / "out"))
    page.run_button.click()
    qtbot.waitUntil(lambda: controller._last_view is not None and controller._last_view.share > 0.05, timeout=60000)
    page.pause_button.click()
    page.run_button.click()
    qtbot.waitUntil(lambda: not controller.is_running(), timeout=60000)
    qtbot.wait(50)
    selection = controller.current()
    assert selection.run == RUN_STOPPED and 0 < selection.share < 1 and selection.stage
    assert page.matrix_button.state == MATRIX_PARTIAL
    assert page.status.text() == widget._t("batch.stopped")
    started = []
    presenter = widget._presenter
    presenter._on_primary_run_requested = lambda: started.append(True)
    page.matrix_button.click()
    assert started and widget.root_stack.currentWidget() is widget.workspace
    assert presenter.folder_list.count() == 2


def test_recent_selections_put_their_paths_back_and_stay_after_a_restart(tmp_path, qtbot) -> None:
    first = _snapshot(tmp_path / "first", frames=1)
    second = _snapshot(tmp_path / "second", pair=False, frames=1)
    widget, controller, page = _home(tmp_path, qtbot)
    controller.choose_snapshot(str(first))
    page.format_buttons["jpg"].click()
    page.set_output_dir(str(tmp_path / "maps"))
    page.output_edit.editingFinished.emit()
    # A path typed in: Enter reads it.
    page.snapshot_edit.setText(str(second))
    page.snapshot_edit.returnPressed.emit()
    assert controller._layout.root == second and page.errors_format() == "jpg"
    # A new snapshot exports into its project's «Выгрузки» unless a folder is chosen.
    assert page.output_dir() == ""
    assert [page.selections.item(row).text().splitlines()[0] for row in range(page.selections.count())] == ["second", "first"]
    page.selections.itemClicked.emit(page.selections.item(1))
    assert page.snapshot_dir() == str(first) and page.errors_format() == "jpg"
    assert page.output_dir() == str(tmp_path / "maps")
    assert page.selections.currentRow() == 0
    # After a restart the list is there; removing one forgets it.
    _other, again, again_page = _home(tmp_path, qtbot)
    assert [selection.name for selection in again.selections()] == ["first", "second"]
    again_page.selections.setCurrentRow(1)
    again_page.forget_button.click()
    assert [selection.name for selection in again.selections()] == ["first"]


def test_the_matrix_button_with_no_snapshot_opens_an_empty_matrix(tmp_path, qtbot) -> None:
    from karakal.core.recent_selections import MATRIX_EMPTY

    widget, controller, page = _home(tmp_path, qtbot)
    assert page.matrix_button.state == MATRIX_EMPTY and not page.run_button.isEnabled()
    page.matrix_button.click()
    assert widget.root_stack.currentWidget() is widget.workspace



def test_without_a_chosen_folder_the_maps_go_into_the_project(tmp_path, qtbot) -> None:
    from karakal.core.data_folder import EXPORTS

    snapshot = _snapshot(tmp_path / "snap", pair=False, frames=1)
    widget, controller, page = _home(tmp_path, qtbot)
    controller.choose_snapshot(str(snapshot))
    assert page.output_dir() == ""
    report = _run_home(qtbot, controller, page)[0]
    assert report.project is not None and report.output_dir is not None
    assert report.output_dir.parent == report.project / EXPORTS
    assert any((report.output_dir).rglob("*.png"))


def test_an_unrecognised_folder_is_told_by_hand_and_remembered(tmp_path, qtbot) -> None:
    from karakal.core.snapshot_layout import KIND_MASKS, KIND_SKIP, KIND_UNKNOWN, MANIFEST_FILE

    snapshot = _snapshot(tmp_path / "snap", pair=False, frames=1)
    (snapshot / "binary_cif").rename(snapshot / "сеть_v2")
    _write(snapshot / "notes" / "x.png", lattice())
    layout = discover_snapshot(snapshot)
    assert set(layout.unrecognized) == {"notes", "сеть_v2"} and not layout.ready
    widget, controller, page = _home(tmp_path, qtbot)
    controller.choose_snapshot(str(snapshot))
    # The run waits; the table of folders is open by itself.
    assert not page.run_button.isEnabled() and page.folders_table.isVisibleTo(page)
    controller.set_folder("сеть_v2", KIND_MASKS, "Единицы", "ones")
    controller.set_folder("notes", KIND_SKIP, "", "")
    assert page.run_button.isEnabled() and controller._layout.ready
    ones = controller._layout.layers[0]
    assert (ones.label, ones.cell_class, ones.mask_folder.name) == ("Единицы", "ones", "сеть_v2")
    assert ones.confidence_folder.name == "confidence" and ones.source_folder.name == "jpg"
    # Next time the snapshot is recognised at once.
    assert (snapshot / MANIFEST_FILE).is_file()
    again = discover_snapshot(snapshot)
    assert again.ready and again.from_manifest and again.layers[0].label == "Единицы"
    assert {folder.name: folder.kind for folder in again.folders}["notes"] == KIND_SKIP
    assert KIND_UNKNOWN not in {folder.kind for folder in again.folders}


def test_folders_named_after_the_cell_class_make_a_pair(tmp_path) -> None:
    root = tmp_path / "snap"
    _write(root / "ones" / "A.png", lattice())
    _write(root / "zeros" / "A.png", 255 - lattice())
    layout = discover_snapshot(root)
    assert layout.ready and layout.is_pair
    assert [(layer.label, layer.cell_class) for layer in layout.layers] == [("ones", "ones"), ("zeros", "zeros")]
