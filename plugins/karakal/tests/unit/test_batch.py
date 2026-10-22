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


@pytest.mark.parametrize("fmt", ["jpg", "png"])
def test_compute_and_export_writes_error_maps_and_a_project(tmp_path, qtbot, fmt) -> None:
    from karakal.app.main_window import KarakalWidget

    snapshot = _snapshot(tmp_path / "snapshot_4")
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    controller = presenter._batch_controller
    page = widget.batch_page
    layout = controller.choose_snapshot(str(snapshot))
    assert page.layers_table.rowCount() == 2 and page.run_button.isEnabled()
    page.format_combo.setCurrentIndex(page.format_combo.findData(fmt))
    page.set_output_dir(str(tmp_path / "out"))
    finished = []
    controller.finished.connect(finished.append)
    page.run_button.click()
    qtbot.waitUntil(lambda: bool(finished) and not controller.is_running(), timeout=120000)
    report = finished[0]
    assert report.frames == 3 and report.missing == ()
    errors = report.output_dir / "Весь прогон" / "Ошибки"
    for layer in layout.layers:
        written = sorted(path.name for path in (errors / layer.label).iterdir())
        assert len(written) == 3 and all(name.endswith(f".{fmt}") for name in written)
    assert page.open_folder_button.isVisibleTo(page) and page.open_project_button.isVisibleTo(page)

    # «Open in the matrix»: the run at once, coloured, no recomputing.
    presenter._start_compute_grid_inspection = lambda *_args, **_kwargs: pytest.fail("recomputed")
    page.open_project_button.click()
    state = presenter._current_tab_state()
    assert state is not None and state.grid_inspection_results_ready
    assert len(state.build_result.records) == 3
    # Ones TEST_00001 and zeros TEST_inv_00001 are one frame, each with its own confidence map.
    for record in state.build_result.records:
        assert len(record.model_mask_paths) == 2 and all(record.model_prob_paths.values())
        assert any("_inv_" in path for path in record.model_prob_paths.values())
    assert len(presenter._grid_inspection_views()["unified"]._grid_inspection_payload_by_key) == 3
    assert presenter._project_controller.recent()[0] == report.project


def _page_run(tmp_path, qtbot, snapshot):
    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    controller = widget._presenter._batch_controller
    controller.choose_snapshot(str(snapshot))
    widget.batch_page.set_output_dir(str(tmp_path / "out"))
    finished = []
    controller.finished.connect(finished.append)
    return widget, controller, finished


def test_the_page_shows_every_stage_the_total_and_frames_with_problems(tmp_path, qtbot) -> None:
    snapshot = _snapshot(tmp_path / "snap")
    # The zeros network has no mask of frame 3: a frame with a problem.
    (snapshot / "binary_cif_inv" / "TEST_inv_00003.png").unlink()
    widget, controller, finished = _page_run(tmp_path, qtbot, snapshot)
    page = widget.batch_page
    page.run_button.click()
    assert page.run_button.property("runState") == "stop" and page.pause_button.isVisibleTo(page)
    qtbot.waitUntil(lambda: bool(finished) and not controller.is_running(), timeout=120000)
    qtbot.wait(50)
    titles = [row[1].text() for row in page._stage_rows.values()]
    assert titles[0] == widget._t("batch.stage_title.index")
    assert any("binary_cif_inv" in title for title in titles)
    assert all(row[0].text() == "✓" for row in page._stage_rows.values())
    assert page.total_label.text().startswith(widget._t("batch.total_done", elapsed="")[:6])
    assert page.run_button.property("runState") == "go" and not page.pause_button.isVisibleTo(page)
    # Frame 3: no zeros mask.
    assert page.problems_button.isVisibleTo(page)
    assert page.problems_button.text() == widget._t("batch.problems", count=1)
    page.problems_button.click()
    assert page.problems_list.count() == 1 and "TEST_00003" in page.problems_list.item(0).text()
    assert finished[0].problems and finished[0].problems[0][1] == "binary_cif_inv"


def test_pause_holds_the_run_and_resume_finishes_it(tmp_path, qtbot) -> None:
    widget, controller, finished = _page_run(tmp_path, qtbot, _snapshot(tmp_path / "snap"))
    page = widget.batch_page
    page.run_button.click()
    page.pause_button.click()
    assert controller._thread.paused and page.pause_button.text() == widget._t("run.resume")
    qtbot.wait(400)
    assert not finished
    page.pause_button.click()
    assert not controller._thread.paused
    qtbot.waitUntil(lambda: bool(finished) and not controller.is_running(), timeout=120000)
    assert finished[0].frames == 3
