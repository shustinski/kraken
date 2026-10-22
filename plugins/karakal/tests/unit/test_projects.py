"""Projects: a computed run saved and opened again at once, also after its folders moved."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import pytest
from PyQt6.QtCore import QSettings

from karakal.core.domain import BuildResult, FrameRecord, ModelSpec
from karakal.core.project_store import (
    PROJECT_SUFFIX,
    find_projects,
    project_folder,
    read_manifest,
    read_results,
    relocate_state,
    resolve_folder,
    stable_model_id,
    write_manifest,
    write_results,
)

from test_dirt_regions import lattice, smear, sure


def test_store_writes_reads_and_numbers_projects(tmp_path) -> None:
    first = project_folder(tmp_path, 'Нули: прогон "1"')
    assert first.name == f"Нули прогон 1{PROJECT_SUFFIX}"
    write_manifest(first, {"name": "x", "frames": 3})
    write_results(first, {"answer": [1, 2, 3]})
    assert read_manifest(first)["frames"] == 3 and read_results(first) == {"answer": [1, 2, 3]}
    second = project_folder(tmp_path, 'Нули: прогон "1"')
    assert second != first and second.name.endswith(f"(2){PROJECT_SUFFIX}")
    assert find_projects(tmp_path) == [first]
    with pytest.raises(ValueError):
        (tmp_path / "plain").mkdir()
        (tmp_path / "plain" / "project.json").write_text("{}", encoding="utf-8")
        read_manifest(tmp_path / "plain")


def test_relocation_moves_folders_and_layer_ids(tmp_path) -> None:
    old_root, new_root = str(tmp_path / "old" / "zeros"), str(tmp_path / "new" / "zeros")
    old_id, new_id = stable_model_id(old_root), stable_model_id(new_root)
    assert old_id != new_id and old_id.startswith("zeros_")
    record = FrameRecord("F_001", "F_001.png", model_mask_paths={old_id: old_root + "\\F_001.png"})
    build = BuildResult(records=(record,), model_specs=(ModelSpec(old_id, "zeros", Path(old_root)),))
    state = {
        "build_result": build,
        "tab": {"payloads": {old_id: {"F_001": object()}}, "selection": f"pair:{old_id}::other", "keys": {"F_001"}},
    }
    moved = relocate_state(state, [(old_root, new_root)], {old_id: new_id})
    moved_record = moved["build_result"].records[0]
    assert moved_record.model_mask_paths == {new_id: new_root + "\\F_001.png"}
    assert moved["build_result"].model_specs[0].model_id == new_id
    assert str(moved["build_result"].model_specs[0].mask_folder) == new_root
    assert set(moved["tab"]["payloads"]) == {new_id}
    assert moved["tab"]["payloads"][new_id]["F_001"] is state["tab"]["payloads"][old_id]["F_001"]
    assert moved["tab"]["selection"] == f"pair:{new_id}::other"
    assert moved["tab"]["keys"] == {"F_001"}
    # Folders: as stored when they exist, else relative to the project, else as stored.
    (tmp_path / "new" / "zeros").mkdir(parents=True)
    project = tmp_path / "new" / "run.karakal"
    assert resolve_folder(old_root, "../zeros", project) == str(Path(new_root))
    assert resolve_folder(old_root, "", project) == old_root


def _write(path: Path, image) -> str:
    cv2.imencode(".png", image)[1].tofile(str(path))
    return str(path)


def _widget(tmp_path: Path, qtbot, name: str):
    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / f"{name}.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    return widget, widget._presenter


def _run(tmp_path: Path, qtbot, data: Path):
    from karakal.core.domain import BuildOptions
    from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE

    folder = data / "zeros"
    folder.mkdir(parents=True)
    paths = {}
    for name, mask in (("F_001", smear(lattice(), 380, 350, 70, 90)), ("F_002", lattice())):
        paths[name] = (_write(folder / f"{name}.png", mask), _write(folder / f"{name}_confidence.png", sure(mask)))
    widget, presenter = _widget(tmp_path, qtbot, "first")
    item = presenter._append_folder_item(folder, checked=True)
    item.setData(FOLDER_CHECKED_ROLE, True)
    model_id = presenter._model_id_for_folder_item(item)
    records = tuple(
        FrameRecord(name, f"{name}.png", model_mask_paths={model_id: mask}, model_prob_paths={model_id: confidence})
        for name, (mask, confidence) in paths.items()
    )
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(BuildResult(records=records, options=BuildOptions(), model_specs=presenter._checked_model_specs()))
    widget.analysis_settings_group.setChecked(True)
    state = presenter._current_tab_state()
    presenter._start_compute_grid_inspection(state)
    qtbot.waitUntil(lambda: state.grid_inspection_results_ready and presenter._worker_thread is None, timeout=60000)
    dirt = presenter._dirt_controller
    dirt.use_processes = False
    qtbot.waitUntil(lambda: dirt.has_results() and not dirt.is_running(), timeout=60000)
    return widget, presenter, state, model_id


@pytest.fixture(autouse=True)
def _own_dirt_cache(tmp_path_factory, monkeypatch):
    import karakal.core.dirt_batch as dirt_batch

    folder = tmp_path_factory.mktemp("dirt_cache")
    monkeypatch.setattr(dirt_batch, "default_cache_dir", lambda: folder)


def test_a_finished_run_is_saved_and_opens_again_at_once(tmp_path, qtbot) -> None:
    widget, presenter, state, model_id = _run(tmp_path, qtbot, tmp_path / "data")
    projects = presenter._project_controller
    qtbot.waitUntil(lambda: projects.project_of(state) is not None and projects._thread is None, timeout=30000)
    # A drop rule, then the app closes: the view goes into the project too.
    state.frame_sets.add_dirt_rule("light", "zeros")
    state.excluded_record_keys = {"F_002"}
    projects.shutdown()
    folder = projects.project_of(state)
    assert folder.parent == projects.projects_dir() and folder.suffix == PROJECT_SUFFIX
    manifest = read_manifest(folder)
    assert manifest["frames"] == 2 and manifest["layers"][0]["model_id"] == model_id

    other, reopened = _widget(tmp_path, qtbot, "second")
    # Opening reads no frame cache and analyses nothing.
    reopened._start_compute_grid_inspection = lambda *_args, **_kwargs: pytest.fail("the project was recomputed")
    assert reopened._project_controller.open(folder)
    opened = reopened._current_tab_state()
    assert opened is not state and opened.grid_inspection_results_ready
    assert set(opened.grid_inspection_payload_by_key) == set(state.grid_inspection_payload_by_key)
    assert opened.grid_inspection_algorithm_version == state.grid_inspection_algorithm_version
    assert [record.key for record in opened.build_result.records] == ["F_001", "F_002"]
    assert any(rule.kind == "dirt" for rule in opened.frame_sets.rules)
    assert opened.excluded_record_keys == {"F_002"}
    dirt = reopened._dirt_controller
    assert set(dirt.frame_levels(model_id)) == {"F_001"}
    assert other.app_mode_combo.currentData() == "grid_inspection"
    assert reopened._project_controller.recent()[0] == folder


def test_a_moved_project_finds_its_folders_and_renames_layers(tmp_path, qtbot) -> None:
    widget, presenter, state, model_id = _run(tmp_path, qtbot, tmp_path / "old")
    projects = presenter._project_controller
    # The project lies beside the data, so it can travel with it.
    folder = projects.save(state, folder=tmp_path / "old" / f"run{PROJECT_SUFFIX}", wait=True)
    # Every save of the run is over (the cell check and the dirt search each save it too).
    qtbot.wait(300)
    qtbot.waitUntil(lambda: projects._thread is None and projects._pending is None, timeout=30000)
    shutil.move(str(tmp_path / "old"), str(tmp_path / "moved"))
    moved_project = tmp_path / "moved" / f"run{PROJECT_SUFFIX}"

    _other, reopened = _widget(tmp_path, qtbot, "second")
    assert reopened._project_controller.open(moved_project)
    opened = reopened._current_tab_state()
    new_id = stable_model_id(tmp_path / "moved" / "zeros")
    assert new_id != model_id
    paths = opened.build_result.records[0].model_mask_paths
    assert set(paths) == {new_id} and Path(paths[new_id]).is_file()
    assert new_id in opened.grid_inspection_payloads_by_model or opened.grid_inspection_payload_by_key
    assert set(reopened._dirt_controller.frame_levels(new_id)) == {"F_001"}
    assert Path(reopened.folder_list.item(0).data(256)) == tmp_path / "moved" / "zeros"


def test_a_project_of_the_matrix_is_a_recent_selection_that_opens_at_once(tmp_path, qtbot, monkeypatch) -> None:
    from karakal.core.recent_selections import MATRIX_READY

    widget, presenter, state, _model_id = _run(tmp_path, qtbot, tmp_path / "data")
    projects = presenter._project_controller
    folder = projects.save(state, wait=True)
    qtbot.waitUntil(lambda: projects._thread is None, timeout=30000)

    monkeypatch.setenv("KARAKAL_HOME_PAGE", "1")
    other, reopened = _widget(tmp_path, qtbot, "first")
    assert other.root_stack.currentWidget() is other.home_page
    home = other.home_page
    names = [home.selections.item(row).text().splitlines()[0] for row in range(home.selections.count())]
    assert names[0] == "data"
    # Picked: its snapshot is read, the matrix button is green and opens the project at once.
    home.selections.itemClicked.emit(home.selections.item(0))
    assert home.snapshot_dir() == str(tmp_path / "data")
    assert home.matrix_button.state == MATRIX_READY
    assert reopened._batch_controller.current().project == str(folder)
    reopened._start_compute_grid_inspection = lambda *_args, **_kwargs: pytest.fail("recomputed")
    home.matrix_button.click()
    assert other.root_stack.currentWidget() is other.workspace
    assert reopened._current_tab_state().grid_inspection_results_ready
    other.show_home_page()
    assert other.root_stack.currentWidget() is other.home_page


def _frame_with_cells(frame_id: str = "F"):
    """A frame result with one defect and four normal cells."""

    from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult

    def cell(index: int, status: str) -> GridCellAnalysisResult:
        return GridCellAnalysisResult(
            row=0, col=index, bbox=(index * 10, 0, 8, 8), centroid=(index * 10 + 4.0, 4.0), contour_id=None,
            status=status, score=0.0 if status == "normal" else 1.0, reasons=() if status == "normal" else ("broken_geometry",),
        )

    cells = tuple(cell(index, "normal" if index else "broken") for index in range(5))
    return GridFrameAnalysisResult(
        frame_id=frame_id, frame_path=f"{frame_id}.png", image_width=50, image_height=10, grid_rows=1, grid_cols=5,
        total_expected_cells=5, detected_cells=5, normal_cells=4, suspicious_cells=0, broken_cells=1,
        missing_cells=0, artifact_cells=0, damage_score=0.2, severity_level="low", per_cell_results=cells,
    )


def test_slim_results_keep_defects_counts_and_sharing() -> None:
    from dataclasses import replace

    from karakal.core.project_store import is_slim, slim_results

    frame = _frame_with_cells()
    cells = frame.per_cell_results
    state = {"by_key": {"F": frame}, "by_model": {"m": {"F": frame}}, "clean": replace(frame, per_cell_results=cells[:1], normal_cells=0)}
    slim = slim_results(state)
    kept = slim["by_key"]["F"]
    assert [c.status for c in kept.per_cell_results] == ["broken"]
    assert kept.normal_cells == 4 and kept.damage_score == 0.2 and is_slim(kept) and not is_slim(frame)
    assert slim["by_model"]["m"]["F"] is kept
    assert slim["clean"] is state["clean"] and not is_slim(state["clean"])


def test_an_opened_project_keeps_only_defects_and_the_frame_window_reanalyses(tmp_path, qtbot) -> None:
    from karakal.core.project_store import is_slim

    widget, presenter, state, _model_id = _run(tmp_path, qtbot, tmp_path / "data")
    # A frame of the run with normal cells (the small test masks give none).
    full = _frame_with_cells("F_001")
    state.grid_inspection_payload_by_key = {**state.grid_inspection_payload_by_key, "F_001": full}
    state.grid_inspection_payloads_by_layer = {
        layer: {**payloads, "F_001": full} for layer, payloads in state.grid_inspection_payloads_by_layer.items()
    }
    state.grid_inspection_payloads_by_model = {
        model: {layer: {**payloads, "F_001": full} for layer, payloads in layers.items()}
        for model, layers in state.grid_inspection_payloads_by_model.items()
    }
    projects = presenter._project_controller
    folder = projects.save(state, wait=True)
    qtbot.waitUntil(lambda: projects._thread is None, timeout=30000)
    _other, reopened = _widget(tmp_path, qtbot, "second")
    assert reopened._project_controller.open(folder)
    opened = reopened._current_tab_state()
    assert [cell.status for cell in opened.grid_inspection_payload_by_key["F_001"].per_cell_results] == ["broken"]
    results = list(opened.grid_inspection_payload_by_key.values())
    assert results and all(
        all(str(cell.status) != "normal" for cell in result.per_cell_results) for result in results
    )
    started = []
    reopened._start_grid_details_analyze = lambda dialog, record, *_args, **_kwargs: started.append(record.key)
    slim_record = next(
        record for record in opened.build_result.records if is_slim(opened.grid_inspection_payload_by_key.get(record.key))
    )
    reopened._open_record_details(slim_record, opened)
    qtbot.addWidget(reopened._details_dialogs[-1])
    assert started == [slim_record.key]


def test_a_save_waited_for_during_another_save_is_on_disk_when_it_returns(tmp_path, qtbot) -> None:
    widget, presenter, state, _model_id = _run(tmp_path, qtbot, tmp_path / "data")
    projects = presenter._project_controller
    folder = tmp_path / f"run{PROJECT_SUFFIX}"
    # One save runs (not waited for), a second one is asked for and waited for.
    projects.save(state, folder=folder)
    state.excluded_record_keys = {"F_001"}
    projects.save(state, folder=folder, wait=True)
    assert projects._thread is None and projects._pending is None
    view = read_results(folder, "view.pickle")
    assert view["tab"]["excluded_record_keys"] == {"F_001"}
    # The first save's own «finished» arrives later and must not touch anything.
    qtbot.wait(200)
    assert projects._thread is None


def _matrix_results(presenter) -> int:
    return len(presenter._grid_inspection_views()["unified"]._grid_inspection_payload_by_key)


def test_an_opened_project_colours_the_matrix(tmp_path, qtbot) -> None:
    widget, presenter, state, _model_id = _run(tmp_path, qtbot, tmp_path / "data")
    assert _matrix_results(presenter) == 2
    projects = presenter._project_controller
    folder = projects.save(state, wait=True)
    _other, reopened = _widget(tmp_path, qtbot, "second")
    assert reopened._project_controller.open(folder)
    # Every frame of the run is coloured at once (it was grey: nothing reached the matrix).
    assert _matrix_results(reopened) == 2


def test_a_mode_switch_keeps_the_matrix_coloured(tmp_path, qtbot) -> None:
    widget, presenter, _state, _model_id = _run(tmp_path, qtbot, tmp_path / "data")
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("validation"))
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    assert _matrix_results(presenter) == 2
