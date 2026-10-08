"""Each layer of the run is its own project: own frames, own source photos, own matrix width."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings

from karakal.core.analysis_profiles import AnalysisProfileKind, build_standalone_preflight
from karakal.core.analytics import collect_frame_records
from karakal.core.domain import BuildOptions, ModelSpec
from karakal.core.image_io import _grayscale_array_to_qimage


def _frames(folder: Path, names) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        assert _grayscale_array_to_qimage(np.full((8, 8), 255, dtype=np.uint8)).save(str(folder / name))


def _two_projects(tmp_path: Path) -> tuple[ModelSpec, ModelSpec]:
    _frames(tmp_path / "a_masks", ["A_0001.jpg", "A_0002.jpg", "SHARED_0005.jpg"])
    _frames(tmp_path / "a_src", ["A_0001.jpg", "A_0002.jpg", "SHARED_0005.jpg"])
    _frames(tmp_path / "b_masks", ["B_0001.jpg", "B_0002.jpg", "B_0003.jpg", "SHARED_0005.jpg"])
    _frames(tmp_path / "b_src", ["B_0001.jpg", "B_0002.jpg", "B_0003.jpg", "SHARED_0005.jpg"])
    return (
        ModelSpec("a", "A", tmp_path / "a_masks", original_folder=tmp_path / "a_src"),
        ModelSpec("b", "B", tmp_path / "b_masks", original_folder=tmp_path / "b_src"),
    )


def test_layers_keep_their_own_frames_and_source_photos(tmp_path) -> None:
    specs = _two_projects(tmp_path)
    result = collect_frame_records(specs, BuildOptions(union_layer_frames=True))
    by_key = {record.key: record for record in result.records}
    assert set(by_key) == {"A_0001.jpg", "A_0002.jpg", "B_0001.jpg", "B_0002.jpg", "B_0003.jpg", "SHARED_0005.jpg"}
    only_b = by_key["B_0003.jpg"]
    assert set(only_b.model_mask_paths) == {"b"}
    assert Path(only_b.original_path).parent.name == "b_src"
    assert Path(only_b.first_path).parent.name == "b_masks"
    # A name met in both projects: one frame, each layer with its own photo.
    shared = by_key["SHARED_0005.jpg"]
    assert set(shared.model_mask_paths) == {"a", "b"}
    assert {model: Path(path).parent.name for model, path in shared.model_original_paths.items()} == {
        "a": "a_src",
        "b": "b_src",
    }
    assert Path(shared.original_path).parent.name == "a_src"


def test_model_comparison_still_takes_only_common_frames(tmp_path) -> None:
    result = collect_frame_records(_two_projects(tmp_path), BuildOptions())
    assert [record.key for record in result.records] == ["SHARED_0005.jpg"]


def test_preflight_does_not_cross_check_layers_of_grid_defects(tmp_path) -> None:
    specs = _two_projects(tmp_path)
    report = build_standalone_preflight(AnalysisProfileKind.GRID_DEFECTS, None, specs)
    assert not any(issue.code == "partial_coverage" for issue in report.issues)
    assert report.total_frames == 6
    original = next(role for role in report.roles if role.role.value == "original")
    assert original.state == "ready" and original.frame_count == 6


def test_matrix_shows_the_frames_and_width_of_the_chosen_layer(tmp_path, qtbot) -> None:
    from karakal.app.main_window import KarakalWidget
    from karakal.ui.ui_constants import FOLDER_FRAMES_PER_ROW_ROLE, FOLDER_ORIGINAL_ROLE

    specs = _two_projects(tmp_path)
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    for spec in specs:
        item = presenter._append_folder_item(spec.mask_folder, checked=True)
        item.setData(FOLDER_ORIGINAL_ROLE, str(spec.original_folder))
    presenter._refresh_folder_rows()
    checked = presenter._checked_model_specs()
    assert [Path(spec.original_folder).name for spec in checked] == ["a_src", "b_src"]
    model_a, model_b = (spec.model_id for spec in checked)

    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(collect_frame_records(checked, BuildOptions(union_layer_frames=True)))
    state = presenter._current_tab_state()
    combo = widget.grid_matrix_layer_combo
    assert len(presenter._display_records_for_state(state)) == 6

    presenter._folder_item_for_model_id(model_b).setData(FOLDER_FRAMES_PER_ROW_ROLE, 2)
    combo.setCurrentIndex(combo.findData(model_b))
    shown = presenter._display_records_for_state(state)
    assert {record.key for record in shown} == {"B_0001.jpg", "B_0002.jpg", "B_0003.jpg", "SHARED_0005.jpg"}
    assert presenter._matrix_layout_config(state).frames_per_row == 2
    view = widget.grid_inspection_matrix_view
    assert {record.key for record in view._records} == {record.key for record in shown}

    combo.setCurrentIndex(combo.findData(model_a))
    assert {record.key for record in presenter._display_records_for_state(state)} == {
        "A_0001.jpg",
        "A_0002.jpg",
        "SHARED_0005.jpg",
    }
    # Layer A has no width of its own: the run's common one.
    assert presenter._matrix_layout_config(state).frames_per_row == state.layout_config.frames_per_row

    # Settings keep each layer's source folder and width.
    payload = presenter._build_folder_manager_payload()
    assert [Path(entry["original_path"]).name for entry in payload["folders"]] == ["a_src", "b_src"]
    assert [entry["frames_per_row"] for entry in payload["folders"]] == [0, 2]
    qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=30000)
