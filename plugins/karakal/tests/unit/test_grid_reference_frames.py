"""Reference frames build the cell template; the tester entry point runs the tester UI from source."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings

from karakal.app import presenter as presenter_module
from karakal.app.main_window import KarakalWidget
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE


def _widget_with_frames(tmp_path: Path, qtbot, names: list[str]):
    folder = tmp_path / "masks"
    folder.mkdir()
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:30, 10:30] = 255
    for name in names:
        assert _grayscale_array_to_qimage(image).save(str(folder / name))
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
    assert state is not None
    return widget, presenter, state, model_id, records


def test_reference_frames_from_a_folder_match_by_file_name(tmp_path, qtbot) -> None:
    names = [f"TEST_{index:05d}.jpg" for index in range(6)]
    widget, presenter, state, _model_id, _records = _widget_with_frames(tmp_path, qtbot, names)
    shots = tmp_path / "good_shots"
    shots.mkdir()
    for name in (names[1], names[4], "TEST_99999.jpg"):
        (shots / name).write_bytes(b"jpg")

    matched = presenter._grid_reference_records_from_folder(state, shots)
    presenter._add_grid_reference_records(state, matched)

    assert state.grid_inspection_reference_record_keys == ("TEST_00001", "TEST_00004")
    assert "2" in widget.grid_reference_frame_label.text()
    presenter._on_grid_reference_clear_requested()
    assert state.grid_inspection_reference_record_keys == ()


def test_template_is_built_from_reference_frames_only(tmp_path, qtbot, monkeypatch) -> None:
    names = [f"TEST_{index:05d}.jpg" for index in range(6)]
    _widget, presenter, state, model_id, records = _widget_with_frames(tmp_path, qtbot, names)
    calls: list[dict] = []

    def fake_estimate(paths, **kwargs):
        calls.append({"paths": list(paths), **kwargs})
        return None

    monkeypatch.setattr(presenter_module, "estimate_run_cell_reference_profile", fake_estimate)
    presenter._estimate_run_cell_reference_profile_for_state(state, None, model_id=model_id, records=records)
    assert len(calls[-1]["paths"]) == 6 and calls[-1]["sample_limit"] == 32

    presenter._add_grid_reference_records(state, [records[2]])
    presenter._estimate_run_cell_reference_profile_for_state(state, None, model_id=model_id, records=records)
    # One good frame is enough, and every reference frame is used.
    assert [Path(path).name for path in calls[-1]["paths"]] == [names[2]]
    assert calls[-1]["sample_limit"] == 0 and calls[-1]["min_frames"] == 1


def test_tester_entry_point_runs_the_tester_profile_with_its_own_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KARAKAL_BUILD_PROFILE", raising=False)
    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path / "tester"))
    sys.modules.pop("karakal.debug.tester_run", None)
    importlib.import_module("karakal.debug.tester_run")

    from karakal.core.features import tester_build
    from karakal.infra.services import default_settings

    assert tester_build()
    assert Path(default_settings().fileName()) == tmp_path / "tester" / "settings.ini"
