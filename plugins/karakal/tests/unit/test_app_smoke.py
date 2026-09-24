"""Offscreen smoke check for the grid calibration windows."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPixmap

from karakal.app.main_window import KarakalMainWindow, QtUpdateController
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.plugin.plugin import KarakalPlugin
from karakal.ui.details_dialog import ExtendFrameDetailsDialog
from karakal.ui.grid_examples_dialog import GridExamplesDialog
from karakal.ui.grid_tuning_dialog import GridTuningDialog
from karakal.ui.i18n import Translator
from karakal.ui.ui_constants import GRID_INSPECTION_PRESET_VALUES


def test_grid_calibration_windows_open_and_close(tmp_path, qtbot, monkeypatch) -> None:
    monkeypatch.setattr(QtUpdateController, "check_for_updates", lambda self, manual=False: None)
    mask_path = tmp_path / "mask.png"
    assert _grayscale_array_to_qimage(np.full((48, 48), 255, dtype=np.uint8)).save(str(mask_path))
    settings = QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat)
    window = KarakalMainWindow(settings=settings)
    qtbot.addWidget(window)
    window.show()

    record = FrameRecord("frame-1", "Frame 1", model_mask_paths={"model": str(mask_path)})
    dialog = ExtendFrameDetailsDialog(
        record,
        BuildResult(records=(record,), options=BuildOptions()),
        session_view_state={"preferred_model_id": "model", "result_kind": "grid_cell_defects"},
        allowed_result_kinds=("grid_cell_defects",),
        grid_inspection_source_path=str(mask_path),
    )
    dialog.show()

    translator = Translator("ru")

    def translate(key: str, **kwargs) -> str:
        return translator.tr(key, **kwargs)

    tuning = GridTuningDialog(translate, dict(GRID_INSPECTION_PRESET_VALUES["balanced"]), parent=dialog)
    examples = GridExamplesDialog(translate, [], QPixmap(), parent=dialog)
    tuning.show()
    examples.show()
    qtbot.wait(50)

    detail_thread = dialog._detail_thread
    dialog._stop_detail_worker()
    if detail_thread is not None:
        detail_thread.wait(2000)
    started = perf_counter()
    examples.close()
    tuning.close()
    dialog.close()
    window.close()
    assert perf_counter() - started < 5.0


def test_details_close_while_loading_lets_the_thread_finish(tmp_path, qtbot, monkeypatch) -> None:
    from time import sleep

    from karakal.core.workers import DetailPayloadWorker
    from karakal.ui.details_dialog import _ORPHAN_THREADS

    def slow_run(self) -> None:
        sleep(0.3)
        self.finished.emit({})

    monkeypatch.setattr(DetailPayloadWorker, "run", slow_run)
    mask_path = tmp_path / "mask.png"
    assert _grayscale_array_to_qimage(np.full((32, 32), 255, dtype=np.uint8)).save(str(mask_path))
    record = FrameRecord("frame-slow", "Frame", model_mask_paths={"model": str(mask_path)})
    dialog = ExtendFrameDetailsDialog(
        record,
        BuildResult(records=(record,), options=BuildOptions()),
        session_view_state={"preferred_model_id": "model", "result_kind": "grid_cell_defects"},
        allowed_result_kinds=("grid_cell_defects",),
    )
    dialog.show()
    qtbot.wait(20)
    dialog.close()
    qtbot.waitUntil(lambda: not any(thread.isRunning() for thread, _worker in _ORPHAN_THREADS), timeout=3000)


def test_details_open_close_twenty_times(tmp_path, qtbot, monkeypatch) -> None:
    from time import sleep

    from karakal.core.workers import DetailPayloadWorker

    def slow_run(self) -> None:
        sleep(0.05)
        self.finished.emit({})

    monkeypatch.setattr(DetailPayloadWorker, "run", slow_run)
    mask_path = tmp_path / "mask.png"
    assert _grayscale_array_to_qimage(np.full((32, 32), 255, dtype=np.uint8)).save(str(mask_path))
    record = FrameRecord("frame-loop", "Frame", model_mask_paths={"model": str(mask_path)})
    for _cycle in range(20):
        dialog = ExtendFrameDetailsDialog(
            record,
            BuildResult(records=(record,), options=BuildOptions()),
            session_view_state={"preferred_model_id": "model", "result_kind": "grid_cell_defects"},
            allowed_result_kinds=("grid_cell_defects",),
        )
        dialog.show()
        dialog.close()
    qtbot.wait(50)


def test_plugin_widget_shutdown_is_repeatable(qtbot, monkeypatch) -> None:
    monkeypatch.setattr(QtUpdateController, "check_for_updates", lambda self, manual=False: None)
    plugin = KarakalPlugin()
    widget = plugin.create_widget(host=None)
    qtbot.addWidget(widget)
    widget.show()
    plugin.shutdown()
    plugin.shutdown()


def test_calibration_arrows_stay_inside_selected_frames(qtbot) -> None:
    records = tuple(FrameRecord(f"f{index}", f"Frame {index}") for index in range(4))
    dialog = ExtendFrameDetailsDialog(
        records[0],
        BuildResult(records=records, options=BuildOptions()),
        session_view_state={"result_kind": "grid_cell_defects"},
        allowed_result_kinds=("grid_cell_defects",),
    )
    dialog.set_calibration_frame_keys(("f0", "f2"))
    assert dialog._legacy_step_record(1)
    assert str(dialog._record.key) == "f2"
    assert not dialog._legacy_step_record(1)
    dialog.hide()
