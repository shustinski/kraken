"""UI automation for live acceptance jobs; uses the actual plugin processors."""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QSettings, Qt, QTimer
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMessageBox

    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(3)
    root = Path(os.environ["KRAKEN_STAGING_ROOT"])
    import PyQt6.QtCore as qt_core

    class AcceptanceSettings(QSettings):
        def __init__(self, *_args, **_kwargs):
            super().__init__(str(root / "settings.ini"), QSettings.Format.IniFormat)

    # Explicit organization constructors use the Windows registry regardless of
    # setDefaultFormat. Keep test settings and remote-compute preferences local.
    qt_core.QSettings = AcceptanceSettings
    errors = []
    started = time.monotonic()

    def messages():
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QMessageBox):
            return
        if dialog.icon() in {QMessageBox.Icon.Critical, QMessageBox.Icon.Warning}:
            errors.append(f"{dialog.windowTitle()}: {dialog.text()}")
        button = dialog.button(QMessageBox.StandardButton.Yes) or dialog.button(QMessageBox.StandardButton.Ok)
        if button is not None:
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    def guarded(operation):
        def run():
            try:
                if errors:
                    raise RuntimeError(errors[-1])
                if time.monotonic() - started > 600:
                    raise TimeoutError("Plugin UI workflow did not complete in 10 minutes")
                operation()
            except Exception as exc:
                errors.append(str(exc))
                traceback.print_exc()
                QApplication.instance().exit(1)

        return run

    if sys.argv[1] == "contour":
        from contour.kraken_bridge import prepare_contour_launch
        from contour.application import bootstrap
        from contour.application.cli import main as contour_main

        session, _arguments = prepare_contour_launch([])
        original = bootstrap.build_application
        stage = {"started": False, "index": 0}

        def tick():
            widget = window._widget
            assert widget._image_list_model.rowCount() == len(session.input_paths)
            if widget._workspace.current_state is None:
                QTimer.singleShot(100, guarded(tick))
                return
            dataset = session.manifest.operation == "frames.dataset.prepare.v1"
            if not stage["started"]:
                stage["started"] = True
                if dataset:
                    widget.dataset_dir_edit.setText(str(session.output_directory))
                else:
                    QTest.mouseClick(widget.batch_button, Qt.MouseButton.LeftButton)
            if dataset:
                # Ten sample pairs are enough to exercise the training workflow.
                index = stage["index"]
                expected = str(session.input_paths[index])
                if widget._workspace.current_image_path != expected:
                    widget.image_list.setCurrentIndex(widget.image_list.model().index(index, 0))
                    QTimer.singleShot(100, guarded(tick))
                    return
                QTest.mouseClick(widget.export_dataset_button, Qt.MouseButton.LeftButton)
                assert (session.output_directory / "images" / session.input_paths[index].name).is_file()
                assert (session.output_directory / "cif" / f"{session.input_paths[index].stem}.cif").is_file()
                stage["index"] += 1
                if stage["index"] < min(10, len(session.input_paths)):
                    QTimer.singleShot(100, guarded(tick))
                    return
            else:
                count = len(list(session.output_directory.glob("*.cif")))
                if widget._batch_controller.is_running or count != len(session.input_paths):
                    QTimer.singleShot(200, guarded(tick))
                    return
            assert all(not path.exists() for path in session.input_paths)
            window.grab().save(str(root / "completed-window.png"))
            window._kraken_return_action.trigger()
            assert session.result_manifest_path.is_file()
            QApplication.instance().quit()

        def build(argv):
            nonlocal window
            app, window = original([*argv, "--no-qss", "--log-file", str(root / "contour.log")])
            timer = QTimer(window)
            timer.timeout.connect(messages)
            timer.start(20)
            QTimer.singleShot(500, guarded(tick))
            return app, window

        window = None
        bootstrap.build_application = build
        contour_main([])
    else:
        from neuralimage.kraken_bridge import load_session_from_values
        from neuralimage import controller as controller_module
        from neuralimage.application.dto import MainWindowState, SettingsState
        from neuralimage.lib.data_interfaces import WorkMode

        session = load_session_from_values(job_manifest=None, result_manifest=None, staging_root=None)
        original = controller_module.AppController
        stage = {"started": False}

        def tick():
            presenter = controller.main_window_presenter
            window = presenter.view
            if not stage["started"]:
                stage["started"] = True
                state = MainWindowState(
                    work_mode=WorkMode.train_only.value,
                    sample_folder=str(session.dataset_source.image_root),
                    label_folder=str(session.dataset_source.label_root),
                    result_folder=str(session.output_directory),
                    epochs=1,
                    ui_mode="advanced",
                )
                settings = SettingsState(
                    model="S 660k",
                    sample_size=(128, 128),
                    train_patch_size=(128, 128),
                    batch_size=2,
                    train_batch_size=2,
                    dataloader_num_workers=0,
                    vertical_rotation=False,
                    horizontal_rotation=False,
                    use_validation=False,
                    resize_enabled=True,
                    target_size=(128, 128),
                    step=128,
                )
                # Populate the same widgets used by task property restoration.
                # Execution itself starts only from the user's Start button.
                presenter._restore_task_state_to_ui(state, settings)
                assert window.btn_start.isEnabled()
                QTest.mouseClick(window.btn_start, Qt.MouseButton.LeftButton)
                assert presenter._processing_session.queue_snapshot()
            tasks = presenter._processing_session.queue_snapshot()
            assert tasks
            status = tasks[0].status
            if status == "finished_error":
                raise RuntimeError(f"NeuralImage UI training failed: {tasks[0]}")
            if presenter._processing_session.active_task is not None or not list(
                session.output_directory.rglob("*.pth")
            ):
                QTimer.singleShot(200, guarded(tick))
                return
            window.grab().save(str(root / "completed-window.png"))
            window._kraken_return_action.trigger()
            assert session.result_manifest_path.is_file()
            QApplication.instance().quit()

        def build(*args, **kwargs):
            nonlocal controller
            controller = original(*args, **kwargs)
            timer = QTimer(controller.main_window_presenter.view)
            timer.timeout.connect(messages)
            timer.start(20)
            QTimer.singleShot(500, guarded(tick))
            return controller

        controller = None
        controller_module.AppController = build
        session.run_interactive()
    if errors:
        raise RuntimeError(errors[-1])


if __name__ == "__main__":
    main()
