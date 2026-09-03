from __future__ import annotations

import logging
import sys
import threading
from types import SimpleNamespace

from PyQt6 import QtCore

from contour.infrastructure import logging as contour_logging


def _flush_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_background_error_channels_are_written_to_app_log(tmp_path) -> None:
    log_file = tmp_path / "app.log"
    root = logging.getLogger()
    original_level = root.level
    handler = logging.FileHandler(log_file, encoding="utf-8")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    try:
        try:
            raise RuntimeError("thread boom")
        except RuntimeError as exc:
            contour_logging._log_thread_exception(
                SimpleNamespace(
                    exc_type=type(exc),
                    exc_value=exc,
                    exc_traceback=exc.__traceback__,
                    thread=SimpleNamespace(name="frame-worker"),
                )
            )

        try:
            raise ValueError("finalizer boom")
        except ValueError as exc:
            contour_logging._log_unraisable_exception(
                SimpleNamespace(
                    exc_value=exc,
                    exc_traceback=exc.__traceback__,
                    err_msg="during cleanup",
                    object="frame-cache",
                )
            )

        contour_logging._log_qt_message(
            QtCore.QtMsgType.QtCriticalMsg,
            SimpleNamespace(file="worker.cpp", line=42, function="run", category=None),
            "Qt worker failure",
        )
        _flush_handlers()
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(original_level)

    contents = log_file.read_text(encoding="utf-8")
    assert "Unhandled exception in thread frame-worker" in contents
    assert "RuntimeError: thread boom" in contents
    assert "Unraisable exception (during cleanup) in object 'frame-cache'" in contents
    assert "ValueError: finalizer boom" in contents
    assert "Qt: Qt worker failure (worker.cpp:42 run)" in contents


def test_background_exception_hook_installation_is_idempotent(monkeypatch) -> None:
    installed_qt_handlers = []
    original_thread_hook = threading.excepthook
    original_unraisable_hook = sys.unraisablehook
    monkeypatch.setattr(threading, "excepthook", original_thread_hook)
    monkeypatch.setattr(sys, "unraisablehook", original_unraisable_hook)
    monkeypatch.setattr(contour_logging, "_EXCEPTION_HOOKS_INSTALLED", False)
    monkeypatch.setattr(contour_logging, "_PREVIOUS_QT_MESSAGE_HANDLER", None)
    monkeypatch.setattr(
        QtCore,
        "qInstallMessageHandler",
        lambda handler: installed_qt_handlers.append(handler),
    )

    try:
        contour_logging.install_background_exception_hooks()
        installed_thread_hook = threading.excepthook
        installed_unraisable_hook = sys.unraisablehook
        contour_logging.install_background_exception_hooks()

        assert installed_thread_hook is not original_thread_hook
        assert installed_unraisable_hook is not original_unraisable_hook
        assert threading.excepthook is installed_thread_hook
        assert sys.unraisablehook is installed_unraisable_hook
        assert installed_qt_handlers == [contour_logging._log_qt_message]
    finally:
        logging.captureWarnings(False)
