"""Application logging configuration.

Provides a single entry point :func:`configure_logging` that sets up
a rotating file handler under the user's local app data directory and
a console handler for interactive runs.

The function is safe to call multiple times; the second invocation is a no-op
unless ``force=True`` is passed, which re-installs the handlers (useful in
tests).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = [
    "DEFAULT_APP_NAME",
    "DEFAULT_ORG_NAME",
    "configure_logging",
    "default_log_directory",
    "default_log_file",
    "install_background_exception_hooks",
]

DEFAULT_ORG_NAME = "ViaLaNet"
DEFAULT_APP_NAME = "Contour"

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 1 * 1024 * 1024
_BACKUP_COUNT = 5

_CONFIGURED = False
_EXCEPTION_HOOKS_INSTALLED = False
_PREVIOUS_THREADING_EXCEPTHOOK = threading.excepthook
_PREVIOUS_UNRAISABLEHOOK = sys.unraisablehook
_PREVIOUS_QT_MESSAGE_HANDLER: Any | None = None

_LOGGER = logging.getLogger(__name__)


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    if issubclass(args.exc_type, SystemExit):
        return
    thread_name = args.thread.name if args.thread is not None else "<unknown>"
    _LOGGER.critical(
        "Unhandled exception in thread %s",
        thread_name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _log_unraisable_exception(args: Any) -> None:
    exc_value = getattr(args, "exc_value", None)
    exc_type = type(exc_value) if exc_value is not None else RuntimeError
    exc_traceback: TracebackType | None = getattr(args, "exc_traceback", None)
    _LOGGER.error(
        "Unraisable exception%s in object %r",
        f" ({args.err_msg})" if getattr(args, "err_msg", None) else "",
        getattr(args, "object", None),
        exc_info=(exc_type, exc_value, exc_traceback) if exc_value is not None else None,
    )


def _log_qt_message(message_type: Any, context: Any, message: str) -> None:
    try:
        from PyQt6.QtCore import QtMsgType

        levels = {
            QtMsgType.QtDebugMsg: logging.DEBUG,
            QtMsgType.QtInfoMsg: logging.INFO,
            QtMsgType.QtWarningMsg: logging.WARNING,
            QtMsgType.QtCriticalMsg: logging.ERROR,
            QtMsgType.QtFatalMsg: logging.CRITICAL,
        }
        level = levels.get(message_type, logging.ERROR)
    except Exception:
        level = logging.ERROR
    location = ""
    if context is not None and getattr(context, "file", None):
        location = f" ({context.file}:{getattr(context, 'line', 0)} {getattr(context, 'function', '')})"
    _LOGGER.log(level, "Qt: %s%s", message, location)
    if _PREVIOUS_QT_MESSAGE_HANDLER is not None:
        _PREVIOUS_QT_MESSAGE_HANDLER(message_type, context, message)


def install_background_exception_hooks() -> None:
    """Log uncaught thread, unraisable, warning, and Qt runtime errors."""

    global _EXCEPTION_HOOKS_INSTALLED
    global _PREVIOUS_QT_MESSAGE_HANDLER, _PREVIOUS_THREADING_EXCEPTHOOK, _PREVIOUS_UNRAISABLEHOOK
    if _EXCEPTION_HOOKS_INSTALLED:
        return

    _PREVIOUS_THREADING_EXCEPTHOOK = threading.excepthook
    _PREVIOUS_UNRAISABLEHOOK = sys.unraisablehook

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        _log_thread_exception(args)
        _PREVIOUS_THREADING_EXCEPTHOOK(args)

    def unraisable_hook(args: Any) -> None:
        _log_unraisable_exception(args)
        _PREVIOUS_UNRAISABLEHOOK(args)

    threading.excepthook = thread_hook
    sys.unraisablehook = unraisable_hook
    logging.captureWarnings(True)
    try:
        from PyQt6.QtCore import qInstallMessageHandler

        _PREVIOUS_QT_MESSAGE_HANDLER = qInstallMessageHandler(_log_qt_message)
    except Exception:
        _LOGGER.exception("Failed to install the Qt message logging hook")
    _EXCEPTION_HOOKS_INSTALLED = True


def default_log_directory(
    org_name: str = DEFAULT_ORG_NAME,
    app_name: str = DEFAULT_APP_NAME,
) -> Path:
    """Return the platform-appropriate directory for application logs."""
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Logs"
    else:
        root = Path(os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state"))
    return root / org_name / app_name / "logs"


def default_log_file(
    org_name: str = DEFAULT_ORG_NAME,
    app_name: str = DEFAULT_APP_NAME,
) -> Path:
    """Return the default log file path used by :func:`configure_logging`."""
    return default_log_directory(org_name, app_name) / "app.log"


def configure_logging(
    *,
    verbose: bool = False,
    log_file: Path | str | None = None,
    force: bool = False,
    org_name: str = DEFAULT_ORG_NAME,
    app_name: str = DEFAULT_APP_NAME,
) -> Path:
    """Configure root logging with a rotating file + console handler.

    Parameters
    ----------
    verbose:
        When ``True`` the console handler level is lowered from ``WARNING`` to
        ``DEBUG`` and the file handler captures ``DEBUG`` messages.
    log_file:
        Optional path to the log file. Defaults to
        ``<local-app-data>/ViaLaNet/Contour/logs/app.log``.
    force:
        Re-initialise handlers even if logging has been configured before.
    org_name, app_name:
        Used to compute the default log directory.

    Returns
    -------
    Path
        The absolute path to the active log file.
    """
    global _CONFIGURED

    resolved_log_file = Path(log_file) if log_file else default_log_file(org_name, app_name)

    if _CONFIGURED and not force:
        return resolved_log_file

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        resolved_log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    logging.getLogger("PyQt6").setLevel(logging.WARNING)

    _CONFIGURED = True
    return resolved_log_file
