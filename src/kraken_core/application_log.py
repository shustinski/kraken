"""Persistent application log for the installed desktop build."""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER = logging.getLogger("kraken")
_HANDLER_FLAG = "_kraken_application_log"


def configure(directory: str | Path) -> Path:
    """Append warnings and errors to kraken.log beside the application."""

    path = Path(directory).expanduser().resolve() / "kraken.log"
    root = logging.getLogger()
    if any(getattr(handler, _HANDLER_FLAG, False) for handler in root.handlers):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8", delay=True)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    setattr(handler, _HANDLER_FLAG, True)
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    sys.excepthook = _log_unhandled
    threading.excepthook = _log_thread
    return path


def _log_unhandled(exc_type, exc, traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, traceback)
        return
    _LOGGER.error("Unhandled exception", exc_info=(exc_type, exc, traceback))


def _log_thread(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is not None and issubclass(args.exc_type, SystemExit):
        return
    _LOGGER.error(
        "Unhandled thread exception in %s",
        getattr(args.thread, "name", "thread"),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
