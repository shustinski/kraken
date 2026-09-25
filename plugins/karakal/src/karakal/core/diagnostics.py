"""Application diagnostics for hang traces and stage logs. Stdlib only."""

from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import sys
import threading
import traceback
import zipfile
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_FAULT_HANDLE = None
_HANG_HANDLE = None
_APP_LOGGER: logging.Logger | None = None
_LOG_DIR: Path | None = None
_STAGE_TIMERS: dict[str, float] = {}


def app_log_dir() -> Path:
    """Return logs/ next to the frozen exe, or under the package cwd."""

    global _LOG_DIR
    if _LOG_DIR is not None:
        return _LOG_DIR
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path.cwd()
    path = base / "logs"
    path.mkdir(parents=True, exist_ok=True)
    _LOG_DIR = path
    return path


def install_diagnostics(*, hang_interval_seconds: float = 30.0) -> Path:
    """Enable hang dumps, rotating app.log, and global exception hooks."""

    global _FAULT_HANDLE, _HANG_HANDLE, _APP_LOGGER
    log_dir = app_log_dir()
    hang_path = log_dir / "hang_trace.log"
    app_path = log_dir / "app.log"

    _FAULT_HANDLE = hang_path.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_FAULT_HANDLE, all_threads=True)
    try:
        faulthandler.dump_traceback_later(
            float(hang_interval_seconds),
            repeat=True,
            file=_FAULT_HANDLE,
        )
    except RuntimeError:
        # Some interpreters disallow dump_traceback_later without a main thread signal path.
        pass
    _HANG_HANDLE = _FAULT_HANDLE

    logger = logging.getLogger("karakal.app")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = RotatingFileHandler(
        app_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    _APP_LOGGER = logger

    previous_hook = sys.excepthook

    def _excepthook(exc_type, exc, tb) -> None:
        log_exception("sys.excepthook", exc)
        previous_hook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    if hasattr(threading, "excepthook"):

        def _thread_excepthook(args) -> None:  # type: ignore[no-untyped-def]
            log_exception(f"threading.excepthook[{getattr(args, 'thread', None)}]", args.exc_value)

        threading.excepthook = _thread_excepthook  # type: ignore[assignment]

    atexit.register(_shutdown_diagnostics)
    log_event("diagnostics.installed", pid=os.getpid(), log_dir=str(log_dir))
    return log_dir


def _shutdown_diagnostics() -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass
    for handle in (_FAULT_HANDLE, _HANG_HANDLE):
        try:
            if handle is not None and not handle.closed:
                handle.close()
        except Exception:
            pass


def get_app_logger() -> logging.Logger:
    global _APP_LOGGER
    if _APP_LOGGER is None:
        install_diagnostics()
    assert _APP_LOGGER is not None
    return _APP_LOGGER


def log_event(message: str, **fields: Any) -> None:
    parts = [message]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    get_app_logger().info(" ".join(parts))


def log_exception(where: str, error: BaseException | None = None) -> None:
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__)) if error else traceback.format_exc()
    get_app_logger().error("%s\n%s", where, text)


def stage_enter(name: str, **fields: Any) -> None:
    import time

    _STAGE_TIMERS[name] = time.perf_counter()
    log_event(f"stage.enter.{name}", **fields)


def stage_exit(name: str, **fields: Any) -> None:
    import time

    started = _STAGE_TIMERS.pop(name, None)
    elapsed_ms = None if started is None else round((time.perf_counter() - started) * 1000.0, 1)
    log_event(f"stage.exit.{name}", elapsed_ms=elapsed_ms, **fields)


def pack_diagnostics_zip(
    destination: Path | None = None,
    *,
    version: str = "",
    launch_args: list[str] | None = None,
) -> Path:
    """Pack logs/ plus version and argv into one zip for support."""

    log_dir = app_log_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = Path(destination) if destination is not None else log_dir / f"karakal_diagnostics_{stamp}.zip"
    meta = log_dir / "_diagnostics_meta.txt"
    meta.write_text(
        "\n".join(
            [
                f"version={version}",
                f"pid={os.getpid()}",
                f"argv={list(launch_args if launch_args is not None else sys.argv)}",
                f"cwd={Path.cwd()}",
                f"executable={sys.executable}",
                f"frozen={getattr(sys, 'frozen', False)}",
                f"created={datetime.now().isoformat(timespec='seconds')}",
            ]
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(meta, arcname="diagnostics_meta.txt")
        for path in sorted(log_dir.glob("*")):
            if path.is_file() and path.name != target.name and path.name != meta.name:
                archive.write(path, arcname=path.name)
    try:
        meta.unlink(missing_ok=True)
    except OSError:
        pass
    log_event("diagnostics.packaged", path=str(target))
    return target
