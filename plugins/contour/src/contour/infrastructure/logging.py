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
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = [
    "DEFAULT_APP_NAME",
    "DEFAULT_ORG_NAME",
    "configure_logging",
    "default_log_directory",
    "default_log_file",
]

DEFAULT_ORG_NAME = "ViaLaNet"
DEFAULT_APP_NAME = "Contour"

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 1 * 1024 * 1024
_BACKUP_COUNT = 5

_CONFIGURED = False


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
