"""Read-only runtime configuration loaded from ``contour.ini``.

The shipped file is a template and a safe default.  A frozen application copies
it to the per-user application directory on first start; ``CONTOUR_CONFIG`` can
point to another file for deployments and tests.
"""

from __future__ import annotations

import configparser
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from .logging import DEFAULT_APP_NAME, DEFAULT_ORG_NAME, default_log_directory

CONFIG_ENV = "CONTOUR_CONFIG"
CONFIG_FILENAME = "contour.ini"


def bundled_config_path() -> Path:
    if bool(getattr(sys, "frozen", False)):
        root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        candidates = (root / CONFIG_FILENAME, root / "contour" / "resources" / CONFIG_FILENAME)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]
    return Path(__file__).resolve().parents[1] / "resources" / CONFIG_FILENAME


def user_config_path() -> Path:
    return default_log_directory(DEFAULT_ORG_NAME, DEFAULT_APP_NAME).parent / CONFIG_FILENAME


def runtime_config_path() -> Path:
    explicit = os.environ.get(CONFIG_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if bool(getattr(sys, "frozen", False)):
        target = user_config_path()
        if not target.exists():
            source = bundled_config_path()
            if source.is_file():
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                except OSError:
                    return source
        return target if target.is_file() else bundled_config_path()
    return bundled_config_path()


@lru_cache(maxsize=1)
def load_runtime_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(runtime_config_path(), encoding="utf-8")
    return parser


def clear_runtime_config_cache() -> None:
    """Reload configuration on the next access (primarily useful in tests)."""

    load_runtime_config.cache_clear()


def config_string(section: str, option: str, default: str) -> str:
    return load_runtime_config().get(section, option, fallback=default).strip()


def config_bool(section: str, option: str, default: bool) -> bool:
    try:
        return load_runtime_config().getboolean(section, option, fallback=default)
    except ValueError:
        return default


def config_int(section: str, option: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = load_runtime_config().getint(section, option, fallback=default)
    except ValueError:
        value = default
    return max(minimum, value) if minimum is not None else value


def config_float(section: str, option: str, default: float, *, minimum: float | None = None) -> float:
    try:
        value = load_runtime_config().getfloat(section, option, fallback=default)
    except ValueError:
        value = default
    return max(minimum, value) if minimum is not None else value


def contour_application_directory() -> Path:
    """Return the directory containing Contour.exe or the source launcher."""

    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def profiling_log_path() -> Path:
    raw = config_string("profiling", "log_file", "profiling.log")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return contour_application_directory() / "logs" / path
