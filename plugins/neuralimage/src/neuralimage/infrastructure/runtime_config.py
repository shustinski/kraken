"""Read-only runtime configuration loaded from ``neuralimage.ini``.

The bundled file in ``resources/`` is the default.  ``NEURALIMAGE_CONFIG`` can
point to another file for deployments and tests.
"""

from __future__ import annotations

import configparser
import os
from functools import lru_cache
from pathlib import Path

from neuralimage.lib.runtime_paths import resolve_resource_path

CONFIG_ENV = "NEURALIMAGE_CONFIG"
CONFIG_FILENAME = "neuralimage.ini"


def bundled_config_path() -> Path:
    return resolve_resource_path(CONFIG_FILENAME)


def runtime_config_path() -> Path:
    explicit = os.environ.get(CONFIG_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
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
