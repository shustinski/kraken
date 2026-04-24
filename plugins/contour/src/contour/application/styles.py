from __future__ import annotations

import re
from pathlib import Path

from kraken_core.styles import (
    load_shared_stylesheet as load_core_shared_stylesheet,
    shared_icon_path,
    shared_styles_root as core_shared_styles_root,
    rewrite_relative_urls as _rewrite_relative_urls,
)

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_URL_PATTERN = re.compile(r'url\((?P<quote>["\']?)(?P<path>[^)"\']+)(?P=quote)\)')


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def styles_root() -> Path:
    return core_shared_styles_root()


def resolve_style_path(*parts: str) -> Path:
    if parts and parts[0] == "icons":
        icon_name = Path(*parts[1:]).name
        if icon_name in {"icon.png", "icon.ico"}:
            return shared_icon_path("contour", suffix=Path(icon_name).suffix)
    return styles_root().joinpath(*parts)


def load_stylesheet(name: str = "dark_modern.qss") -> str:
    return load_core_shared_stylesheet(name)


def shared_styles_root() -> Path:
    return styles_root()


def resolve_shared_style_path(*parts: str) -> Path:
    return resolve_style_path(*parts)


def load_shared_stylesheet(name: str = "dark_modern.qss") -> str:
    return load_stylesheet(name)


def _is_relative_url(raw_path: str) -> bool:
    if not raw_path:
        return False
    lowered = raw_path.lower()
    if raw_path.startswith("/") or lowered.startswith((":/", "qrc:/", "file:/", "http://", "https://", "data:")):
        return False
    return _WINDOWS_ABSOLUTE_PATH.match(raw_path) is None
