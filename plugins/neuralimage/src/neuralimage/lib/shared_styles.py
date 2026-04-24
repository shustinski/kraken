from __future__ import annotations

import re
from pathlib import Path

from kraken_core.styles import load_stylesheet as load_core_stylesheet
from kraken_core.styles import rewrite_relative_urls as _rewrite_relative_urls
from kraken_core.styles import shared_styles_root as core_shared_styles_root

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_URL_PATTERN = re.compile(r'url\((?P<quote>["\']?)(?P<path>[^)"\']+)(?P=quote)\)')


def shared_styles_root() -> Path:
    return core_shared_styles_root()


def resolve_shared_style_path(*parts: str) -> Path:
    return shared_styles_root().joinpath(*parts)


def load_shared_stylesheet(name: str = "dark_modern.qss") -> str:
    return load_stylesheet(resolve_shared_style_path(name))


def load_stylesheet(stylesheet_path: str | Path) -> str:
    return load_core_stylesheet(stylesheet_path)


def _is_relative_url(raw_path: str) -> bool:
    if not raw_path:
        return False
    lowered = raw_path.lower()
    if raw_path.startswith("/") or lowered.startswith((":/", "qrc:/", "file:/", "http://", "https://", "data:")):
        return False
    return _WINDOWS_ABSOLUTE_PATH.match(raw_path) is None
