from __future__ import annotations

import re
from pathlib import Path

from kraken_core import styles as core_styles

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_URL_PATTERN = re.compile(r'url\((?P<quote>["\']?)(?P<path>[^)"\']+)(?P=quote)\)')


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def styles_root() -> Path:
    return core_styles.shared_styles_root()


def resolve_style_path(*parts: str) -> Path:
    if parts and parts[0] == "icons":
        icon_name = Path(*parts[1:]).name
        if icon_name in {"icon.png", "icon.ico"}:
            candidate = core_styles.plugin_icon_path("contour", suffix=Path(icon_name).suffix)
            if candidate.exists():
                return candidate
            return core_styles.shared_icon_path("kraken", suffix=Path(icon_name).suffix)
    return styles_root().joinpath(*parts)


def load_stylesheet(name: str = "dark_modern.qss") -> str:
    return core_styles.load_shared_stylesheet(name)


def shared_styles_root() -> Path:
    return styles_root()


def resolve_shared_style_path(*parts: str) -> Path:
    return resolve_style_path(*parts)


def load_shared_stylesheet(name: str = "dark_modern.qss") -> str:
    return load_stylesheet(name)


def _rewrite_relative_urls(content: str, base_dir: str | Path) -> str:
    return core_styles.rewrite_relative_urls(content, base_dir)


def _is_relative_url(raw_path: str) -> bool:
    if not raw_path:
        return False
    lowered = raw_path.lower()
    if raw_path.startswith("/") or lowered.startswith((":/", "qrc:/", "file:/", "http://", "https://", "data:")):
        return False
    return _WINDOWS_ABSOLUTE_PATH.match(raw_path) is None
