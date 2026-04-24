from __future__ import annotations

import re
from pathlib import Path

from lib.runtime_paths import resolve_internal_path

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_URL_PATTERN = re.compile(r'url\((?P<quote>["\']?)(?P<path>[^)"\']+)(?P=quote)\)')


def shared_styles_root() -> Path:
    return resolve_internal_path("resources")


def resolve_shared_style_path(*parts: str) -> Path:
    return shared_styles_root().joinpath(*parts)


def load_shared_stylesheet(name: str = "dark_modern.qss") -> str:
    return load_stylesheet(resolve_shared_style_path(name))


def load_stylesheet(stylesheet_path: str | Path) -> str:
    path = Path(stylesheet_path)
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8")
    return _rewrite_relative_urls(content, path.parent)


def _rewrite_relative_urls(content: str, base_dir: Path) -> str:
    def replacer(match: re.Match[str]) -> str:
        raw_path = match.group("path").strip()
        if not _is_relative_url(raw_path):
            return match.group(0)
        resolved_path = (base_dir / raw_path).resolve()
        return f'url("{resolved_path.as_posix()}")'

    return _URL_PATTERN.sub(replacer, content)


def _is_relative_url(raw_path: str) -> bool:
    if not raw_path:
        return False
    lowered = raw_path.lower()
    if raw_path.startswith("/") or lowered.startswith((":/", "qrc:/", "file:/", "http://", "https://", "data:")):
        return False
    return _WINDOWS_ABSOLUTE_PATH.match(raw_path) is None
