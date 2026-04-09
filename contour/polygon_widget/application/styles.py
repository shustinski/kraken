from __future__ import annotations

import re
from pathlib import Path

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_URL_PATTERN = re.compile(r'url\((?P<quote>["\']?)(?P<path>[^)"\']+)(?P=quote)\)')


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def styles_root() -> Path:
    return package_root() / "resources" / "styles"


def resolve_style_path(*parts: str) -> Path:
    return styles_root().joinpath(*parts)


def load_stylesheet(name: str = "dark_modern.qss") -> str:
    stylesheet_path = resolve_style_path(name)
    if not stylesheet_path.exists():
        return ""
    content = stylesheet_path.read_text(encoding="utf-8")
    return _rewrite_relative_urls(content, stylesheet_path.parent)


def shared_styles_root() -> Path:
    return styles_root()


def resolve_shared_style_path(*parts: str) -> Path:
    return resolve_style_path(*parts)


def load_shared_stylesheet(name: str = "dark_modern.qss") -> str:
    return load_stylesheet(name)


def _rewrite_relative_urls(content: str, base_dir: Path) -> str:
    def replacer(match: re.Match[str]) -> str:
        raw_path = match.group("path").strip()
        if not _is_relative_url(raw_path):
            return match.group(0)
        return f'url("{(base_dir / raw_path).resolve().as_uri()}")'

    return _URL_PATTERN.sub(replacer, content)


def _is_relative_url(raw_path: str) -> bool:
    if not raw_path:
        return False
    lowered = raw_path.lower()
    if raw_path.startswith("/") or lowered.startswith((":/", "qrc:/", "file:/", "http://", "https://", "data:")):
        return False
    return _WINDOWS_ABSOLUTE_PATH.match(raw_path) is None
