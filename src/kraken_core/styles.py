from __future__ import annotations

import re
from pathlib import Path

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_URL_PATTERN = re.compile(r'url\((?P<quote>["\']?)(?P<path>[^)"\']+)(?P=quote)\)')


def resources_root() -> Path:
    return Path(__file__).resolve().parent / "resources"


def shared_styles_root() -> Path:
    return resources_root() / "styles"


def shared_icon_path(name: str, *, suffix: str | None = None) -> Path:
    requested_suffix = suffix or (".ico" if name.endswith(".ico") else ".png" if name.endswith(".png") else "")
    normalized = name if requested_suffix and name.endswith(requested_suffix) else f"{name}{requested_suffix or '.png'}"
    return resources_root() / "icons" / normalized


def load_shared_stylesheet(name: str = "dark_modern.qss") -> str:
    return load_named_stylesheet(shared_styles_root(), name)


def load_stylesheet(path: str | Path) -> str:
    stylesheet_path = Path(path)
    if not stylesheet_path.exists():
        return ""
    content = stylesheet_path.read_text(encoding="utf-8")
    return rewrite_relative_urls(content, stylesheet_path.parent)


def load_named_stylesheet(styles_root: str | Path, name: str = "dark_modern.qss") -> str:
    return load_stylesheet(Path(styles_root) / name)


def rewrite_relative_urls(content: str, base_dir: str | Path) -> str:
    root = Path(base_dir)

    def replacer(match: re.Match[str]) -> str:
        raw_path = match.group("path").strip()
        if not is_relative_url(raw_path):
            return match.group(0)
        return f'url("{(root / raw_path).resolve().as_posix()}")'

    return _URL_PATTERN.sub(replacer, content)


def is_relative_url(raw_path: str) -> bool:
    if not raw_path:
        return False
    lowered = raw_path.lower()
    if raw_path.startswith("/") or lowered.startswith((":/", "qrc:/", "file:/", "http://", "https://", "data:")):
        return False
    return _WINDOWS_ABSOLUTE_PATH.match(raw_path) is None
