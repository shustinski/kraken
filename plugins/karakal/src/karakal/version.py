"""Single version source for Karakal builds and auto-update."""

from __future__ import annotations

APP_VERSION = "0.1.0-beta1"
__version__ = APP_VERSION


def numeric_version(version: str | None = None) -> str:
    """Return Inno-compatible numeric VersionInfoVersion (X.Y.Z.0)."""

    text = str(version or APP_VERSION).strip()
    core = text.split("-", 1)[0].split("+", 1)[0]
    parts = [part for part in core.split(".") if part.isdigit()]
    while len(parts) < 4:
        parts.append("0")
    return ".".join(parts[:4])
