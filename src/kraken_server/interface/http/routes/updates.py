"""Public read-only distribution of operator-published update files."""

from __future__ import annotations

import os
import re
import json
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

from kraken_server.interface.http.prefix import API_PREFIX

_PRODUCTS = frozenset({"kraken-hub", "contour", "neuralimage"})
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _valid_manifest(path: Path, directory: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError, ValueError, UnicodeError:
        return False
    if not isinstance(payload, dict) or not str(payload.get("version", "")).strip():
        return False
    releases = payload.get("releases", [])
    if not isinstance(releases, list):
        return False
    candidates = [payload, *releases] if releases else [payload]
    for release in candidates:
        if not isinstance(release, dict):
            return False
        filename = release.get("download_url")
        if not filename:
            if release is payload and releases:
                continue
            return False
        digest = release.get("sha256", payload.get("sha256"))
        if (
            not isinstance(filename, str)
            or not _NAME.fullmatch(filename)
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            return False
        installer = (directory / filename).resolve()
        if not installer.is_relative_to(directory) or not installer.is_file():
            return False
    return True


def register(app: object) -> None:
    @app.get(f"{API_PREFIX}/updates/{{product}}/{{channel}}/{{filename}}")
    def read_update_file(product: str, channel: str, filename: str) -> FileResponse:
        root = os.getenv("KRAKEN_UPDATES_ROOT", "").strip()
        if not root or product not in _PRODUCTS or not _NAME.fullmatch(channel) or not _NAME.fullmatch(filename):
            raise HTTPException(status_code=404)
        resolved_root = Path(root).resolve()
        directory = (resolved_root / product / channel).resolve()
        target = (directory / filename).resolve()
        if not directory.is_relative_to(resolved_root) or not target.is_relative_to(directory) or not target.is_file():
            raise HTTPException(status_code=404)
        if filename == "version.json" and not _valid_manifest(target, directory):
            raise HTTPException(status_code=404)
        return FileResponse(
            target, media_type="application/json" if filename == "version.json" else "application/octet-stream"
        )
