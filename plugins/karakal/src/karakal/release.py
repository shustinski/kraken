"""Version numbers for publishing to the update folder (used by scripts/publish.ps1).

The base number comes from the code: ``<release>.<algorithm><interface>``
(``version.py``). A beta gets the next free ``-betaN`` for that base in the beta
channel; a stable release is the base itself. Once a base went out as stable,
the next beta needs a new algorithm or interface number, otherwise the beta would
be older than the release the testers already have.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .version import base_version

_BETA_RE = re.compile(r"^(?P<base>.+)-beta(?P<number>\d+)$", re.IGNORECASE)


def channel_versions(share_root: str | Path, channel: str) -> list[str]:
    """Every version listed in ``<root>/<channel>/version.json``."""

    manifest = Path(share_root) / channel / "version.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    versions = [str(payload.get("version", "") or "")]
    for item in payload.get("releases", []) or []:
        if isinstance(item, dict):
            versions.append(str(item.get("version", "") or ""))
    return [version for version in versions if version and version != "0.0.0"]


def next_publish_version(channel: str, share_root: str | Path, base: str | None = None) -> str:
    base = base or base_version()
    stable = set(channel_versions(share_root, "stable"))
    if channel == "stable":
        if base in stable:
            raise ValueError(f"Релиз {base} уже опубликован: поднимите номер интерфейса или алгоритма")
        return base
    if channel != "beta":
        raise ValueError(f"Неизвестный канал: {channel}")
    if base in stable:
        raise ValueError(
            f"{base} уже вышел релизом: новая бета была бы старше релиза. "
            "Поднимите INTERFACE_NUMBER в version.py (или номер алгоритма)"
        )
    numbers = [
        int(match.group("number"))
        for version in channel_versions(share_root, "beta")
        if (match := _BETA_RE.match(version)) and match.group("base") == base
    ]
    return f"{base}-beta{max(numbers) + 1 if numbers else 0}"


def set_code_beta(version_file: str | Path, version: str) -> bool:
    """Write the published beta number into ``version.py``; False for a stable version."""

    match = _BETA_RE.match(version)
    if match is None:
        return False
    path = Path(version_file)
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(r"^BETA = \d+$", f"BETA = {int(match.group('number'))}", text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"В {path} не найдена строка BETA = N")
    path.write_text(updated, encoding="utf-8")
    return True
