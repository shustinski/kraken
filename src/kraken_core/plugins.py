from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import request

from .runtime import current_platform


@dataclass(frozen=True)
class PluginExecutable:
    path: str = ""
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginMetadata:
    id: str
    display_name: str
    description: str = ""
    version: str = "0.0.0"
    enabled: bool = True
    platforms: tuple[str, ...] = ("windows", "linux")
    executables: dict[str, PluginExecutable] = field(default_factory=dict)
    update_manifests: dict[str, str] = field(default_factory=dict)
    actions: tuple[str, ...] = ()

    def executable_for(self, platform: str | None = None) -> PluginExecutable:
        return self.executables.get((platform or current_platform()).lower(), PluginExecutable())

    def update_manifest_for(self, platform: str | None = None) -> str:
        return self.update_manifests.get((platform or current_platform()).lower(), "")


def load_plugin_catalog(path: str | Path) -> list[PluginMetadata]:
    source = str(path)
    try:
        if source.startswith(("http://", "https://")):
            with request.urlopen(source, timeout=5.0) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                payload = json.loads(response.read().decode(charset))
        else:
            catalog_path = Path(path)
            if not catalog_path.is_file():
                return []
            payload = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
            return []
    raw_plugins = payload.get("plugins", payload if isinstance(payload, list) else [])
    if not isinstance(raw_plugins, list):
        return []
    return [parse_plugin_metadata(item) for item in raw_plugins if isinstance(item, dict)]


def parse_plugin_metadata(payload: dict[str, Any]) -> PluginMetadata:
    executables: dict[str, PluginExecutable] = {}
    for platform, raw_value in (payload.get("executables") or {}).items():
        if isinstance(raw_value, str):
            executables[str(platform).lower()] = PluginExecutable(path=raw_value)
        elif isinstance(raw_value, dict):
            command = raw_value.get("command", ())
            executables[str(platform).lower()] = PluginExecutable(
                path=str(raw_value.get("path", "") or ""),
                command=tuple(str(part) for part in command) if isinstance(command, list) else (),
            )
    return PluginMetadata(
        id=str(payload["id"]),
        display_name=str(payload.get("display_name", payload["id"])),
        description=str(payload.get("description", "") or ""),
        version=str(payload.get("version", "0.0.0") or "0.0.0"),
        enabled=bool(payload.get("enabled", True)),
        platforms=tuple(str(item).lower() for item in payload.get("platforms", ("windows", "linux"))),
        executables=executables,
        update_manifests={str(k).lower(): str(v) for k, v in (payload.get("update_manifests") or {}).items()},
        actions=tuple(str(item) for item in payload.get("actions", ())),
    )
