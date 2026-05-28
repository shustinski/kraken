from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from urllib import request

from .runtime import current_platform


@dataclass(frozen=True)
class PluginExecutable:
    path: str = ""
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginVersionEntry:
    version: str
    notes: str = ""


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
    source_dir: str = ""
    version_history: tuple[PluginVersionEntry, ...] = ()

    def executable_for(self, platform: str | None = None) -> PluginExecutable:
        return self.executables.get((platform or current_platform()).lower(), PluginExecutable())

    def update_manifest_for(self, platform: str | None = None) -> str:
        return self.update_manifests.get((platform or current_platform()).lower(), "")


@dataclass(frozen=True)
class PluginInventoryItem:
    metadata: PluginMetadata
    installed: bool = False


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
        source_dir=str(payload.get("source_dir", "") or ""),
        version_history=parse_version_history(payload),
    )


def scan_plugin_directory(path: str | Path) -> list[PluginMetadata]:
    plugins_dir = Path(path).expanduser().resolve()
    if not plugins_dir.is_dir():
        return []
    plugins: list[PluginMetadata] = []
    for plugin_root in sorted(item for item in plugins_dir.iterdir() if item.is_dir()):
        plugin = load_plugin_metadata_from_directory(plugin_root)
        if plugin is not None:
            plugins.append(plugin)
    return plugins


def load_plugin_metadata_from_directory(plugin_root: Path) -> PluginMetadata | None:
    manifest = plugin_root / "resources" / "plugin.json"
    if not manifest.is_file():
        manifest = plugin_root / "plugin.json"
    payload: dict[str, Any] = {}
    if manifest.is_file():
        try:
            loaded = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            loaded = {}
        if isinstance(loaded, dict):
            payload.update(loaded)

    pyproject_payload = load_pyproject_metadata(plugin_root / "pyproject.toml")
    pyproject_project = pyproject_payload.get("project", {}) if isinstance(pyproject_payload.get("project"), dict) else {}
    plugin_id = str(payload.get("id") or pyproject_project.get("name") or plugin_root.name).strip()
    if not plugin_id:
        return None

    payload.setdefault("id", plugin_id)
    payload.setdefault("display_name", humanize_plugin_name(plugin_id))
    description = pyproject_project.get("description")
    version = pyproject_project.get("version")
    if isinstance(description, str) and description:
        payload.setdefault("description", description)
    if isinstance(version, str) and version:
        payload.setdefault("version", version)
    payload["source_dir"] = str(plugin_root)

    plugin = parse_plugin_metadata(payload)
    if plugin.version_history:
        return plugin
    return PluginMetadata(
        id=plugin.id,
        display_name=plugin.display_name,
        description=plugin.description,
        version=plugin.version,
        enabled=plugin.enabled,
        platforms=plugin.platforms,
        executables=plugin.executables,
        update_manifests=plugin.update_manifests,
        actions=plugin.actions,
        source_dir=plugin.source_dir,
        version_history=load_changelog_history(plugin_root),
    )


def load_pyproject_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def merge_plugin_sources(primary: list[PluginMetadata], fallback: list[PluginMetadata]) -> list[PluginMetadata]:
    merged: dict[str, PluginMetadata] = {plugin.id: plugin for plugin in fallback}
    for plugin in primary:
        previous = merged.get(plugin.id)
        merged[plugin.id] = merge_plugin_metadata(plugin, previous) if previous else plugin
    return sorted(merged.values(), key=lambda item: item.display_name.lower())


def merge_plugin_metadata(primary: PluginMetadata, fallback: PluginMetadata | None) -> PluginMetadata:
    if fallback is None:
        return primary
    return PluginMetadata(
        id=primary.id,
        display_name=primary.display_name or fallback.display_name,
        description=primary.description or fallback.description,
        version=primary.version if primary.version != "0.0.0" else fallback.version,
        enabled=primary.enabled,
        platforms=primary.platforms or fallback.platforms,
        executables=primary.executables or fallback.executables,
        update_manifests=primary.update_manifests or fallback.update_manifests,
        actions=primary.actions or fallback.actions,
        source_dir=primary.source_dir or fallback.source_dir,
        version_history=primary.version_history or fallback.version_history,
    )


def build_plugin_inventory(plugins: list[PluginMetadata]) -> list[PluginInventoryItem]:
    return [PluginInventoryItem(metadata=plugin, installed=is_plugin_installed(plugin)) for plugin in plugins]


def is_plugin_installed(plugin: PluginMetadata) -> bool:
    candidates = {plugin.id, normalize_distribution_name(plugin.id), normalize_distribution_name(plugin.display_name)}
    for candidate in candidates:
        if not candidate:
            continue
        try:
            importlib_metadata.version(candidate)
        except importlib_metadata.PackageNotFoundError:
            continue
        return True
    return False


def parse_version_history(payload: dict[str, Any]) -> tuple[PluginVersionEntry, ...]:
    raw_history = payload.get("version_history", payload.get("versions", ()))
    if not isinstance(raw_history, list):
        return ()
    entries: list[PluginVersionEntry] = []
    for item in raw_history:
        if isinstance(item, str):
            entries.append(PluginVersionEntry(version=item))
        elif isinstance(item, dict):
            version = str(item.get("version", "") or "").strip()
            if version:
                entries.append(PluginVersionEntry(version=version, notes=str(item.get("notes", "") or "").strip()))
    return tuple(entries)


def load_changelog_history(plugin_root: Path) -> tuple[PluginVersionEntry, ...]:
    for relative_path in (
        Path("resources") / "changelog.md",
        Path("resources") / "changelog_en.md",
        Path("resources") / "docs" / "CHANGELOG.md",
        Path("CHANGELOG.md"),
    ):
        changelog = plugin_root / relative_path
        if changelog.is_file():
            return parse_changelog(changelog)
    return ()


def parse_changelog(path: Path, *, limit: int = 12) -> tuple[PluginVersionEntry, ...]:
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return ()
    entries: list[PluginVersionEntry] = []
    current_version = ""
    current_notes: list[str] = []
    for line in lines:
        heading = re.match(r"^##\s+\[?([^\]\n]+?)\]?\s*(?:[-\u2014].*)?$", line.strip())
        if heading:
            if current_version:
                entries.append(PluginVersionEntry(version=current_version, notes="\n".join(current_notes).strip()))
                if len(entries) >= limit:
                    break
            current_version = heading.group(1).strip()
            current_notes = []
            continue
        if current_version and (line.startswith("- ") or line.startswith("* ") or line.startswith("### ")):
            current_notes.append(line.strip())
    if current_version and len(entries) < limit:
        entries.append(PluginVersionEntry(version=current_version, notes="\n".join(current_notes).strip()))
    return tuple(entries)


def normalize_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "").strip()).lower()


def humanize_plugin_name(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()
