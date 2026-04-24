from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse as urlparse, request

from .runtime import current_platform

_VERSION_PART_RE = re.compile(r"\d+")
_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    download_url: str = ""
    notes: str = ""
    channel: str = "stable"
    platform: str = ""


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    download_url: str = ""
    release_notes: str = ""
    mandatory: bool = False
    releases: tuple[ReleaseInfo, ...] = ()
    channel: str = "stable"


def parse_version_parts(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _VERSION_PART_RE.findall(str(version)))


def compare_versions(left: str, right: str) -> int:
    left_parts = parse_version_parts(left)
    right_parts = parse_version_parts(right)
    max_len = max(len(left_parts), len(right_parts))
    left_padded = left_parts + (0,) * (max_len - len(left_parts))
    right_padded = right_parts + (0,) * (max_len - len(right_parts))
    return (left_padded > right_padded) - (left_padded < right_padded)


def is_newer_version(candidate: str, current: str) -> bool:
    return compare_versions(candidate, current) > 0


def parse_update_payload(payload: dict[str, Any], *, source: str = "") -> UpdateInfo | None:
    version = str(payload.get("version", "")).strip()
    if not version:
        return None
    channel = str(payload.get("channel", "stable")).strip().lower() or "stable"
    release_notes = str(payload.get("release_notes", payload.get("notes", "")) or "").strip()
    top_download_url = str(payload.get("download_url", "") or "").strip()
    releases: list[ReleaseInfo] = []
    for item in payload.get("releases", []) if isinstance(payload.get("releases"), list) else []:
        if not isinstance(item, dict):
            continue
        item_version = str(item.get("version", "")).strip()
        if not item_version:
            continue
        releases.append(
            ReleaseInfo(
                version=item_version,
                download_url=str(item.get("download_url", "") or "").strip(),
                notes=str(item.get("notes", item.get("release_notes", "")) or "").strip(),
                channel=str(item.get("channel", channel) or channel).strip().lower(),
                platform=str(item.get("platform", "") or "").strip().lower(),
            )
        )
    if not releases:
        releases.append(ReleaseInfo(version=version, download_url=top_download_url, notes=release_notes, channel=channel))
    return UpdateInfo(
        version=version,
        download_url=top_download_url,
        release_notes=release_notes,
        mandatory=bool(payload.get("mandatory", False)),
        releases=tuple(releases),
        channel=channel,
    )


def fetch_update_info(manifest_url: str, timeout_seconds: float = 5.0) -> UpdateInfo | None:
    source = str(manifest_url).strip()
    if not source:
        return None
    try:
        if is_filesystem_source(source):
            path = Path(source)
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        else:
            with request.urlopen(source, timeout=timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                payload = json.loads(response.read().decode(charset))
    except (OSError, ValueError, error.URLError):
        return None
    return parse_update_payload(payload, source=source) if isinstance(payload, dict) else None


def select_platform_release(update_info: UpdateInfo, platform: str | None = None) -> ReleaseInfo | None:
    requested = (platform or current_platform()).lower()
    exact = [release for release in update_info.releases if release.platform == requested]
    candidates = exact or [release for release in update_info.releases if not release.platform]
    if not candidates:
        return None
    for release in candidates:
        if release.version == update_info.version:
            return release
    return candidates[0]


def download_update_installer(release: ReleaseInfo, *, app_id: str) -> Path:
    if not release.download_url:
        raise ValueError("Update release does not contain download_url.")
    target_dir = Path(tempfile.gettempdir()) / "KrakenUpdater" / sanitize_filename(app_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    source = release.download_url
    target_path = target_dir / resolve_installer_name(release, source)
    if is_filesystem_source(source):
        shutil.copyfile(Path(source), target_path)
    else:
        with request.urlopen(source, timeout=30.0) as response:
            with target_path.open("wb") as file:
                shutil.copyfileobj(response, file)
    ensure_posix_executable(target_path)
    return target_path


def launch_installer(path: str | Path) -> None:
    installer = Path(path)
    if os.name == "nt":
        subprocess.Popen([str(installer)], close_fds=True)
    else:
        ensure_posix_executable(installer)
        subprocess.Popen([str(installer)], close_fds=True)


def resolve_installer_name(release: ReleaseInfo, source: str = "") -> str:
    parsed = urlparse.urlparse(source)
    name = Path(parsed.path if parsed.scheme else source).name
    if name:
        return name
    suffix = ".exe" if current_platform() == "windows" else ""
    return f"{sanitize_filename(release.version or 'latest')}{suffix}"


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "kraken")).strip("_") or "kraken"


def is_filesystem_source(value: str) -> bool:
    normalized = str(value or "").strip()
    return bool(normalized) and _URL_SCHEME_RE.match(normalized) is None


def ensure_posix_executable(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        return
