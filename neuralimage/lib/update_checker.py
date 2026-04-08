from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse as urlparse, request

from lib.runtime_paths import resolve_resource_path


_VERSION_PART_RE = re.compile(r'\d+')
_URL_SCHEME_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.-]*://')
_UPDATE_SETTINGS_ORG = 'NeuralImage'
_UPDATE_SETTINGS_APP = 'Updater'
_UPDATE_SETTINGS_KEY = 'last_notified_version'
_UPDATE_CHANNEL_SETTINGS_KEY = 'selected_channel'
_UPDATE_STAGING_DIR_NAME = 'NeuralImageUpdater'
_UPDATE_CLEANUP_MAX_RETRIES = 300
_UPDATE_CLEANUP_DELAY_SECONDS = 2


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    download_url: str = ''
    notes: str = ''
    channel: str = ''


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    download_url: str = ''
    release_notes: str = ''
    mandatory: bool = False
    releases: tuple[ReleaseInfo, ...] = ()
    channel: str = 'stable'


@dataclass(frozen=True)
class UpdateClientConfig:
    manifest_urls: tuple[tuple[str, str], ...] = ()
    default_channel: str = 'stable'

    @property
    def available_channels(self) -> tuple[str, ...]:
        if self.manifest_urls:
            return tuple(channel for channel, _url in self.manifest_urls)
        return (self.default_channel,)

    def get_manifest_url(self, channel: str | None = None) -> str:
        requested = normalize_update_channel(channel or self.default_channel)
        manifest_map = dict(self.manifest_urls)
        if requested in manifest_map:
            return str(manifest_map[requested]).strip()
        if channel is not None:
            return ''
        if self.default_channel in manifest_map:
            return str(manifest_map[self.default_channel]).strip()
        for _resolved_channel, manifest_url in self.manifest_urls:
            if str(manifest_url).strip():
                return str(manifest_url).strip()
        return ''


def parse_version_parts(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _VERSION_PART_RE.findall(str(version)))


def compare_versions(left: str, right: str) -> int:
    left_parts = parse_version_parts(left)
    right_parts = parse_version_parts(right)
    max_len = max(len(left_parts), len(right_parts))
    left_padded = left_parts + (0,) * (max_len - len(left_parts))
    right_padded = right_parts + (0,) * (max_len - len(right_parts))
    if left_padded < right_padded:
        return -1
    if left_padded > right_padded:
        return 1
    return 0


def is_newer_version(candidate: str, current: str) -> bool:
    return compare_versions(candidate, current) > 0


def should_notify_version(candidate: str, current: str, last_notified: str) -> bool:
    if not is_newer_version(candidate, current):
        return False
    if not str(last_notified).strip():
        return True
    return is_newer_version(candidate, last_notified)


def normalize_update_channel(value: str | None) -> str:
    normalized = str(value or '').strip().lower()
    return normalized or 'stable'


def load_update_client_config() -> UpdateClientConfig:
    env_url = str(os.getenv('NEURALIMAGE_UPDATE_URL', '')).strip()
    if env_url:
        env_channel = normalize_update_channel(os.getenv('NEURALIMAGE_UPDATE_CHANNEL'))
        return UpdateClientConfig(
            manifest_urls=((env_channel, env_url),),
            default_channel=env_channel,
        )
    config_path = resolve_resource_path('update_client.json')
    if not config_path.exists():
        return UpdateClientConfig()
    try:
        payload = json.loads(config_path.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError):
        return UpdateClientConfig()
    if not isinstance(payload, dict):
        return UpdateClientConfig()

    manifest_urls: list[tuple[str, str]] = []
    raw_channels = payload.get('channels', payload.get('manifest_urls'))
    if isinstance(raw_channels, dict):
        for raw_channel, raw_url in raw_channels.items():
            url = str(raw_url or '').strip()
            if not url:
                continue
            manifest_urls.append((normalize_update_channel(str(raw_channel)), url))

    if not manifest_urls:
        manifest_url = str(payload.get('manifest_url', '')).strip()
        if manifest_url:
            default_channel = normalize_update_channel(
                str(payload.get('default_channel', payload.get('channel', 'stable')))
            )
            manifest_urls.append((default_channel, manifest_url))

    default_channel = normalize_update_channel(
        str(
            payload.get(
                'default_channel',
                manifest_urls[0][0] if manifest_urls else 'stable',
            )
        )
    )
    return UpdateClientConfig(
        manifest_urls=tuple(manifest_urls),
        default_channel=default_channel,
    )


def load_update_manifest_url(channel: str | None = None) -> str:
    config = load_update_client_config()
    selected_channel = load_selected_update_channel(
        config.default_channel,
        available_channels=config.available_channels,
    )
    return config.get_manifest_url(channel or selected_channel)


def parse_update_payload(payload: dict[str, Any], *, source: str = '') -> UpdateInfo | None:
    version = str(payload.get('version', '')).strip()
    if not version:
        return None
    channel = normalize_update_channel(str(payload.get('channel', 'stable')))
    top_level_download_url = str(payload.get('download_url', '')).strip()
    release_notes = _resolve_release_notes(payload, source=source)
    releases = _parse_release_entries(payload.get('releases'), source=source, default_channel=channel)
    if releases:
        normalized_releases: list[ReleaseInfo] = []
        for release in releases:
            download_url = release.download_url
            notes = release.notes
            if release.version == version and not download_url:
                download_url = top_level_download_url
            if release.version == version and not notes:
                notes = release_notes
            normalized_releases.append(
                ReleaseInfo(
                    version=release.version,
                    download_url=download_url,
                    notes=notes,
                    channel=release.channel or channel,
                )
            )
        releases = tuple(normalized_releases)
    if not releases:
        releases = (
            ReleaseInfo(
                version=version,
                download_url=top_level_download_url,
                notes=release_notes,
                channel=channel,
            ),
        )
    return UpdateInfo(
        version=version,
        download_url=top_level_download_url,
        release_notes=release_notes,
        mandatory=bool(payload.get('mandatory', False)),
        releases=releases,
        channel=channel,
    )


def fetch_update_info(
    manifest_url: str,
    timeout_seconds: float = 2.5,
    *,
    expected_channel: str | None = None,
) -> UpdateInfo | None:
    url = str(manifest_url).strip()
    if not url:
        return None
    try:
        if _is_filesystem_source(url):
            manifest_path = Path(url)
            if not manifest_path.is_file():
                return None
            payload = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        else:
            with request.urlopen(url, timeout=timeout_seconds) as response:
                charset = response.headers.get_content_charset() or 'utf-8'
                payload = json.loads(response.read().decode(charset))
    except (OSError, ValueError, error.URLError):
        return None
    if not isinstance(payload, dict):
        return None
    update_info = parse_update_payload(payload, source=url)
    if update_info is None:
        return None
    requested_channel = normalize_update_channel(expected_channel) if expected_channel is not None else ''
    if requested_channel and update_info.channel and update_info.channel != requested_channel:
        return None
    return update_info


def _last_notified_key(channel: str | None = None) -> str:
    if channel is None:
        return _UPDATE_SETTINGS_KEY
    normalized_channel = normalize_update_channel(channel)
    return f'{_UPDATE_SETTINGS_KEY}_{normalized_channel}'


def load_selected_update_channel(
    default_channel: str = 'stable',
    *,
    available_channels: tuple[str, ...] | list[str] | None = None,
) -> str:
    try:
        from PyQt6.QtCore import QSettings
    except ImportError:
        return normalize_update_channel(default_channel)
    settings = _create_settings(QSettings)
    value = settings.value(_UPDATE_CHANNEL_SETTINGS_KEY, default_channel, type=str)
    settings.sync()
    normalized = normalize_update_channel(str(value or default_channel))
    if available_channels:
        normalized_available = tuple(
            normalize_update_channel(channel) for channel in available_channels if str(channel).strip()
        )
        if normalized in normalized_available:
            return normalized
        fallback = normalize_update_channel(default_channel)
        if fallback in normalized_available:
            return fallback
        return normalized_available[0]
    return normalized


def save_selected_update_channel(channel: str) -> None:
    normalized = normalize_update_channel(channel)
    if not normalized:
        return
    try:
        from PyQt6.QtCore import QSettings
    except ImportError:
        return
    settings = _create_settings(QSettings)
    settings.setValue(_UPDATE_CHANNEL_SETTINGS_KEY, normalized)
    settings.sync()


def load_last_notified_version(channel: str | None = None) -> str:
    try:
        from PyQt6.QtCore import QSettings
    except ImportError:
        return ''
    settings = _create_settings(QSettings)
    value = settings.value(_last_notified_key(channel), '', type=str)
    if not str(value or '').strip() and channel:
        value = settings.value(_UPDATE_SETTINGS_KEY, '', type=str)
    settings.sync()
    return str(value or '').strip()


def save_last_notified_version(version: str, channel: str | None = None) -> None:
    normalized = str(version or '').strip()
    if not normalized:
        return
    try:
        from PyQt6.QtCore import QSettings
    except ImportError:
        return
    settings = _create_settings(QSettings)
    settings.setValue(_last_notified_key(channel), normalized)
    settings.sync()


def download_update_installer(release_info: ReleaseInfo | UpdateInfo) -> Path:
    if not release_info.download_url:
        raise ValueError('Update manifest does not contain download_url.')
    target_dir = get_update_staging_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    source = str(release_info.download_url).strip()
    target_path = target_dir / _resolve_installer_name(release_info, source)
    if _is_filesystem_source(source):
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f'Installer file not found: {source_path}')
        shutil.copyfile(source_path, target_path)
        return target_path
    with request.urlopen(source, timeout=30.0) as response:
        with target_path.open('wb') as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
    return target_path


def get_update_staging_dir() -> Path:
    return Path(tempfile.gettempdir()) / _UPDATE_STAGING_DIR_NAME


def launch_update_installer(installer_path: str | Path) -> None:
    installer = Path(installer_path)
    if os.name != 'nt':
        subprocess.Popen([str(installer)], close_fds=True)
        return
    launcher_path = _write_update_launcher_script()
    creationflags = int(getattr(subprocess, 'DETACHED_PROCESS', 0)) | int(
        getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    )
    subprocess.Popen(
        ['cmd.exe', '/d', '/c', str(launcher_path), str(installer), str(installer.parent)],
        close_fds=True,
        creationflags=creationflags,
    )


def collect_release_history(update_info: UpdateInfo) -> str:
    relevant_entries = list(update_info.releases)
    if not relevant_entries:
        return update_info.release_notes
    chunks: list[str] = []
    for release in relevant_entries:
        version_label = str(release.version).strip()
        notes = str(release.notes).strip()
        if notes:
            chunks.append(f'### {version_label}\n\n{notes}')
        else:
            chunks.append(f'### {version_label}')
    return '\n\n'.join(chunks)


def _resolve_installer_name(update_info: ReleaseInfo | UpdateInfo, source: str = '') -> str:
    source_name = Path(source).name if _is_filesystem_source(source) else Path(source).name
    if source_name.lower().endswith('.exe'):
        return source_name
    version = re.sub(r'[^0-9A-Za-z._-]+', '_', update_info.version.strip() or 'latest')
    return f'NeuralImage-{version}.exe'


def _is_filesystem_source(value: str) -> bool:
    normalized = str(value or '').strip()
    if not normalized:
        return False
    return _URL_SCHEME_RE.match(normalized) is None


def _resolve_release_notes(payload: dict[str, Any], *, source: str = '') -> str:
    inline_markdown = str(
        payload.get(
            'release_notes_markdown',
            payload.get('notes_markdown', payload.get('changes_markdown', '')),
        )
        or ''
    ).strip()
    if inline_markdown:
        return inline_markdown
    for key in ('release_notes', 'notes', 'changes'):
        value = str(payload.get(key, '') or '').strip()
        if value:
            return _resolve_text_payload(value, source=source)
    for key in ('release_notes_path', 'notes_path', 'changes_path'):
        value = str(payload.get(key, '') or '').strip()
        if value:
            return _resolve_text_reference(value, source=source)
    return ''


def _parse_release_entries(
    raw_value: Any,
    *,
    source: str = '',
    default_channel: str = 'stable',
) -> tuple[ReleaseInfo, ...]:
    if not isinstance(raw_value, list):
        return ()
    entries: list[ReleaseInfo] = []
    for item in raw_value:
        if not isinstance(item, dict):
            continue
        version = str(item.get('version', '')).strip()
        if not version:
            continue
        download_url = str(item.get('download_url', '')).strip()
        notes = _resolve_release_notes(item, source=source)
        channel = normalize_update_channel(str(item.get('channel', default_channel)))
        entries.append(
            ReleaseInfo(
                version=version,
                download_url=download_url,
                notes=notes,
                channel=channel,
            )
        )
    return tuple(entries)


def _resolve_text_payload(value: str, *, source: str = '') -> str:
    normalized = str(value or '').strip()
    if not normalized:
        return ''
    if normalized.lower().endswith(('.md', '.markdown')):
        return _resolve_text_reference(normalized, source=source)
    return normalized


def _resolve_text_reference(reference: str, *, source: str = '') -> str:
    resolved_reference = _resolve_reference_path(reference, source=source)
    if not resolved_reference:
        return str(reference or '').strip()
    try:
        if _is_filesystem_source(resolved_reference):
            path = Path(resolved_reference)
            if not path.is_file():
                return str(reference or '').strip()
            return path.read_text(encoding='utf-8-sig')
        with request.urlopen(resolved_reference, timeout=10.0) as response:
            charset = response.headers.get_content_charset() or 'utf-8'
            return response.read().decode(charset)
    except (OSError, error.URLError, UnicodeDecodeError):
        return str(reference or '').strip()


def _resolve_reference_path(reference: str, *, source: str = '') -> str:
    normalized_reference = str(reference or '').strip()
    if not normalized_reference:
        return ''
    if not source:
        return normalized_reference
    if _is_filesystem_source(normalized_reference):
        reference_path = Path(normalized_reference)
        if reference_path.is_absolute():
            return str(reference_path)
        if _is_filesystem_source(source):
            return str((Path(source).resolve().parent / reference_path).resolve())
        return normalized_reference
    if _is_filesystem_source(source):
        return normalized_reference
    return urlparse.urljoin(source, normalized_reference)


def _write_update_launcher_script() -> Path:
    launcher_path = Path(tempfile.gettempdir()) / f'neuralimage_update_cleanup_{uuid.uuid4().hex}.cmd'
    launcher_path.write_text(
        '\n'.join(
            (
                '@echo off',
                'setlocal',
                '"%~1"',
                'set /a RETRIES=0',
                ':retry_cleanup',
                'rmdir /s /q "%~2" >nul 2>&1',
                'if exist "%~2" (',
                '    set /a RETRIES+=1',
                f'    if %RETRIES% LSS {_UPDATE_CLEANUP_MAX_RETRIES} (',
                f'        timeout /t {_UPDATE_CLEANUP_DELAY_SECONDS} /nobreak >nul',
                '        goto retry_cleanup',
                '    )',
                ')',
                '(goto) 2>nul & del "%~f0"',
            )
        ),
        encoding='ascii',
    )
    return launcher_path


def _create_settings(qsettings_cls):
    settings_root = str(os.getenv('NEURALIMAGE_SETTINGS_DIR', '')).strip()
    if settings_root:
        path = Path(settings_root)
        path.mkdir(parents=True, exist_ok=True)
        return qsettings_cls(
            str(path / f'{_UPDATE_SETTINGS_ORG}_{_UPDATE_SETTINGS_APP}.ini'),
            qsettings_cls.Format.IniFormat,
        )
    return qsettings_cls(_UPDATE_SETTINGS_ORG, _UPDATE_SETTINGS_APP)
