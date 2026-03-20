from __future__ import annotations

import json
import os
import re
import tempfile
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


_VERSION_PART_RE = re.compile(r'\d+')
_UPDATE_SETTINGS_ORG = 'NeuralImage'
_UPDATE_SETTINGS_APP = 'Updater'
_UPDATE_SETTINGS_KEY = 'last_notified_version'


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    download_url: str = ''
    sha256: str = ''
    release_notes: str = ''
    mandatory: bool = False


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


def load_update_manifest_url() -> str:
    env_url = str(os.getenv('NEURALIMAGE_UPDATE_URL', '')).strip()
    if env_url:
        return env_url
    config_path = Path(__file__).resolve().parent.parent / 'resources' / 'update_client.json'
    if not config_path.exists():
        return ''
    try:
        payload = json.loads(config_path.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError):
        return ''
    if not isinstance(payload, dict):
        return ''
    return str(payload.get('manifest_url', '')).strip()


def parse_update_payload(payload: dict[str, Any]) -> UpdateInfo | None:
    version = str(payload.get('version', '')).strip()
    if not version:
        return None
    return UpdateInfo(
        version=version,
        download_url=str(payload.get('download_url', '')).strip(),
        sha256=_normalize_sha256(payload.get('sha256', '')),
        release_notes=str(payload.get('release_notes', '')).strip(),
        mandatory=bool(payload.get('mandatory', False)),
    )


def fetch_update_info(manifest_url: str, timeout_seconds: float = 2.5) -> UpdateInfo | None:
    url = str(manifest_url).strip()
    if not url:
        return None
    try:
        with request.urlopen(url, timeout=timeout_seconds) as response:
            charset = response.headers.get_content_charset() or 'utf-8'
            payload = json.loads(response.read().decode(charset))
    except (OSError, ValueError, error.URLError):
        return None
    if not isinstance(payload, dict):
        return None
    return parse_update_payload(payload)


def load_last_notified_version() -> str:
    try:
        from PyQt6.QtCore import QSettings
    except ImportError:
        return ''
    settings = _create_settings(QSettings)
    value = settings.value(_UPDATE_SETTINGS_KEY, '', type=str)
    settings.sync()
    return str(value or '').strip()


def save_last_notified_version(version: str) -> None:
    normalized = str(version or '').strip()
    if not normalized:
        return
    try:
        from PyQt6.QtCore import QSettings
    except ImportError:
        return
    settings = _create_settings(QSettings)
    settings.setValue(_UPDATE_SETTINGS_KEY, normalized)
    settings.sync()


def download_update_installer(update_info: UpdateInfo) -> Path:
    if not update_info.download_url:
        raise ValueError('Update manifest does not contain download_url.')
    target_dir = Path(tempfile.gettempdir()) / 'NeuralImageUpdater'
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / _resolve_installer_name(update_info)
    with request.urlopen(update_info.download_url, timeout=30.0) as response:
        with target_path.open('wb') as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
    return target_path


def validate_downloaded_installer(path: Path, expected_sha256: str) -> bool:
    expected = _normalize_sha256(expected_sha256)
    if not expected:
        return True
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().lower() == expected


def _resolve_installer_name(update_info: UpdateInfo) -> str:
    version = re.sub(r'[^0-9A-Za-z._-]+', '_', update_info.version.strip() or 'latest')
    return f'NeuralImage-{version}.exe'


def _normalize_sha256(value: Any) -> str:
    raw = str(value or '').strip().lower()
    if len(raw) != 64:
        return ''
    if any(ch not in '0123456789abcdef' for ch in raw):
        return ''
    return raw


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
