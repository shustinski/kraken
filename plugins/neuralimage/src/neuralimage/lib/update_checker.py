from __future__ import annotations

from pathlib import Path

from neuralimage.lib.runtime_paths import resolve_resource_path
import updater.client as _client
from updater.client import (
    ReleaseInfo,
    UpdateClientConfig,
    UpdateInfo,
    collect_release_history,
    compare_versions,
    fetch_update_info,
    is_newer_version,
    load_selected_update_channel as _load_selected_update_channel,
    load_update_client_config as _load_update_client_config,
    normalize_update_channel,
    parse_update_payload,
    parse_version_parts,
    save_selected_update_channel as _save_selected_update_channel,
    should_notify_version,
)
from updater.client import download_update_installer as _download_update_installer
from updater.client import get_update_staging_dir as _get_update_staging_dir
from updater.client import launch_update_installer as _launch_update_installer
from updater.client import load_last_notified_version as _load_last_notified_version
from updater.client import load_update_manifest_url as _load_update_manifest_url
from updater.client import save_last_notified_version as _save_last_notified_version

_APP_ID = "NeuralImage"
_SETTINGS_APP = "Updater"

os = _client.os
subprocess = _client.subprocess
tempfile = _client.tempfile

__all__ = [
    "ReleaseInfo",
    "UpdateClientConfig",
    "UpdateInfo",
    "collect_release_history",
    "compare_versions",
    "download_update_installer",
    "fetch_update_info",
    "get_update_staging_dir",
    "is_newer_version",
    "launch_update_installer",
    "load_last_notified_version",
    "load_selected_update_channel",
    "load_update_client_config",
    "load_update_manifest_url",
    "normalize_update_channel",
    "parse_update_payload",
    "parse_version_parts",
    "save_last_notified_version",
    "save_selected_update_channel",
    "should_notify_version",
]


def _config_path() -> Path:
    return resolve_resource_path("update_client.json")


def load_update_client_config() -> UpdateClientConfig:
    return _load_update_client_config(app_id=_APP_ID, config_path=_config_path(), env_prefix="NEURALIMAGE")


def load_update_manifest_url(channel: str | None = None) -> str:
    return _load_update_manifest_url(
        channel,
        app_id=_APP_ID,
        config_path=_config_path(),
        env_prefix="NEURALIMAGE",
        settings_org=_APP_ID,
        settings_app=_SETTINGS_APP,
    )


def load_selected_update_channel(
    default_channel: str = "stable",
    *,
    available_channels: tuple[str, ...] | list[str] | None = None,
) -> str:
    return _load_selected_update_channel(
        default_channel,
        available_channels=available_channels,
        settings_org=_APP_ID,
        settings_app=_SETTINGS_APP,
    )


def save_selected_update_channel(channel: str) -> None:
    _save_selected_update_channel(channel, settings_org=_APP_ID, settings_app=_SETTINGS_APP)


def load_last_notified_version(channel: str | None = None) -> str:
    return _load_last_notified_version(channel, settings_org=_APP_ID, settings_app=_SETTINGS_APP)


def save_last_notified_version(version: str, channel: str | None = None) -> None:
    _save_last_notified_version(version, channel, settings_org=_APP_ID, settings_app=_SETTINGS_APP)


def download_update_installer(release_info: ReleaseInfo | UpdateInfo) -> Path:
    return _download_update_installer(release_info, app_id=_APP_ID)


def get_update_staging_dir() -> Path:
    return _get_update_staging_dir(_APP_ID)


def launch_update_installer(installer_path: str | Path) -> None:
    _launch_update_installer(installer_path)


def _resolve_installer_name(update_info: ReleaseInfo | UpdateInfo, source: str = "") -> str:
    return _client._resolve_installer_name(update_info, source, app_id=_APP_ID)  # type: ignore[attr-defined]
