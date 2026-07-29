from __future__ import annotations

from pathlib import Path

from updater.client import (
    UpdateClientConfig,
    load_selected_update_channel,
    load_update_client_config,
    save_selected_update_channel,
)
from updater.qt import QtUpdateController

from .version import __version__

KARAKAL_UPDATE_APP_ID = "karakal"
KARAKAL_UPDATE_APP_NAME = "Karakal"
KARAKAL_UPDATE_ENV_PREFIX = "KARAKAL"
KARAKAL_UPDATE_SETTINGS_ORG = "Karakal"
KARAKAL_UPDATE_SETTINGS_APP = "Updater"
KARAKAL_UPDATE_CLIENT_FILENAME = "update_client.json"


def karakal_update_client_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / KARAKAL_UPDATE_CLIENT_FILENAME


def load_karakal_update_client_config() -> UpdateClientConfig:
    return load_update_client_config(
        app_id=KARAKAL_UPDATE_APP_ID,
        config_path=karakal_update_client_config_path(),
        env_prefix=KARAKAL_UPDATE_ENV_PREFIX,
    )


def load_karakal_update_channel(config: UpdateClientConfig | None = None) -> str:
    resolved_config = config or load_karakal_update_client_config()
    return load_selected_update_channel(
        resolved_config.default_channel,
        available_channels=resolved_config.available_channels,
        settings_org=KARAKAL_UPDATE_SETTINGS_ORG,
        settings_app=KARAKAL_UPDATE_SETTINGS_APP,
    )


def save_karakal_update_channel(channel: str) -> None:
    save_selected_update_channel(
        channel,
        settings_org=KARAKAL_UPDATE_SETTINGS_ORG,
        settings_app=KARAKAL_UPDATE_SETTINGS_APP,
    )


def create_karakal_update_controller(window) -> QtUpdateController:
    return QtUpdateController(
        window,
        app_id=KARAKAL_UPDATE_APP_ID,
        app_name=KARAKAL_UPDATE_APP_NAME,
        current_version=__version__,
        config_path=karakal_update_client_config_path(),
        env_prefix=KARAKAL_UPDATE_ENV_PREFIX,
        settings_org=KARAKAL_UPDATE_SETTINGS_ORG,
        settings_app=KARAKAL_UPDATE_SETTINGS_APP,
    )
