from __future__ import annotations

import sys
from pathlib import Path

from updater.client import (
    UpdateClientConfig,
    load_selected_update_channel,
    load_update_client_config,
    load_update_manifest_url,
    save_selected_update_channel,
)
from updater.qt import QtUpdateController

from .__version__ import __version__

CONTOUR_UPDATE_APP_ID = "Contour"
CONTOUR_UPDATE_APP_NAME = "Contour"
CONTOUR_UPDATE_ENV_PREFIX = "CONTOUR"
CONTOUR_UPDATE_SETTINGS_ORG = "Contour"
CONTOUR_UPDATE_SETTINGS_APP = "Updater"
CONTOUR_UPDATE_CLIENT_FILENAME = "update_client.json"


def contour_resources_root() -> Path:
    if not bool(getattr(sys, "frozen", False)):
        return Path(__file__).resolve().parents[2] / "resources"

    executable_dir = Path(sys.executable).resolve().parent
    internal_dir = executable_dir / "_internal"
    if internal_dir.exists():
        return internal_dir / "resources"

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "resources"

    return executable_dir / "resources"


def contour_update_client_config_path() -> Path:
    return contour_resources_root() / CONTOUR_UPDATE_CLIENT_FILENAME


def load_contour_update_client_config() -> UpdateClientConfig:
    return load_update_client_config(
        app_id=CONTOUR_UPDATE_APP_ID,
        config_path=contour_update_client_config_path(),
        env_prefix=CONTOUR_UPDATE_ENV_PREFIX,
    )


def load_contour_update_manifest_url(channel: str | None = None) -> str:
    return load_update_manifest_url(
        channel,
        app_id=CONTOUR_UPDATE_APP_ID,
        config_path=contour_update_client_config_path(),
        env_prefix=CONTOUR_UPDATE_ENV_PREFIX,
        settings_org=CONTOUR_UPDATE_SETTINGS_ORG,
        settings_app=CONTOUR_UPDATE_SETTINGS_APP,
    )


def load_contour_update_channel(config: UpdateClientConfig | None = None) -> str:
    resolved_config = config or load_contour_update_client_config()
    return load_selected_update_channel(
        resolved_config.default_channel,
        available_channels=resolved_config.available_channels,
        settings_org=CONTOUR_UPDATE_SETTINGS_ORG,
        settings_app=CONTOUR_UPDATE_SETTINGS_APP,
    )


def save_contour_update_channel(channel: str) -> None:
    save_selected_update_channel(
        channel,
        settings_org=CONTOUR_UPDATE_SETTINGS_ORG,
        settings_app=CONTOUR_UPDATE_SETTINGS_APP,
    )


def create_contour_update_controller(window) -> QtUpdateController:
    return QtUpdateController(
        window,
        app_id=CONTOUR_UPDATE_APP_ID,
        app_name=CONTOUR_UPDATE_APP_NAME,
        current_version=__version__,
        config_path=contour_update_client_config_path(),
        env_prefix=CONTOUR_UPDATE_ENV_PREFIX,
        settings_org=CONTOUR_UPDATE_SETTINGS_ORG,
        settings_app=CONTOUR_UPDATE_SETTINGS_APP,
        status_callback=window.show_status_message,
    )
