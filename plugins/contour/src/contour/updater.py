from __future__ import annotations

from pathlib import Path

from updater.qt import QtUpdateController

from .__version__ import __version__

CONTOUR_UPDATE_APP_ID = "contour"
CONTOUR_UPDATE_APP_NAME = "Contour"
CONTOUR_UPDATE_ENV_PREFIX = "CONTOUR"
CONTOUR_UPDATE_SETTINGS_ORG = "Contour"
CONTOUR_UPDATE_CLIENT_FILENAME = "update_client.json"


def contour_update_client_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / CONTOUR_UPDATE_CLIENT_FILENAME


def create_contour_update_controller(window) -> QtUpdateController:
    return QtUpdateController(
        window,
        app_id=CONTOUR_UPDATE_APP_ID,
        app_name=CONTOUR_UPDATE_APP_NAME,
        current_version=__version__,
        config_path=contour_update_client_config_path(),
        env_prefix=CONTOUR_UPDATE_ENV_PREFIX,
        settings_org=CONTOUR_UPDATE_SETTINGS_ORG,
        status_callback=window.show_status_message,
    )
