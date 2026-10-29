"""manager_app: run manager gui."""

from __future__ import annotations
import os
from kraken_core.plugins import PluginInventoryItem
from kraken_core.qt import configure_application_identity
from kraken_core.theme import THEME_SYSTEM, apply_app_theme
from kraken_manager.presentation.qt import (
    ProjectManagerShell,
)
from kraken_hub.composition.catalog import DualCatalogService, build_workspace_service

from kraken_hub.presentation.qt.parts.connect_saved_server import (
    _configure_administration_page,
    _connect_saved_server,
)

from kraken_hub.presentation.qt.parts.deletion_reason_dialog import _plugin_panel

from kraken_hub.presentation.qt.parts.development_session import _development_session

from kraken_hub.presentation.qt.parts.login import _login

from kraken_hub.presentation.qt.parts.my_work_panel import _my_work_panel

from kraken_hub.presentation.qt.parts.performer_panel import _performer_panel

from kraken_hub.presentation.qt.parts.statistics_panel import _statistics_panel


def run_manager_gui(
    items: list[PluginInventoryItem],
    *,
    update_url: str = "",
    thumbnail_store_uri: str = "",
) -> int:
    from PyQt6.QtCore import QTimer
    from kraken_core.settings import QSettings
    from PyQt6.QtWidgets import QApplication, QInputDialog

    from kraken_core.update_ui import QtUpdateController
    from kraken_core.updater import UpdateService
    from kraken_hub.composition.app import application_version

    app = QApplication.instance() or QApplication([])
    configure_application_identity(app, app_id="Kraken.ProjectManager", icon_name="kraken")
    apply_app_theme(str(QSettings("Kraken", "KrakenHub").value("appearance/theme", THEME_SYSTEM)), app)
    service = build_workspace_service()
    local = service.local if isinstance(service, DualCatalogService) else service
    session = _development_session(local) or _login(None, local)
    if session is None:
        return 1
    _connect_saved_server(None, service, session.principal)
    if isinstance(service, DualCatalogService) and service.remote is not None:
        # Shared mutations use the server principal carried by the remote client.
        session_summary = f"{session.principal.display_name}\nЛокальная сессия + сервер {service.remote.base_url}"
    else:
        session_summary = f"{session.principal.display_name}\nЛокальная сессия"
    shell = ProjectManagerShell()
    shell.set_session_summary(session_summary)
    from kraken_hub.presentation.qt.manager_app import DesktopController

    shell._desktop_controller = DesktopController(  # keep Qt slots alive
        shell,
        service,
        session,
        thumbnail_store_uri=thumbnail_store_uri,
        plugin_items=items,
    )
    controller = shell._desktop_controller
    plugins_page = shell.page("plugins")
    if plugins_page is not None:
        plugins_page.set_content(_plugin_panel(items))
    performers_page = shell.page("performers")
    if performers_page is not None:
        performers_page.set_content(_performer_panel(service))
    my_work_page = shell.page("my_work")
    if my_work_page is not None:
        my_work_page.set_content(_my_work_panel(service, session, controller))
    statistics_page = shell.page("statistics")
    if statistics_page is not None:
        statistics_page.set_content(
            _statistics_panel(
                service, session.principal, background_workers=controller._background_workers, async_reads=True
            )
        )
    _configure_administration_page(shell, service, session)
    settings = QSettings("Kraken", "KrakenHub")
    configured_update_url = update_url or os.getenv("KRAKEN_UPDATE_URL", "") or str(settings.value("updates/url", ""))
    update_service = UpdateService(
        configured_update_url,
        current_version=application_version(),
        app_id="kraken-hub",
    )
    update_controller = QtUpdateController(shell, update_service, application_name="Kraken Hub")
    shell._update_controller = update_controller

    def configure_update_source() -> None:
        value, accepted = QInputDialog.getText(
            shell,
            "Источник обновлений",
            "URL или локальный путь манифеста:",
            text=update_service.manifest_url,
        )
        if not accepted:
            return
        update_service.manifest_url = value.strip()
        settings.setValue("updates/url", update_service.manifest_url)
        shell.statusBar().showMessage(
            "Источник обновлений сохранён" if update_service.manifest_url else "Источник обновлений не задан",
            8000,
        )

    from PyQt6.QtGui import QAction

    source_action = QAction("Источник обновлений…", shell)
    source_action.triggered.connect(configure_update_source)
    check_action = QAction("Проверить обновления", shell)
    check_action.triggered.connect(lambda: update_controller.check(include_dismissed=True))
    shell.view_menu.addSeparator()
    shell.view_menu.addAction(source_action)
    shell.view_menu.addAction(check_action)
    if configured_update_url:
        QTimer.singleShot(0, lambda: update_controller.check(quiet=True))
    shell.show()
    return app.exec()
