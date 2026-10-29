"""manager_app: connect saved server."""
from __future__ import annotations
from kraken_hub.composition import DesktopSession
from kraken_hub.dual_catalog import DualCatalogService

from kraken_hub.manager_app_parts.administration_panel import _administration_panel

def _configure_administration_page(shell, service, session: DesktopSession) -> None:
    local = service.local if isinstance(service, DualCatalogService) else service
    page = shell.page("administration")
    shell.set_page_visible("administration", True)
    if page is not None:
        page.set_content(_administration_panel(local, session))

def _server_project_hint(settings) -> str:
    from kraken_hub.preferences_dialog import (
        PROVIDER_LABELS,
        default_program_account,
        program_server_url,
    )

    url = program_server_url(settings)
    account = default_program_account(settings)
    if url and account is not None:
        kind = PROVIDER_LABELS.get(account.provider, account.provider)
        return (
            f"Сервер: {url}\n"
            f"Учётная запись по умолчанию: {account.display_name} ({kind}).\n"
            "Если такой записи ещё нет на сервере, она будет создана. "
            "Новая запись не является администратором: без назначенных ролей видны только свои проекты."
        )
    return "Адрес сервера и учётная запись задаются в меню Вид → Настройки."

def _sign_in_program_account(service, url: str, account, *, principal):
    if account.provider == "local":
        configure_local = getattr(service, "configure_remote_local", None)
        if not callable(configure_local):
            raise ValueError("Вход в Kraken Server недоступен в этой сборке")
        configure_local(
            base_url=url,
            username=account.username,
            password=account.secret,
        )
        return
    configure = getattr(service, "configure_remote", None)
    if not callable(configure):
        raise ValueError("Настройка Kraken Server недоступна в этой сборке")
    configure(base_url=url, access_token=account.secret, principal=principal)

def _connect_saved_server(parent, service, principal) -> None:
    from kraken_core.settings import QSettings
    from PyQt6.QtWidgets import QMessageBox

    from kraken_hub.preferences_dialog import default_program_account, program_server_url
    from kraken_hub.infrastructure.remote.client import RemoteServerError

    if not isinstance(service, DualCatalogService) or service.remote is not None:
        return
    settings = QSettings("Kraken", "KrakenHub")
    account = default_program_account(settings)
    url = program_server_url(settings)
    if account is None or not url:
        return
    try:
        _sign_in_program_account(
            service,
            url,
            account,
            principal=principal,
        )
    except (OSError, RemoteServerError, ValueError) as exc:
        QMessageBox.warning(
            parent,
            "Не удалось подключиться к серверу",
            f"Сохранённое подключение к {url} не восстановлено.\n{exc}",
        )
