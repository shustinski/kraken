"""manager_app: connect program account."""
from __future__ import annotations
from kraken_hub.dual_catalog import DualCatalogService

from kraken_hub.manager_app_parts.connect_saved_server import (
    _configure_administration_page,
    _sign_in_program_account,
)


class ConnectProgramAccountMixin:
    def open_preferences(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        from kraken_hub.preferences_dialog import ProgramPreferencesDialog

        dialog = ProgramPreferencesDialog(self.shell)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            self._connect_program_account()
        except Exception as exc:  # noqa: BLE001 - UI boundary reports connection failures
            QMessageBox.warning(self.shell, "Не удалось подключиться к серверу", str(exc))
            return
        _configure_administration_page(self.shell, self.service, self.session)
        self.refresh_projects()

    def _connect_program_account(self) -> None:
        from PyQt6.QtCore import QSettings

        from kraken_hub.preferences_dialog import default_program_account, program_server_url

        if not isinstance(self.service, DualCatalogService):
            raise ValueError("Подключение к Kraken Server недоступно в этой сборке")
        settings = QSettings("Kraken", "KrakenHub")
        account = default_program_account(settings)
        url = program_server_url(settings)
        if account is None or not url:
            raise ValueError(
                "Укажите адрес сервера и учётную запись по умолчанию в меню Вид → Настройки."
            )
        current = self.service.remote
        current_principal = getattr(getattr(current, "auth", None), "principal", None)
        same_account = (
            current is not None
            and str(getattr(current, "base_url", "")).rstrip("/") == url
            and str(getattr(current_principal, "subject", "")) == account.username
        )
        if same_account:
            return
        _sign_in_program_account(
            self.service,
            url,
            account,
            principal=self.session.principal,
        )
