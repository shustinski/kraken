"""Local Qt console for Kraken Server administration."""

from __future__ import annotations

import sys
from pathlib import Path

from kraken_server.configuration import default_local_config_path
from kraken_admin.local_admin import (
    account_rows,
    audit_rows,
    connect_local_admin,
    create_account,
    maintainer_rows,
    project_rows,
    reset_password,
    revoke_maintainer,
    revoke_sessions,
    set_account_enabled,
    set_server_admin,
)

_AUDIT_LABELS = {
    "account.created": "Создан пользователь",
    "account.enabled": "Пользователь включён",
    "account.disabled": "Пользователь отключён",
    "account.password_reset": "Сброшен пароль",
    "sessions.revoked": "Завершены сеансы",
    "global_role.granted": "Назначен администратор",
    "global_role.revoked": "Снят администратор",
}


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("Для окна администратора установите PyQt6: uv sync --extra desktop", file=sys.stderr)
        return 2
    from kraken_core.qt import configure_application_identity
    from kraken_core.styles import load_shared_stylesheet

    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("Kraken Admin")
    configure_application_identity(application, app_id="kraken-admin")
    application.setStyleSheet(load_shared_stylesheet())
    window = AdminWindow()
    window.open_config(default_local_config_path())
    window.show()
    return int(application.exec())


class AdminWindow:
    """Built lazily so tests can construct the widgets after QApplication exists."""

    def __init__(self) -> None:
        from PyQt6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QListWidget,
            QPushButton,
            QSplitter,
            QTabWidget,
            QTableWidget,
            QVBoxLayout,
            QWidget,
        )

        self._accounts = None
        self._services = None
        self._config_path: Path | None = None
        self._filling = False

        self.window = QWidget()
        self.window.setObjectName("krakenAdminWindow")
        self.window.setWindowTitle("Kraken Admin")
        self.window.resize(960, 640)
        layout = QVBoxLayout(self.window)

        header = QHBoxLayout()
        self.status = QLabel("Конфигурация не открыта")
        self.status.setObjectName("adminConfigStatus")
        open_button = QPushButton("Открыть server.toml…")
        open_button.clicked.connect(self._choose_config)
        refresh_button = QPushButton("Обновить")
        refresh_button.clicked.connect(self.reload)
        header.addWidget(self.status, 1)
        header.addWidget(open_button)
        header.addWidget(refresh_button)
        layout.addLayout(header)

        tabs = QTabWidget(self.window)
        tabs.addTab(self._build_accounts_tab(), "Учётные записи")
        tabs.addTab(self._build_projects_tab(), "Проекты")
        tabs.addTab(self._build_audit_tab(), "Журнал")
        layout.addWidget(tabs, 1)

    def show(self) -> None:
        self.window.show()

    def open_config(self, path: Path) -> None:
        if not path.is_file():
            self.status.setText(f"Нет файла {path}")
            return
        try:
            accounts, services = connect_local_admin(path)
        except Exception as exc:  # noqa: BLE001 - UI reports the connection failure
            self._report("Не удалось подключиться к базе", exc)
            return
        self._accounts = accounts
        self._services = services
        self._config_path = path
        self.status.setText(str(path))
        self.reload()

    def reload(self) -> None:
        self._fill_accounts()
        self._fill_projects()
        self._fill_audit()

    def _build_accounts_tab(self):
        from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QTableWidget, QVBoxLayout, QWidget

        page = QWidget()
        layout = QVBoxLayout(page)
        self.accounts = QTableWidget(0, 5, page)
        self.accounts.setObjectName("adminAccountsTable")
        self.accounts.setHorizontalHeaderLabels(("Пользователь", "Логин", "Состояние", "Администратор", "Создан"))
        self.accounts.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.accounts.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.accounts.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.accounts, 1)
        actions = QHBoxLayout()
        buttons = (
            ("Создать…", self._create_account),
            ("Включить", lambda: self._set_enabled(True)),
            ("Отключить", lambda: self._set_enabled(False)),
            ("Назначить администратора", lambda: self._set_admin(True)),
            ("Снять администратора", lambda: self._set_admin(False)),
            ("Новый пароль…", self._reset_password),
            ("Завершить сеансы", self._revoke_sessions),
        )
        for title, slot in buttons:
            button = QPushButton(title)
            button.clicked.connect(slot)
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return page

    def _build_projects_tab(self):
        from PyQt6.QtWidgets import QListWidget, QPushButton, QSplitter, QTableWidget, QVBoxLayout, QWidget

        page = QWidget()
        splitter = QSplitter(page)
        outer = QVBoxLayout(page)
        outer.addWidget(splitter, 1)
        self.projects = QListWidget()
        self.projects.setObjectName("adminProjectsList")
        self.maintainers = QTableWidget(0, 2)
        self.maintainers.setObjectName("adminMaintainersTable")
        self.maintainers.setHorizontalHeaderLabels(("Логин", "Пользователь"))
        self.maintainers.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.maintainers.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.maintainers.horizontalHeader().setStretchLastSection(True)
        self.projects.currentRowChanged.connect(lambda _row: self._fill_maintainers())
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.maintainers, 1)
        revoke = QPushButton("Снять maintainer")
        revoke.clicked.connect(self._revoke_maintainer)
        side_layout.addWidget(revoke)
        splitter.addWidget(self.projects)
        splitter.addWidget(side)
        splitter.setStretchFactor(1, 1)
        return page

    def _build_audit_tab(self):
        from PyQt6.QtWidgets import QTableWidget, QVBoxLayout, QWidget

        page = QWidget()
        layout = QVBoxLayout(page)
        self.audit = QTableWidget(0, 3, page)
        self.audit.setObjectName("adminAuditTable")
        self.audit.setHorizontalHeaderLabels(("Время", "Действие", "Пользователь"))
        self.audit.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.audit, 1)
        return page

    def _choose_config(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        selected, _filter = QFileDialog.getOpenFileName(
            self.window,
            "server.toml",
            str(self._config_path or default_local_config_path().parent),
            "TOML (*.toml)",
        )
        if selected:
            self.open_config(Path(selected))

    def _fill_accounts(self) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QTableWidgetItem

        self._filling = True
        try:
            rows = [] if self._accounts is None else account_rows(self._accounts)
            self.accounts.setRowCount(len(rows))
            for row, value in enumerate(rows):
                name = QTableWidgetItem(str(value["display_name"]))
                name.setData(Qt.ItemDataRole.UserRole, str(value["username"]))
                self.accounts.setItem(row, 0, name)
                self.accounts.setItem(row, 1, QTableWidgetItem(str(value["username"])))
                self.accounts.setItem(row, 2, QTableWidgetItem("активна" if value["enabled"] else "отключена"))
                admin = "да" if "server_admin" in value["system_roles"] else ""
                self.accounts.setItem(row, 3, QTableWidgetItem(admin))
                self.accounts.setItem(row, 4, QTableWidgetItem(str(value["created_at"])))
        finally:
            self._filling = False

    def _fill_projects(self) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QListWidgetItem

        current = self._selected_project_id()
        self.projects.clear()
        if self._services is None:
            self._fill_maintainers()
            return
        for project in project_rows(self._services):
            label = project["name"]
            if project["state"] and project["state"] != "active":
                label = f"{label} ({project['state']})"
            item = QListWidgetItem(str(label))
            item.setData(Qt.ItemDataRole.UserRole, project["project_id"])
            self.projects.addItem(item)
            if project["project_id"] == current:
                self.projects.setCurrentItem(item)
        if self.projects.currentRow() < 0 and self.projects.count():
            self.projects.setCurrentRow(0)
        self._fill_maintainers()

    def _fill_maintainers(self) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QTableWidgetItem

        project_id = self._selected_project_id()
        rows = []
        if self._services is not None and project_id:
            rows = maintainer_rows(self._services, project_id)
        self.maintainers.setRowCount(len(rows))
        for row, value in enumerate(rows):
            login = QTableWidgetItem(value["username"])
            login.setData(Qt.ItemDataRole.UserRole, value["username"])
            self.maintainers.setItem(row, 0, login)
            self.maintainers.setItem(row, 1, QTableWidgetItem(value["display_name"]))

    def _fill_audit(self) -> None:
        from PyQt6.QtWidgets import QTableWidgetItem

        rows = [] if self._accounts is None else audit_rows(self._accounts)
        self.audit.setRowCount(len(rows))
        for row, event in enumerate(rows):
            action = str(event.get("action", ""))
            target = str(event.get("target_account_id") or "")
            if self._accounts is not None and target:
                account = self._accounts.get_account(target)
                if account is not None:
                    target = account.username
            values = (
                str(event.get("recorded_at", "")),
                _AUDIT_LABELS.get(action, action),
                target,
            )
            for column, value in enumerate(values):
                self.audit.setItem(row, column, QTableWidgetItem(value))

    def _selected_username(self) -> str:
        from PyQt6.QtCore import Qt

        row = self.accounts.currentRow()
        if row < 0:
            return ""
        item = self.accounts.item(row, 0)
        return "" if item is None else str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _selected_project_id(self) -> str:
        from PyQt6.QtCore import Qt

        item = self.projects.currentItem()
        return "" if item is None else str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _selected_maintainer(self) -> str:
        from PyQt6.QtCore import Qt

        row = self.maintainers.currentRow()
        if row < 0:
            return ""
        item = self.maintainers.item(row, 0)
        return "" if item is None else str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _require_account(self) -> str:
        username = self._selected_username()
        if not username:
            self._report("Kraken Admin", "Выберите учётную запись.")
        return username

    def _create_account(self) -> None:
        if self._accounts is None:
            self._report("Kraken Admin", "Сначала откройте server.toml.")
            return
        entered = _account_dialog(self.window)
        if entered is None:
            return
        username, display_name, password = entered
        try:
            create_account(self._accounts, username, display_name, password)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._report("Не удалось создать учётную запись", exc)
            return
        self.reload()

    def _set_enabled(self, enabled: bool) -> None:
        username = self._require_account()
        if not username or self._accounts is None:
            return
        try:
            set_account_enabled(self._accounts, username, enabled=enabled)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._report("Не удалось изменить состояние", exc)
            return
        self.reload()

    def _set_admin(self, grant: bool) -> None:
        username = self._require_account()
        if not username or self._accounts is None:
            return
        try:
            set_server_admin(self._accounts, username, grant=grant)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._report("Не удалось изменить роль администратора", exc)
            return
        self.reload()

    def _reset_password(self) -> None:
        username = self._require_account()
        if not username or self._accounts is None:
            return
        password = _password_dialog(self.window, "Новый пароль")
        if password is None:
            return
        try:
            reset_password(self._accounts, username, password)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._report("Не удалось сменить пароль", exc)
            return
        self.reload()

    def _revoke_sessions(self) -> None:
        username = self._require_account()
        if not username or self._accounts is None:
            return
        try:
            revoke_sessions(self._accounts, username)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._report("Не удалось завершить сеансы", exc)
            return
        self.reload()

    def _revoke_maintainer(self) -> None:
        project_id = self._selected_project_id()
        username = self._selected_maintainer()
        if self._services is None or self._accounts is None:
            self._report("Kraken Admin", "Сначала откройте server.toml.")
            return
        if not project_id or not username:
            self._report("Kraken Admin", "Выберите проект и сопровождающего.")
            return
        try:
            revoke_maintainer(self._services, self._accounts, project=project_id, username=username)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._report("Не удалось снять maintainer", exc)
            return
        self.reload()

    def _report(self, title: str, exc: object) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.warning(self.window, title, str(exc))


def _account_dialog(parent):
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QMessageBox

    dialog = QDialog(parent)
    dialog.setWindowTitle("Новая учётная запись")
    form = QFormLayout(dialog)
    username = QLineEdit(dialog)
    display_name = QLineEdit(dialog)
    password = QLineEdit(dialog)
    password.setEchoMode(QLineEdit.EchoMode.Password)
    confirmation = QLineEdit(dialog)
    confirmation.setEchoMode(QLineEdit.EchoMode.Password)
    form.addRow("Логин", username)
    form.addRow("Отображаемое имя", display_name)
    form.addRow("Пароль", password)
    form.addRow("Повтор пароля", confirmation)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    if not username.text().strip() or not display_name.text().strip():
        QMessageBox.warning(parent, "Kraken Admin", "Укажите логин и отображаемое имя.")
        return None
    if password.text() != confirmation.text() or not password.text():
        QMessageBox.warning(parent, "Kraken Admin", "Пароли не совпадают.")
        return None
    return username.text().strip(), display_name.text().strip(), password.text()


def _password_dialog(parent, title: str) -> str | None:
    from PyQt6.QtWidgets import QInputDialog, QLineEdit, QMessageBox

    password, accepted = QInputDialog.getText(parent, title, "Пароль:", QLineEdit.EchoMode.Password)
    if not accepted or not password:
        return None
    confirmation, accepted = QInputDialog.getText(parent, title, "Повтор пароля:", QLineEdit.EchoMode.Password)
    if not accepted:
        return None
    if password != confirmation:
        QMessageBox.warning(parent, "Kraken Admin", "Пароли не совпадают.")
        return None
    return password
