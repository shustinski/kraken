"""manager_app: login."""
from __future__ import annotations
from kraken_manager.application.results import UserSession as DesktopSession
from kraken_hub.application.workspace import ProjectWorkspaceService

from kraken_hub.presentation.qt.parts.development_session import (
    _autofill_credentials,
    _remember_credentials,
)

def _login(parent, service: ProjectWorkspaceService) -> DesktopSession | None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

    if not service.has_accounts:
        dialog = QDialog(parent)
        dialog.setObjectName("initialAccountDialog")
        dialog.setWindowTitle("Создание аккаунта Kraken")
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "В текущем локальном хранилище аккаунтов пока нет. "
            "Создайте первый аккаунт для этой рабочей станции. Если аккаунт уже создавался, "
            "проверьте указанный ниже путь: войти можно только из того же хранилища."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        data_directory = QLabel(f"Хранилище: {service.data_dir}")
        data_directory.setObjectName("initialAccountDataDirectory")
        data_directory.setTextInteractionFlags(
            data_directory.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        data_directory.setWordWrap(True)
        layout.addWidget(data_directory)
        form = QFormLayout()
        username = QLineEdit()
        username.setObjectName("initialAccountUsername")
        username.setPlaceholderText("admin")
        display_name = QLineEdit()
        display_name.setObjectName("initialAccountDisplayName")
        password = QLineEdit()
        password.setObjectName("initialAccountPassword")
        password.setEchoMode(QLineEdit.EchoMode.Password)
        confirmation = QLineEdit()
        confirmation.setObjectName("initialAccountPasswordConfirmation")
        confirmation.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Имя пользователя", username)
        form.addRow("Отображаемое имя", display_name)
        form.addRow("Пароль", password)
        form.addRow("Повторите пароль", confirmation)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Создать аккаунт")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        username.setFocus()
        while dialog.exec() == QDialog.DialogCode.Accepted:
            if not username.text().strip() or any(char.isspace() for char in username.text().strip()):
                QMessageBox.warning(
                    dialog,
                    "Не удалось создать аккаунт",
                    "Введите имя пользователя без пробелов.",
                )
                username.setFocus()
                continue
            if not display_name.text().strip():
                QMessageBox.warning(dialog, "Не удалось создать аккаунт", "Введите отображаемое имя.")
                display_name.setFocus()
                continue
            if password.text() != confirmation.text():
                QMessageBox.warning(dialog, "Не удалось создать аккаунт", "Пароли не совпадают.")
                password.clear()
                confirmation.clear()
                password.setFocus()
                continue
            try:
                session = service.create_initial_account(
                    username=username.text(),
                    display_name=display_name.text(),
                    password=password.text(),
                )
            except Exception as exc:
                QMessageBox.warning(dialog, "Не удалось создать аккаунт", str(exc))
                continue
            _remember_credentials(username.text().strip(), password.text(), data_directory=service.data_dir)
            return session
        return None
    dialog = QDialog(parent)
    dialog.setObjectName("loginDialog")
    dialog.setWindowTitle("Вход в Kraken")
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Локальная сессия рабочей станции"))
    data_directory = QLabel(f"Хранилище: {service.data_dir}")
    data_directory.setObjectName("loginDataDirectory")
    data_directory.setTextInteractionFlags(
        data_directory.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
    )
    data_directory.setWordWrap(True)
    layout.addWidget(data_directory)
    form = QFormLayout()
    username = QLineEdit()
    username.setObjectName("loginUsername")
    password = QLineEdit()
    password.setObjectName("loginPassword")
    password.setEchoMode(QLineEdit.EchoMode.Password)
    form.addRow("Имя пользователя", username)
    form.addRow("Пароль", password)
    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Войти")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    # Native Windows verification is modal. Request it only from the visible
    # login window so it cannot block startup before the user sees the app.
    autofill = QPushButton("Заполнить сохранённые данные…", dialog)
    autofill.setObjectName("loginAutofill")
    autofill.setAutoDefault(False)
    autofill.clicked.connect(
        lambda: _autofill_credentials(dialog, username, password, data_directory=service.data_dir)
    )
    layout.addWidget(autofill)
    while dialog.exec() == QDialog.DialogCode.Accepted:
        entered_username = username.text()
        entered_password = password.text()
        session = service.login(entered_username, entered_password)
        password.clear()
        if session is not None:
            _remember_credentials(entered_username.strip(), entered_password, data_directory=service.data_dir)
            return session
        QMessageBox.warning(dialog, "Kraken", "Неверные учётные данные или аккаунт временно заблокирован.")
    return None
