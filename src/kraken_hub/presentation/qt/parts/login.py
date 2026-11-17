"""manager_app: login."""
from __future__ import annotations
import os
import urllib.parse
from kraken_manager.application.results import UserSession as DesktopSession
from kraken_hub.application.remote_error import RemoteServerError
from kraken_hub.application.workspace import ProjectWorkspaceService
from kraken_hub.infrastructure.credentials import windows as windows_credentials
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_hub.presentation.qt import windows_hello
from kraken_hub.presentation.qt.parts.development_session import (
    _autofill_credentials,
    _remember_credentials,
)
from kraken_hub.presentation.qt.parts.domain_login import (
    STATUS_DISABLED,
    STATUS_ENABLED,
    STATUS_UNREACHABLE,
    domain_login_error_text,
    domain_server_status,
    upsert_program_account,
)
from kraken_hub.presentation.qt.preferences_dialog import (
    program_server_url,
    save_program_server_url,
)

def _build_domain_tab():
    """«Доменная учётная запись» page: server, LDAP status, credentials, submit."""
    from PyQt6.QtWidgets import QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

    page = QWidget()
    layout = QVBoxLayout(page)
    form = QFormLayout()
    server = QLineEdit()
    server.setObjectName("loginDomainServer")
    server.setPlaceholderText("http://127.0.0.1:8080")
    server.setText(program_server_url())
    form.addRow("Сервер", server)
    status = QLabel("")
    status.setObjectName("loginDomainStatus")
    form.addRow("Статус", status)
    username = QLineEdit()
    username.setObjectName("loginDomainUsername")
    username.setText(os.environ.get("USERNAME", ""))
    form.addRow("Логин", username)
    password = QLineEdit()
    password.setObjectName("loginDomainPassword")
    password.setEchoMode(QLineEdit.EchoMode.Password)
    form.addRow("Пароль", password)
    layout.addLayout(form)
    submit = QPushButton("Войти", page)
    submit.setObjectName("loginDomainSubmit")
    submit.setAutoDefault(False)
    layout.addWidget(submit)
    return page


def _server_url_problem(url: str) -> str | None:
    """Пользовательское предупреждение о неприемлемом адресе сервера либо None."""
    if not url:
        return "Укажите адрес сервера."
    if not url.startswith(("http://", "https://")):
        return "Адрес сервера должен начинаться с http:// или https://."
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        return "Сетевой вход в Kraken Server требует HTTPS."
    return None


def _build_server_tab():
    """«Серверная учётная запись» page: server, /health status, credentials, sign in + register."""
    from PyQt6.QtWidgets import QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

    page = QWidget()
    layout = QVBoxLayout(page)
    form = QFormLayout()
    server = QLineEdit()
    server.setObjectName("loginServerAddress")
    server.setPlaceholderText("http://127.0.0.1:8080")
    server.setText(program_server_url())
    form.addRow("Сервер", server)
    status = QLabel("")
    status.setObjectName("loginServerStatus")
    form.addRow("Статус", status)
    username = QLineEdit()
    username.setObjectName("loginServerUsername")
    username.setText(os.environ.get("USERNAME", ""))
    form.addRow("Логин", username)
    password = QLineEdit()
    password.setObjectName("loginServerPassword")
    password.setEchoMode(QLineEdit.EchoMode.Password)
    form.addRow("Пароль", password)
    layout.addLayout(form)
    sign_in = QPushButton("Войти", page)
    sign_in.setObjectName("loginServerSignIn")
    sign_in.setAutoDefault(False)
    register = QPushButton("Создать и войти", page)
    register.setObjectName("loginServerRegister")
    register.setAutoDefault(False)
    layout.addWidget(sign_in)
    layout.addWidget(register)
    return page


def _build_local_tab(service: ProjectWorkspaceService):
    """«Локальная» page: the pre-existing workstation-session widgets, unchanged."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

    page = QWidget()
    layout = QVBoxLayout(page)
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
    layout.addWidget(buttons)
    hint = QLabel("Вход на сервер выполнен. Завершите локальный вход.")
    hint.setObjectName("loginLocalHint")
    hint.setVisible(False)
    layout.addWidget(hint)
    autofill = QPushButton("Заполнить сохранённые данные…", page)
    autofill.setObjectName("loginAutofill")
    autofill.setAutoDefault(False)
    layout.addWidget(autofill)
    return page


def _try_local_autologin(dialog, service: ProjectWorkspaceService) -> DesktopSession | None:
    """Complete the mandatory local session after a successful remote sign-in.

    Mirrors ``development_session._autofill_credentials``: Windows identity
    verification first, then the remembered local credentials. Any failure
    (nothing remembered, verification cancelled, exception) leaves the user
    on the local tab to sign in manually.
    """
    try:
        target = windows_credentials.credential_target(service.data_dir)
        if not windows_credentials.credentials_available(target=target):
            return None
        if not windows_hello.verify_windows_identity(int(dialog.winId())):
            return None
        remembered = windows_credentials.load_credentials(target=target)
        if remembered is None:
            return None
    except Exception:
        return None
    return service.login(remembered[0], remembered[1])


def _login(parent, service: ProjectWorkspaceService) -> DesktopSession | None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QTabWidget,
        QVBoxLayout,
    )

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
    tabs = QTabWidget(dialog)
    tabs.setObjectName("loginTabs")
    tabs.addTab(_build_domain_tab(), "Доменная учётная запись")
    tabs.addTab(_build_server_tab(), "Серверная учётная запись")
    tabs.addTab(_build_local_tab(service), "Локальная")
    layout.addWidget(tabs)
    server = dialog.findChild(QLineEdit, "loginDomainServer")
    status_label = dialog.findChild(QLabel, "loginDomainStatus")
    server_address = dialog.findChild(QLineEdit, "loginServerAddress")
    server_status_label = dialog.findChild(QLabel, "loginServerStatus")
    server_username = dialog.findChild(QLineEdit, "loginServerUsername")
    server_password = dialog.findChild(QLineEdit, "loginServerPassword")
    server_sign_in = dialog.findChild(QPushButton, "loginServerSignIn")
    server_register = dialog.findChild(QPushButton, "loginServerRegister")
    domain_username = dialog.findChild(QLineEdit, "loginDomainUsername")
    domain_password = dialog.findChild(QLineEdit, "loginDomainPassword")
    submit = dialog.findChild(QPushButton, "loginDomainSubmit")
    username = dialog.findChild(QLineEdit, "loginUsername")
    password = dialog.findChild(QLineEdit, "loginPassword")
    hint = dialog.findChild(QLabel, "loginLocalHint")
    buttons = dialog.findChild(QDialogButtonBox)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    # Native Windows verification is modal. Request it only from the visible
    # login window so it cannot block startup before the user sees the app.
    autofill = dialog.findChild(QPushButton, "loginAutofill")
    autofill.clicked.connect(
        lambda: _autofill_credentials(dialog, username, password, data_directory=service.data_dir)
    )

    def refresh_status(url: str | None = None) -> str:
        if url is None:
            url = server.text()
        url = url.strip().rstrip("/")
        if not url:
            status_label.setText("Укажите адрес сервера")
            server_status_label.setText("Укажите адрес сервера")
            return STATUS_UNREACHABLE
        state = domain_server_status(url)
        text = {
            STATUS_ENABLED: "Доменная авторизация включена",
            STATUS_DISABLED: "Доменная авторизация не включена",
            STATUS_UNREACHABLE: "Сервер недоступен",
        }[state]
        status_label.setText(text)
        server_status_label.setText(text)
        return state

    refresh_status()  # blocking probe, <= 3 s
    server.editingFinished.connect(lambda: refresh_status(server.text()))
    server_address.editingFinished.connect(lambda: refresh_status(server_address.text()))

    domain_session: DesktopSession | None = None

    def complete_remote_sign_in(
        *, url: str, entered_username: str, entered_password: str, password_field: QLineEdit
    ) -> DesktopSession | None:
        """Завершить вход после успешной удалённой авторизации: сохранённая локальная
        сессия, иначе вход в локальную учётку-зеркало (создаётся при отсутствии),
        иначе вкладка «Локальная» с подсказкой."""
        upsert_program_account(entered_username, entered_password)
        save_program_server_url(url)
        session = _try_local_autologin(dialog, service)
        if session is None:
            # Удалённо проверенного входа достаточно: локальная учётка-зеркало
            # создаётся, если её нет; чужой локальный пароль не перезаписывается.
            session = service.login_or_create_local(entered_username, entered_password)
            if session is not None:
                _remember_credentials(entered_username, entered_password, data_directory=service.data_dir)
        if session is not None:
            password_field.clear()
            dialog.accept()
            return session
        hint.setVisible(True)
        tabs.setCurrentIndex(2)
        username.setFocus()
        return None

    def try_domain_login() -> None:
        nonlocal domain_session
        url = server.text().strip().rstrip("/")
        problem = _server_url_problem(url)
        if problem is not None:
            QMessageBox.warning(dialog, "Kraken", problem)
            return
        state = refresh_status()
        if state == STATUS_UNREACHABLE:
            QMessageBox.warning(dialog, "Kraken", "Сервер недоступен.")
            return
        if state == STATUS_DISABLED:
            QMessageBox.warning(dialog, "Kraken", "Доменная авторизация не включена на сервере.")
            return
        entered_username = domain_username.text().strip()
        entered_password = domain_password.text()
        if not entered_username or not entered_password:
            QMessageBox.warning(dialog, "Kraken", "Введите логин и пароль доменной учётной записи.")
            return
        try:
            service.configure_remote_local(base_url=url, username=entered_username, password=entered_password)
        except ValueError:
            # URL is already validated above: this is the server's 401.
            QMessageBox.warning(dialog, "Kraken", "Неверный доменный пароль.")
            domain_password.clear()
            return
        except RemoteServerError as exc:
            QMessageBox.warning(dialog, "Kraken", domain_login_error_text(exc))
            return
        domain_session = complete_remote_sign_in(
            url=url,
            entered_username=entered_username,
            entered_password=entered_password,
            password_field=domain_password,
        )

    def prepare_server_entry() -> tuple[str, str, str] | None:
        """Общие проверки серверной вкладки: адрес, доступность сервера, заполненные поля.

        При проблеме показывает предупреждение пользователю и возвращает None.
        """
        url = server_address.text().strip().rstrip("/")
        problem = _server_url_problem(url)
        if problem is not None:
            QMessageBox.warning(dialog, "Kraken", problem)
            return None
        if refresh_status(url) == STATUS_UNREACHABLE:
            QMessageBox.warning(dialog, "Kraken", "Сервер недоступен.")
            return None
        entered_username = server_username.text().strip()
        entered_password = server_password.text()
        if not entered_username or not entered_password:
            QMessageBox.warning(dialog, "Kraken", "Введите логин и пароль серверной учётной записи.")
            return None
        return url, entered_username, entered_password

    def sign_in_server_account(url: str, entered_username: str, entered_password: str) -> None:
        """Вход через configure_remote_local и общий финал; при ошибке предупреждение показано."""
        nonlocal domain_session
        try:
            service.configure_remote_local(base_url=url, username=entered_username, password=entered_password)
        except ValueError:
            # URL уже проверен выше: это 401 сервера.
            server_password.clear()
            QMessageBox.warning(dialog, "Kraken", "Неверный пароль или учётная запись не найдена на сервере")
            return
        except RemoteServerError as exc:
            QMessageBox.warning(dialog, "Kraken", domain_login_error_text(exc))
            return
        domain_session = complete_remote_sign_in(
            url=url,
            entered_username=entered_username,
            entered_password=entered_password,
            password_field=server_password,
        )

    def try_server_sign_in() -> None:
        prepared = prepare_server_entry()
        if prepared is None:
            return
        sign_in_server_account(*prepared)

    def try_server_register() -> None:
        prepared = prepare_server_entry()
        if prepared is None:
            return
        url, entered_username, entered_password = prepared
        try:
            RemoteServerProjectService.register_local(
                url, username=entered_username, password=entered_password, display_name=entered_username
            )
        except RemoteServerError as exc:
            # Ответ сервера показываем как есть: 403 LAN gate, 409 занятый логин, 401.
            QMessageBox.warning(dialog, "Kraken", str(exc))
            return
        sign_in_server_account(url, entered_username, entered_password)

    submit.clicked.connect(try_domain_login)
    server_sign_in.clicked.connect(try_server_sign_in)
    server_register.clicked.connect(try_server_register)
    while dialog.exec() == QDialog.DialogCode.Accepted:
        if domain_session is not None:
            return domain_session
        entered_username = username.text()
        entered_password = password.text()
        session = service.login(entered_username, entered_password)
        password.clear()
        if session is not None:
            _remember_credentials(entered_username.strip(), entered_password, data_directory=service.data_dir)
            return session
        QMessageBox.warning(dialog, "Kraken", "Неверные учётные данные или аккаунт временно заблокирован.")
    return None
