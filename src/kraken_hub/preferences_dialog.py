"""Program-wide preferences: server address and the default account."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kraken_hub.secret_store import KeyringSecretStore

SETTINGS_ORGANIZATION = "Kraken"
SETTINGS_APPLICATION = "KrakenHub"
SERVER_URL_KEY = "server/url"
ACCOUNTS_KEY = "accounts/catalog"
_SECRET_SERVICE = "Kraken.Hub.ProgramAccounts"

ACCOUNT_PROVIDERS = (
    ("local", "Учётная запись Kraken"),
    ("gitlab", "GitLab-токен"),
)
PROVIDER_LABELS = dict(ACCOUNT_PROVIDERS)


@dataclass(frozen=True, slots=True)
class ProgramAccount:
    account_id: str
    provider: str
    username: str
    display_name: str
    secret: str
    is_default: bool

    def label(self) -> str:
        kind = PROVIDER_LABELS.get(self.provider, self.provider)
        mark = " · по умолчанию" if self.is_default else ""
        name = self.display_name or self.username
        return f"{kind}: {name}{mark}"


def program_settings() -> QSettings:
    return QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


def program_secret_store() -> KeyringSecretStore:
    return KeyringSecretStore(_SECRET_SERVICE)


def program_server_url(settings: QSettings | None = None) -> str:
    source = settings or program_settings()
    return str(source.value(SERVER_URL_KEY, "") or "").strip().rstrip("/")


def save_program_server_url(url: str, settings: QSettings | None = None) -> None:
    source = settings or program_settings()
    source.setValue(SERVER_URL_KEY, url.strip().rstrip("/"))


def load_program_accounts(
    settings: QSettings | None = None,
    secrets: KeyringSecretStore | None = None,
) -> list[ProgramAccount]:
    source = settings or program_settings()
    store = secrets or program_secret_store()
    payload = _read_account_catalog(source)
    accounts: list[ProgramAccount] = []
    migrated = False
    for item in payload:
        provider = str(item.get("provider", ""))
        username = str(item.get("username", "")).strip()
        account_id = str(item.get("account_id") or "")
        if provider not in PROVIDER_LABELS or not username or not account_id:
            continue
        legacy_secret = str(item.get("secret") or "")
        stored = _read_secret(store, account_id)
        if legacy_secret:
            if legacy_secret != stored:
                store.set(account_id, legacy_secret.encode("utf-8"))
            stored = legacy_secret
            migrated = True
        if not stored:
            continue
        accounts.append(
            ProgramAccount(
                account_id=account_id,
                provider=provider,
                username=username,
                display_name=str(item.get("display_name") or username).strip(),
                secret=stored,
                is_default=bool(item.get("is_default")),
            )
        )
    accounts = _single_default(accounts)
    if migrated:
        save_program_accounts(accounts, source, store)
    return accounts


def save_program_accounts(
    accounts: list[ProgramAccount],
    settings: QSettings | None = None,
    secrets: KeyringSecretStore | None = None,
) -> None:
    source = settings or program_settings()
    store = secrets or program_secret_store()
    previous_ids = {
        str(item.get("account_id") or "")
        for item in _read_account_catalog(source)
        if str(item.get("account_id") or "")
    }
    normalized = _single_default(accounts)
    for account in normalized:
        if not account.secret:
            raise ValueError("У учётной записи нет пароля или токена")
        store.set(account.account_id, account.secret.encode("utf-8"))
    kept = {account.account_id for account in normalized}
    for account_id in previous_ids - kept:
        store.delete(account_id)
    payload = [
        {
            "account_id": account.account_id,
            "provider": account.provider,
            "username": account.username,
            "display_name": account.display_name,
            "is_default": account.is_default,
        }
        for account in normalized
    ]
    source.setValue(ACCOUNTS_KEY, json.dumps(payload, ensure_ascii=False))


def _read_account_catalog(settings: QSettings) -> list[dict]:
    raw = settings.value(ACCOUNTS_KEY, "")
    text = raw if isinstance(raw, str) else ""
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _read_secret(store: KeyringSecretStore, account_id: str) -> str:
    raw = store.get(account_id)
    if raw is None:
        return ""
    return raw.decode("utf-8")


def default_program_account(settings: QSettings | None = None) -> ProgramAccount | None:
    accounts = load_program_accounts(settings)
    for account in accounts:
        if account.is_default:
            return account
    return accounts[0] if accounts else None


def _single_default(accounts: list[ProgramAccount]) -> list[ProgramAccount]:
    if not accounts:
        return []
    chosen = next((account.account_id for account in accounts if account.is_default), accounts[0].account_id)
    return [
        ProgramAccount(
            account.account_id,
            account.provider,
            account.username,
            account.display_name,
            account.secret,
            account.account_id == chosen,
        )
        for account in accounts
    ]


class ProgramPreferencesDialog(QDialog):
    """Preferences window: a page list on the left and the selected page on the right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("programPreferences")
        self.setWindowTitle("Настройки")
        self.resize(860, 560)
        self._settings = program_settings()
        self._accounts = load_program_accounts(self._settings)

        root = QVBoxLayout(self)
        body = QHBoxLayout()
        self._pages = QTreeWidget(self)
        self._pages.setObjectName("preferencePages")
        self._pages.setHeaderHidden(True)
        self._pages.setFixedWidth(220)
        storage_item = QTreeWidgetItem(["Хранение"])
        account_item = QTreeWidgetItem(["Учетная запись"])
        self._pages.addTopLevelItem(storage_item)
        self._pages.addTopLevelItem(account_item)
        body.addWidget(self._pages)

        content = QVBoxLayout()
        self._title = QLabel("Хранение")
        self._title.setObjectName("preferenceTitle")
        content.addWidget(self._title)
        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._build_storage_page())
        self._stack.addWidget(self._build_account_page())
        content.addWidget(self._stack, 1)
        body.addLayout(content, 1)
        root.addLayout(body, 1)

        buttons = QDialogButtonBox(self)
        buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        apply_button = buttons.addButton(QDialogButtonBox.StandardButton.Apply)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        apply_button.setText("Применить")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        apply_button.clicked.connect(self._apply)
        root.addWidget(buttons)

        self._pages.currentItemChanged.connect(self._show_page)
        self._pages.setCurrentItem(storage_item)
        self._refresh_accounts()

    def _build_storage_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        group = QGroupBox("Сервер", page)
        form = QFormLayout(group)
        self._server_url = QLineEdit(program_server_url(self._settings), group)
        self._server_url.setObjectName("preferenceServerUrl")
        self._server_url.setPlaceholderText("http://127.0.0.1:8080")
        form.addRow("Адрес сервера", self._server_url)
        layout.addWidget(group)
        hint = QLabel(
            "Адрес относится ко всей программе и используется при создании сетевого проекта.",
            page,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_account_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        group = QGroupBox("Учётные записи", page)
        group_layout = QVBoxLayout(group)
        self._account_list = QListWidget(group)
        self._account_list.setObjectName("preferenceAccounts")
        group_layout.addWidget(self._account_list)
        actions = QHBoxLayout()
        add_button = QPushButton("Добавить…", group)
        remove_button = QPushButton("Удалить", group)
        default_button = QPushButton("Сделать по умолчанию", group)
        add_button.clicked.connect(self._add_account)
        remove_button.clicked.connect(self._remove_account)
        default_button.clicked.connect(self._make_default)
        actions.addWidget(add_button)
        actions.addWidget(remove_button)
        actions.addWidget(default_button)
        actions.addStretch(1)
        group_layout.addLayout(actions)
        layout.addWidget(group, 1)
        supported = ", ".join(label for _key, label in ACCOUNT_PROVIDERS)
        hint = QLabel(
            f"Поддерживаемые типы: {supported}. "
            "Учётная запись по умолчанию подставляется при создании сетевого проекта. "
            "Если такой записи ещё нет на сервере, она создаётся автоматически и видит только свои проекты.",
            page,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return page

    def _show_page(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        if current is None:
            return
        index = self._pages.indexOfTopLevelItem(current)
        self._stack.setCurrentIndex(index)
        self._title.setText(current.text(0))

    def _refresh_accounts(self) -> None:
        self._account_list.clear()
        for account in self._accounts:
            item = QListWidgetItem(account.label())
            item.setData(Qt.ItemDataRole.UserRole, account.account_id)
            self._account_list.addItem(item)

    def _selected_account_id(self) -> str:
        item = self._account_list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _add_account(self) -> None:
        account = _AccountEditorDialog.edit(self)
        if account is None:
            return
        self._accounts.append(account)
        self._accounts = _single_default(self._accounts)
        self._refresh_accounts()

    def _remove_account(self) -> None:
        account_id = self._selected_account_id()
        if not account_id:
            return
        self._accounts = [account for account in self._accounts if account.account_id != account_id]
        self._accounts = _single_default(self._accounts)
        self._refresh_accounts()

    def _make_default(self) -> None:
        account_id = self._selected_account_id()
        if not account_id:
            return
        self._accounts = [
            ProgramAccount(
                account.account_id,
                account.provider,
                account.username,
                account.display_name,
                account.secret,
                account.account_id == account_id,
            )
            for account in self._accounts
        ]
        self._refresh_accounts()

    def _validated_url(self) -> str:
        url = self._server_url.text().strip().rstrip("/")
        if url and not url.startswith(("http://", "https://")):
            raise ValueError("Адрес сервера должен начинаться с http:// или https://")
        return url

    def _apply(self) -> bool:
        try:
            url = self._validated_url()
        except ValueError as exc:
            QMessageBox.warning(self, "Настройки", str(exc))
            return False
        save_program_server_url(url, self._settings)
        save_program_accounts(self._accounts, self._settings)
        self._accounts = load_program_accounts(self._settings)
        self._refresh_accounts()
        return True

    def _accept(self) -> None:
        if self._apply():
            self.accept()


class _AccountEditorDialog(QDialog):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Учётная запись")
        form = QFormLayout(self)
        self._provider = QComboBox(self)
        for key, label in ACCOUNT_PROVIDERS:
            self._provider.addItem(label, key)
        self._username = QLineEdit(self)
        self._display_name = QLineEdit(self)
        self._secret = QLineEdit(self)
        self._secret.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Тип", self._provider)
        form.addRow("Логин", self._username)
        form.addRow("Отображаемое имя", self._display_name)
        form.addRow("Пароль или токен", self._secret)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_account)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._provider.currentIndexChanged.connect(self._update_secret_hint)
        self._update_secret_hint()
        self.account: ProgramAccount | None = None

    def _update_secret_hint(self) -> None:
        local = self._provider.currentData() == "local"
        self._secret.setPlaceholderText("Пароль" if local else "GitLab access token")

    def _accept_account(self) -> None:
        username = self._username.text().strip()
        secret = self._secret.text()
        if not username or not secret:
            QMessageBox.warning(self, "Учётная запись", "Укажите логин и пароль или токен")
            return
        display_name = self._display_name.text().strip() or username
        self.account = ProgramAccount(
            account_id=str(uuid4()),
            provider=str(self._provider.currentData()),
            username=username,
            display_name=display_name,
            secret=secret,
            is_default=False,
        )
        self.accept()

    @classmethod
    def edit(cls, parent: QWidget | None) -> ProgramAccount | None:
        dialog = cls(parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.account
