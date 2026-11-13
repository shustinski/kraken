# Implementation Plan: окно входа Kraken Hub — вкладка «Серверная учётная запись»

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** в окне входа Kraken Hub третья вкладка «Серверная учётная запись»: вход по обычным серверным учётным данным (логин/пароль, `configure_remote_local`) и кнопка «Создать и войти» (регистрация через `POST /api/v1/auth/accounts`, затем вход). В Kraken Admin тип «локальная» переименовывается в «серверная», подсказки в хабе обновляются с указанием новой вкладки.

**Architecture:** серверная часть уже существует и не меняется: `POST /api/v1/auth/accounts` (`src/kraken_server/interface/http/routes/auth.py`, 403 LAN gate / 409 занятый логин / 401) и клиентский `RemoteServerProjectService.register_local` (`src/kraken_hub/infrastructure/remote/parts/register_local.py`, строка 80). Хаб добавляет `_build_server_tab()` рядом с `_build_domain_tab()`; статус `/health` (один зонд ≤ 3 с) дублируется в строки статуса обеих серверных вкладок; финал успешного серверного входа (upsert программного аккаунта + сохранение URL + автологин или переход на вкладку «Локальная») выносится в общий помощник и делится с доменной вкладкой. `configure_remote_local` (`src/kraken_hub/composition/catalog.py:306`) и её тест `tests/test_no_auto_registration.py` **не трогаются**.

**Tech Stack:** Python 3.11+, PyQt6 (хаб, админ), urllib (HTTP-клиент хаба), pytest. Новых зависимостей нет.

**Spec:** решения согласованы в диалоге 2026-11-13 (вкладка «Серверная учётная запись», тип «серверная», отдельная кнопка регистрации, subagent-driven execution).

## Global Constraints

- UI-тексты — на русском (конвенция репозитория).
- Вкладки: **0 — «Доменная учётная запись» (по умолчанию), 1 — «Серверная учётная запись», 2 — «Локальная»**. Поведение вкладки «Локальная» и `initialAccountDialog` не меняется.
- Вкладка «Серверная» **не блокируется** статусом `STATUS_DISABLED` (в отличие от доменной): обычные серверные аккаунты работают, даже когда LDAP выключен. Блокирует только `STATUS_UNREACHABLE`.
- 401-текст серверной вкладки: «Неверный пароль или учётная запись не найдена на сервере». 401-текст доменной вкладки остаётся «Неверный доменный пароль.»
- Ошибки сервера при регистрации показываются как есть (`str(exc)`): 403 LAN gate, 409 «Логин «…» уже занят», 401.
- `RemoteServerError` — один класс (`kraken_hub.application.remote_error.RemoteServerError`); `register_local` и `configure_remote_local` бросают именно его.
- `RemoteServerProjectService` импортируется в `login.py` **на уровне модуля** (`from kraken_hub.infrastructure.remote.client import RemoteServerProjectService`), чтобы тесты могли monkeypatch-ить `login_parts.RemoteServerProjectService.register_local`.
- Тесты не требуют сети, реального AD или реального сервера (monkeypatch/фейки).
- Коммит после каждой задачи; короткие заголовки, стиль существующих (`feat(hub): …`, `feat(admin): …`).
- Проверка: `C:\code\kraken\.venv\Scripts\python.exe -m pytest <файлы> -v -p no:cacheprovider` (запускать из корня worktree!) и `C:\code\kraken\.venv\Scripts\python.exe -m ruff check <изменённые файлы>`.
- Не трогать пользовательские незакоммиченные файлы main-дерева (`docs/quality/evidence/...`, `docs/superpowers/{plans,specs}/2026-11-09-*`, `src/kraken_admin/blob_gateway/`).

## Review Focus

1. **Регрессия доменного и локального входа** (Critical): доменная вкладка работает как раньше (тот же порядок проверок, те же сообщения, тот же финал); локальная вкладка не потеряна (индекс 2), `_sign_in` harness и приёмочная серия (35 тестов) проходят.
2. **Один зонд `/health`** до `exec` по URL из настроек; текст статуса одинаковый на обеих серверных вкладках; `editingFinished` у каждого адресного поля перезондирует свой URL.
3. **Регистрация**: `register_local(url, username=…, password=…, display_name=username)` → затем `configure_remote_local` (тот же логин/пароль) → общий финал. Ошибка регистрации: серверный текст как есть, `configure_remote_local` не вызывается, диалог остаётся открытым.
4. **Герметичность тестов**: QSettings/keyring/Windows Hello не трогаются (monkeypatch `upsert_program_account`, `save_program_server_url`, `_try_local_autologin`).
5. **Переименование**: только метка типа и две подсказки; «Создать на сервере…» в Настройках остаётся; `PROVIDER_LABELS` (gitlab/local как провайдеры программных аккаунтов) не меняется.

---

### Task 1: Хаб — вкладка «Серверная учётная запись» в окне входа

**Files:**
- Modify: `src/kraken_hub/presentation/qt/parts/login.py`
- Modify: `tests/test_hub_login_window.py` (2 существующих теста: `setCurrentIndex(1)` → `setCurrentIndex(2)`; +5 новых тестов; докстринг модуля)
- Modify: `tests/helpers/qt_harness.py` (`_sign_in`: `tabs.setCurrentIndex(1)` → `tabs.setCurrentIndex(2)`, строка 619)

**Interfaces:**
- Consumes:
  - `domain_server_status`, `upsert_program_account`, `domain_login_error_text`, `STATUS_ENABLED/DISABLED/UNREACHABLE` — `kraken_hub.presentation.qt.parts.domain_login` (без изменений).
  - `RemoteServerProjectService.register_local(base_url, *, username, password, display_name="", data_dir=None, http=None)` — classmethod, `POST /api/v1/auth/accounts`; бросает `RemoteServerError(detail сервера, status=403/409/401/…)`.
  - `service.configure_remote_local(base_url=, username=, password=)` — закреплённый путь, не менять; 401 → `ValueError`.
  - `program_server_url(settings=None)`, `save_program_server_url(url, settings=None)` — `kraken_hub.presentation.qt.preferences_dialog`.
- Produces (контракт виджетов, проверяется тестами):
  - Вкладка 1, текст «Серверная учётная запись»:
    - `loginServerAddress` (QLineEdit, placeholder `http://127.0.0.1:8080`, предзаполнен `program_server_url()`),
    - `loginServerStatus` (QLabel),
    - `loginServerUsername` (QLineEdit, предзаполнен `os.environ.get("USERNAME", "")`, редактируемый),
    - `loginServerPassword` (QLineEdit, `EchoMode.Password`),
    - `loginServerSignIn` (QPushButton «Войти», `setAutoDefault(False)`),
    - `loginServerRegister` (QPushButton «Создать и войти», `setAutoDefault(False)`).
  - Текст статуса (маппинг как у доменной вкладки: enabled → «Доменная авторизация включена», disabled → «Доменная авторизация не включена», unreachable → «Сервер недоступен», пустой URL → «Укажите адрес сервера») пишется **в обе** строки статуса (`loginDomainStatus` и `loginServerStatus`).
  - Переход в «Локальную» после успешного серверного входа: `tabs.setCurrentIndex(2)`; текст `loginLocalHint` обобщён: «Вход на сервер выполнен. Завершите локальный вход.»

- [ ] **Step 1: Write the failing tests**

В `tests/test_hub_login_window.py`. Докстринг модуля: «Tabbed login window (domain LDAP + server account + local) driven offscreen.» (остаток докстринга без изменений).

Пять новых тестов (паттерн watcher/actor — как у соседних тестов; фикстура `service_with_account`):

```python
def test_login_window_server_tab_layout(qapp, tmp_path, monkeypatch, service_with_account):
    """Три вкладки: «Доменная» (по умолчанию), «Серверная», «Локальная»; предзаполнение и кнопки серверной вкладки."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    seen: dict = {}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if not _is_login_dialog(dialog) or seen:
            return
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        if tabs is None:
            return
        sign_in = dialog.findChild(QPushButton, "loginServerSignIn")
        register = dialog.findChild(QPushButton, "loginServerRegister")
        password = dialog.findChild(QLineEdit, "loginServerPassword")
        if sign_in is None or register is None or password is None:
            return
        seen["count"] = tabs.count()
        seen["texts"] = [tabs.tabText(i) for i in range(tabs.count())]
        seen["default"] = tabs.currentIndex()
        seen["address"] = dialog.findChild(QLineEdit, "loginServerAddress").text()
        seen["username"] = dialog.findChild(QLineEdit, "loginServerUsername").text()
        seen["sign_in"] = (type(sign_in).__name__, sign_in.text())
        seen["register"] = (type(register).__name__, register.text())
        seen["domain_status"] = dialog.findChild(QLabel, "loginDomainStatus").text()
        seen["server_status"] = dialog.findChild(QLabel, "loginServerStatus").text()
        seen["password_echo"] = password.echoMode()

        def close() -> None:
            dialog.reject()

        _post_action(actors, close)

    timer.timeout.connect(poll)
    timer.start()
    deadline = _start_deadline_reject()
    try:
        session = login_parts._login(None, service_with_account)
    finally:
        timer.stop()
        deadline.stop()
        for actor in actors:
            actor.stop()
    assert session is None
    assert seen["count"] == 3
    assert seen["texts"] == ["Доменная учётная запись", "Серверная учётная запись", "Локальная"]
    assert seen["default"] == 0
    assert seen["address"] == "http://127.0.0.1:8080"
    assert seen["username"] == os.environ.get("USERNAME", "")
    assert seen["sign_in"] == ("QPushButton", "Войти")
    assert seen["register"] == ("QPushButton", "Создать и войти")
    assert seen["domain_status"] == "Доменная авторизация включена"
    assert seen["server_status"] == "Доменная авторизация включена"
    assert seen["password_echo"] == QLineEdit.EchoMode.Password


def test_server_sign_in_success_completes_session(qapp, tmp_path, monkeypatch, service_with_account):
    """«Войти» серверной вкладки: configure_remote_local с введёнными данными, сохранение учётных данных,
    переход на вкладку «Локальная» с подсказкой; локальный вход завершает сессию."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    monkeypatch.setattr(login_parts, "_try_local_autologin", lambda dialog, service: None)
    calls: dict = {}
    monkeypatch.setattr(
        login_parts, "upsert_program_account",
        lambda username, password, **kw: calls.setdefault("upsert", (username, password)),
    )
    monkeypatch.setattr(login_parts, "save_program_server_url", lambda url, **kw: calls.setdefault("url", url))

    def fake_configure(base_url, username, password):
        calls["configure"] = (base_url, username, password)
        return object()

    monkeypatch.setattr(service_with_account, "configure_remote_local", fake_configure)
    seen: dict = {}
    stage = {"name": "server_sign_in"}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if not _is_login_dialog(dialog):
            return
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        if stage["name"] == "server_sign_in":
            stage["name"] = "expect_handoff"
            sign_in = dialog.findChild(QPushButton, "loginServerSignIn")

            def click_sign_in() -> None:
                if sign_in is None:
                    raise AssertionError("loginServerSignIn not found")
                dialog.findChild(QLineEdit, "loginServerUsername").setText("server-user")
                dialog.findChild(QLineEdit, "loginServerPassword").setText("server-pw")
                QTest.mouseClick(sign_in, Qt.MouseButton.LeftButton)

            _post_action(actors, click_sign_in)
        elif stage["name"] == "expect_handoff":
            seen["handoff_tab"] = tabs.currentIndex()
            seen["hint_visible"] = dialog.findChild(QLabel, "loginLocalHint").isVisible()
            stage["name"] = "local_login"
            username_field = dialog.findChild(QLineEdit, "loginUsername")
            password_field = dialog.findChild(QLineEdit, "loginPassword")
            box = dialog.findChild(QDialogButtonBox)

            def complete_local() -> None:
                username_field.setText("mark")
                password_field.setText("pw")
                QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

            _post_action(actors, complete_local)

    timer.timeout.connect(poll)
    timer.start()
    deadline = _start_deadline_reject()
    try:
        session = login_parts._login(None, service_with_account)
    finally:
        timer.stop()
        deadline.stop()
        for actor in actors:
            actor.stop()
    assert session is not None
    assert calls["configure"] == ("http://127.0.0.1:8080", "server-user", "server-pw")
    assert calls["upsert"] == ("server-user", "server-pw")
    assert calls["url"] == "http://127.0.0.1:8080"
    assert seen["handoff_tab"] == 2
    assert seen["hint_visible"] is True


def test_server_sign_in_rejects_bad_credentials(qapp, tmp_path, monkeypatch, service_with_account):
    """401 от configure_remote_local (ValueError): предупреждение про серверную учётную запись,
    пароль очищается, диалог остаётся открытым, локальный вход доступен."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")

    def bad_configure(**_kwargs):
        raise ValueError("Учётная запись не найдена или неверный пароль")

    monkeypatch.setattr(service_with_account, "configure_remote_local", bad_configure)
    seen: dict = {}
    stage = {"name": "server_sign_in"}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            if "warning_text" not in seen:
                login = seen["login_dialog"]
                seen["warning_text"] = dialog.text()
                seen["password"] = login.findChild(QLineEdit, "loginServerPassword").text()
                ok = dialog.button(QMessageBox.StandardButton.Ok)

                def close_warning() -> None:
                    stage["name"] = "local_login"
                    if ok is not None:
                        QTest.mouseClick(ok, Qt.MouseButton.LeftButton)

                _post_action(actors, close_warning)
            return
        if not _is_login_dialog(dialog):
            return
        seen.setdefault("login_dialog", dialog)
        if stage["name"] == "server_sign_in":
            stage["name"] = "expect_warning"
            sign_in = dialog.findChild(QPushButton, "loginServerSignIn")

            def click_sign_in() -> None:
                if sign_in is None:
                    raise AssertionError("loginServerSignIn not found")
                dialog.findChild(QLineEdit, "loginServerUsername").setText("server-user")
                dialog.findChild(QLineEdit, "loginServerPassword").setText("server-pw")
                QTest.mouseClick(sign_in, Qt.MouseButton.LeftButton)

            _post_action(actors, click_sign_in)
        elif stage["name"] == "local_login":
            username_field = dialog.findChild(QLineEdit, "loginUsername")
            password_field = dialog.findChild(QLineEdit, "loginPassword")
            box = dialog.findChild(QDialogButtonBox)

            def complete_local() -> None:
                username_field.setText("mark")
                password_field.setText("pw")
                QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

            _post_action(actors, complete_local)

    timer.timeout.connect(poll)
    timer.start()
    deadline = _start_deadline_reject()
    try:
        session = login_parts._login(None, service_with_account)
    finally:
        timer.stop()
        deadline.stop()
        for actor in actors:
            actor.stop()
    assert session is not None
    assert "Неверный пароль или учётная запись не найдена на сервере" in seen["warning_text"]
    assert seen["password"] == ""


def test_server_register_creates_account_then_signs_in(qapp, tmp_path, monkeypatch, service_with_account):
    """«Создать и войти»: register_local (display_name == логин), затем configure_remote_local,
    затем переход на вкладку «Локальная»."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    monkeypatch.setattr(login_parts, "_try_local_autologin", lambda dialog, service: None)
    calls: dict = {}
    monkeypatch.setattr(
        login_parts, "upsert_program_account",
        lambda username, password, **kw: calls.setdefault("upsert", (username, password)),
    )
    monkeypatch.setattr(login_parts, "save_program_server_url", lambda url, **kw: calls.setdefault("url", url))

    def fake_register(base_url, *, username, password, display_name="", **_kw):
        calls["register"] = (base_url, username, password, display_name)
        return object()

    monkeypatch.setattr(login_parts.RemoteServerProjectService, "register_local", fake_register)

    def fake_configure(base_url, username, password):
        calls["configure"] = (base_url, username, password)
        return object()

    monkeypatch.setattr(service_with_account, "configure_remote_local", fake_configure)
    seen: dict = {}
    stage = {"name": "server_register"}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if not _is_login_dialog(dialog):
            return
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        if stage["name"] == "server_register":
            stage["name"] = "expect_handoff"
            register = dialog.findChild(QPushButton, "loginServerRegister")

            def click_register() -> None:
                if register is None:
                    raise AssertionError("loginServerRegister not found")
                dialog.findChild(QLineEdit, "loginServerUsername").setText("server-user")
                dialog.findChild(QLineEdit, "loginServerPassword").setText("server-pw")
                QTest.mouseClick(register, Qt.MouseButton.LeftButton)

            _post_action(actors, click_register)
        elif stage["name"] == "expect_handoff":
            seen["handoff_tab"] = tabs.currentIndex()
            stage["name"] = "local_login"
            username_field = dialog.findChild(QLineEdit, "loginUsername")
            password_field = dialog.findChild(QLineEdit, "loginPassword")
            box = dialog.findChild(QDialogButtonBox)

            def complete_local() -> None:
                username_field.setText("mark")
                password_field.setText("pw")
                QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

            _post_action(actors, complete_local)

    timer.timeout.connect(poll)
    timer.start()
    deadline = _start_deadline_reject()
    try:
        session = login_parts._login(None, service_with_account)
    finally:
        timer.stop()
        deadline.stop()
        for actor in actors:
            actor.stop()
    assert session is not None
    assert calls["register"] == ("http://127.0.0.1:8080", "server-user", "server-pw", "server-user")
    assert calls["configure"] == ("http://127.0.0.1:8080", "server-user", "server-pw")
    assert calls["upsert"] == ("server-user", "server-pw")
    assert seen["handoff_tab"] == 2


def test_server_register_shows_server_error_as_is(qapp, tmp_path, monkeypatch, service_with_account):
    """Ошибка сервера при регистрации (403 LAN gate) показывается как есть;
    configure_remote_local не вызывается; локальный вход завершает сессию."""
    from kraken_hub.application.remote_error import RemoteServerError

    message = "Создание локальных аккаунтов из внутренней сети запрещено. Используйте доменную учётную запись."

    def forbidden_register(*_args, **_kwargs):
        raise RemoteServerError(message, status=403)

    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    monkeypatch.setattr(login_parts.RemoteServerProjectService, "register_local", forbidden_register)
    calls: dict = {}
    monkeypatch.setattr(
        service_with_account, "configure_remote_local",
        lambda base_url, username, password: calls.setdefault("configure", (base_url, username, password)),
    )
    seen: dict = {}
    stage = {"name": "server_register"}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            if "warning_text" not in seen:
                seen["warning_text"] = dialog.text()
                ok = dialog.button(QMessageBox.StandardButton.Ok)

                def close_warning() -> None:
                    stage["name"] = "local_login"
                    if ok is not None:
                        QTest.mouseClick(ok, Qt.MouseButton.LeftButton)

                _post_action(actors, close_warning)
            return
        if not _is_login_dialog(dialog):
            return
        if stage["name"] == "server_register":
            stage["name"] = "expect_warning"
            register = dialog.findChild(QPushButton, "loginServerRegister")

            def click_register() -> None:
                if register is None:
                    raise AssertionError("loginServerRegister not found")
                dialog.findChild(QLineEdit, "loginServerUsername").setText("server-user")
                dialog.findChild(QLineEdit, "loginServerPassword").setText("server-pw")
                QTest.mouseClick(register, Qt.MouseButton.LeftButton)

            _post_action(actors, click_register)
        elif stage["name"] == "local_login":
            username_field = dialog.findChild(QLineEdit, "loginUsername")
            password_field = dialog.findChild(QLineEdit, "loginPassword")
            box = dialog.findChild(QDialogButtonBox)

            def complete_local() -> None:
                username_field.setText("mark")
                password_field.setText("pw")
                QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

            _post_action(actors, complete_local)

    timer.timeout.connect(poll)
    timer.start()
    deadline = _start_deadline_reject()
    try:
        session = login_parts._login(None, service_with_account)
    finally:
        timer.stop()
        deadline.stop()
        for actor in actors:
            actor.stop()
    assert session is not None
    assert seen["warning_text"] == message
    assert "configure" not in calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_hub_login_window.py -v -p no:cacheprovider`
Expected: 5 новых тестов FAIL/ERROR (вкладки/виджеты/`login_parts.RemoteServerProjectService` не существуют), 3 существующих PASS.

- [ ] **Step 3: Implement the server tab in `login.py`**

1. Импорты модуля (существующие без изменений + два новых):
```python
from kraken_hub.application.remote_error import RemoteServerError
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
```
Ленивый `from kraken_hub.application.remote_error import RemoteServerError` внутри `try_domain_login` удалить (класс теперь на уровне модуля).

2. Модульный помощник (рядом с `_build_domain_tab`):
```python
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
```

3. `_build_server_tab()` (аналог `_build_domain_tab`, после него):
```python
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
```

4. В `_login`:
   - `tabs.addTab(_build_server_tab(), "Серверная учётная запись")` между доменной и локальной вкладками.
   - Поиск виджетов: `server_address` (`loginServerAddress`), `server_status_label` (`loginServerStatus`), `server_username`, `server_password`, `server_sign_in` (`loginServerSignIn`), `server_register` (`loginServerRegister`).
   - Текст `loginLocalHint`: «Вход на сервер выполнен. Завершите локальный вход.»
   - `refresh_status(url: str | None = None) -> str` — URL по умолчанию из доменного поля; результат (маппинг как сейчас: enabled → «Доменная авторизация включена», disabled → «Доменная авторизация не включена», unreachable/пусто → «Сервер недоступен»/«Укажите адрес сервера») пишется в `status_label` **и** `server_status_label`.
   - Подключения: `server.editingFinished.connect(lambda: refresh_status(server.text()))` (как сейчас) и `server_address.editingFinished.connect(lambda: refresh_status(server_address.text()))`.
   - Общий финал успешного серверного входа (вложенная функция внутри `_login`, после `refresh_status`):
```python
    def complete_remote_sign_in(*, url: str, username: str, password: str, password_field: QLineEdit) -> DesktopSession | None:
        """Сохранить учётные данные, попытаться завершить локальный вход (Windows Hello),
        иначе передать пользователя на вкладку «Локальная» с подсказкой."""
        upsert_program_account(username, password)
        save_program_server_url(url)
        session = _try_local_autologin(dialog, service)
        if session is not None:
            password_field.clear()
            dialog.accept()
            return session
        hint.setVisible(True)
        tabs.setCurrentIndex(2)
        username.setFocus()
        return None
```
   - `try_domain_login` переиспользует `_server_url_problem` (три первых проверки заменяются `problem = _server_url_problem(url); if problem is not None: warn; return` — сообщения те же) и `complete_remote_sign_in` (хвост вместо ручного upsert/save/autologin/handoff): `domain_session = complete_remote_sign_in(url=url, username=entered_username, password=entered_password, password_field=domain_password)`.
   - `try_server_sign_in` (без гейта `STATUS_DISABLED`!):
```python
    def try_server_sign_in() -> None:
        nonlocal domain_session
        url = server_address.text().strip().rstrip("/")
        problem = _server_url_problem(url)
        if problem is not None:
            QMessageBox.warning(dialog, "Kraken", problem)
            return
        if refresh_status(url) == STATUS_UNREACHABLE:
            QMessageBox.warning(dialog, "Kraken", "Сервер недоступен.")
            return
        entered_username = server_username.text().strip()
        entered_password = server_password.text()
        if not entered_username or not entered_password:
            QMessageBox.warning(dialog, "Kraken", "Введите логин и пароль серверной учётной записи.")
            return
        try:
            service.configure_remote_local(base_url=url, username=entered_username, password=entered_password)
        except ValueError:
            # URL уже проверен выше: это 401 сервера.
            QMessageBox.warning(dialog, "Kraken", "Неверный пароль или учётная запись не найдена на сервере")
            server_password.clear()
            return
        except RemoteServerError as exc:
            QMessageBox.warning(dialog, "Kraken", domain_login_error_text(exc))
            return
        domain_session = complete_remote_sign_in(
            url=url, username=entered_username, password=entered_password, password_field=server_password
        )
```
   - `try_server_register`:
```python
    def try_server_register() -> None:
        nonlocal domain_session
        url = server_address.text().strip().rstrip("/")
        problem = _server_url_problem(url)
        if problem is not None:
            QMessageBox.warning(dialog, "Kraken", problem)
            return
        if refresh_status(url) == STATUS_UNREACHABLE:
            QMessageBox.warning(dialog, "Kraken", "Сервер недоступен.")
            return
        entered_username = server_username.text().strip()
        entered_password = server_password.text()
        if not entered_username or not entered_password:
            QMessageBox.warning(dialog, "Kraken", "Введите логин и пароль серверной учётной записи.")
            return
        try:
            RemoteServerProjectService.register_local(
                url, username=entered_username, password=entered_password, display_name=entered_username
            )
        except RemoteServerError as exc:
            # Ответ сервера показываем как есть: 403 LAN gate, 409 занятый логин, 401.
            QMessageBox.warning(dialog, "Kraken", str(exc))
            return
        try:
            service.configure_remote_local(base_url=url, username=entered_username, password=entered_password)
        except ValueError:
            QMessageBox.warning(dialog, "Kraken", "Неверный пароль или учётная запись не найдена на сервере")
            server_password.clear()
            return
        except RemoteServerError as exc:
            QMessageBox.warning(dialog, "Kraken", domain_login_error_text(exc))
            return
        domain_session = complete_remote_sign_in(
            url=url, username=entered_username, password=entered_password, password_field=server_password
        )
```
   - Подключения кнопок: `server_sign_in.clicked.connect(try_server_sign_in)`, `server_register.clicked.connect(try_server_register)`.

- [ ] **Step 4: Update tab indices in existing tests and harness**

- `tests/test_hub_login_window.py`: в `test_login_window_domain_tab_default_and_local_login_completes` и `test_domain_submit_rejects_when_ldap_disabled` — `tabs.setCurrentIndex(1)` → `tabs.setCurrentIndex(2)` (вкладка «Локальная» теперь третья).
- `tests/helpers/qt_harness.py` (`_sign_in`, строка 619): `tabs.setCurrentIndex(1)  # локальная вкладка — путь существующих тестов` → `tabs.setCurrentIndex(2)  # локальная вкладка — путь существующих тестов`.

- [ ] **Step 5: Run tests to verify they pass**

Run (из корня worktree):
- `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_hub_login_window.py tests/test_no_auto_registration.py -v -p no:cacheprovider` — все PASS.
- `C:\code\kraken\.venv\Scripts\python.exe -m ruff check src/kraken_hub/presentation/qt/parts/login.py tests/test_hub_login_window.py tests/helpers/qt_harness.py`.

- [ ] **Step 6: Commit**

```bash
git add src/kraken_hub/presentation/qt/parts/login.py tests/test_hub_login_window.py tests/helpers/qt_harness.py
git commit -m "feat(hub): server account tab in the login window (sign in + create and sign in)"
```

---

### Task 2: Админ — тип «серверная» вместо «локальная» + подсказки хаба

**Files:**
- Modify: `src/kraken_admin/application/accounts.py` (`account_type_label`, строка 135)
- Modify: `tests/test_admin_app.py` (4 строки: 70, 120, 127, 128)
- Modify: `src/kraken_hub/presentation/qt/preferences_dialog.py` (подсказка страницы «Учетная запись», строки 309-313)
- Modify: `src/kraken_hub/presentation/qt/parts/connect_saved_server.py` (`_server_project_hint`, строка 28)

**Interfaces:**
- Consumes: ничего нового.
- Produces:
  - `account_type_label(auth_provider) -> str`: `"доменная"` при `"ldap"`, иначе `"серверная"` (было `"локальная"`). Таблица аккаунтов (`src/kraken_admin/presentation/qt/window.py:168`) заголовков/колонок не меняет.
  - Подсказки хаба упоминают вкладку «Серверная учётная запись» окна входа; кнопка «Создать на сервере…» в Настройках остаётся.
- Не меняется: `PROVIDER_LABELS` (провайдеры программных аккаунтов gitlab/local), текст кнопки «Создать на сервере…», `run_local_admin.py:80` («Неизвестная локальная команда» — про CLI-команды, не про учётные записи).

- [ ] **Step 1: Update the failing assertions in `tests/test_admin_app.py`**

- Строка 70: `assert window.accounts.item(0, 2).text() == "локальная"` → `"серверная"`.
- Строка 120: `assert window.accounts.item(1, 2).text() == "локальная"` → `"серверная"`.
- Строки 127-128: `assert account_type_label("local") == "локальная"` → `"серверная"`; `assert account_type_label(None) == "локальная"` → `"серверная"`.
- Строка 126 (`account_type_label("ldap") == "доменная"`) без изменений.

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_admin_app.py -v -p no:cacheprovider`
Expected: FAIL на `test_admin_window_lists_accounts_projects_and_revokes_maintainer`, `test_admin_window_marks_ldap_accounts`, `test_account_type_label` (ожидалось «серверная», получено «локальная»).

- [ ] **Step 3: Rename the label**

`src/kraken_admin/application/accounts.py` (строка 135):
```python
    return "доменная" if str(auth_provider) == "ldap" else "серверная"
```

- [ ] **Step 4: Update the hub hints**

`src/kraken_hub/presentation/qt/preferences_dialog.py` (строки 309-313) — заменить:
```python
        hint = QLabel(
            "Доменные пользователи вводят доменный логин и пароль — аккаунт создаётся на сервере при первом входе. "
            "Локальные учётные записи (пользователи вне локальной сети) создаются кнопкой «Создать на сервере…».",
            page,
        )
```
на:
```python
        hint = QLabel(
            "Доменные пользователи вводят доменный логин и пароль — аккаунт создаётся на сервере при первом входе. "
            "Серверные учётные записи (пользователи вне локальной сети) создаются в окне входа на вкладке "
            "«Серверная учётная запись» («Создать и войти») или кнопкой «Создать на сервере…».",
            page,
        )
```

`src/kraken_hub/presentation/qt/parts/connect_saved_server.py` (строка 28) — заменить строку
```python
            "Локальная запись создаётся на сервере явно (Настройки → Учетная запись → «Создать на сервере…»). "
```
на:
```python
            "Серверная запись создаётся в окне входа на вкладке «Серверная учётная запись» («Создать и войти») "
            "или в Настройках → Учетная запись («Создать на сервере…»). "
```

- [ ] **Step 5: Run tests to verify they pass + no stale «локальная»-метки**

Run:
- `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_admin_app.py tests/test_hub_login_window.py tests/test_no_auto_registration.py -v -p no:cacheprovider` — все PASS.
- `C:\code\kraken\.venv\Scripts\python.exe -m ruff check src/kraken_admin/application/accounts.py src/kraken_hub/presentation/qt/preferences_dialog.py src/kraken_hub/presentation/qt/parts/connect_saved_server.py tests/test_admin_app.py`.
- Проверка grep: в `src/kraken_admin` и `src/kraken_hub` нет метки «локальная учётная запись»/«локальные учётные записи» в UI-текстах (допустимо: «Локальная сессия рабочей станции», «Локальная» как название вкладки, «Неизвестная локальная команда», `PROVIDER_LABELS`).

- [ ] **Step 6: Commit**

```bash
git add src/kraken_admin/application/accounts.py tests/test_admin_app.py src/kraken_hub/presentation/qt/preferences_dialog.py src/kraken_hub/presentation/qt/parts/connect_saved_server.py
git commit -m "feat(admin): rename 'local' account type to 'server' and mention the login tab in hints"
```

---

## Final Verification (after both tasks, whole branch)

- Полная приёмочная серия ветки (`run-branch-tests.ps1` / `pytest tests/ -v -p no:cacheprovider` из корня worktree с `KRAKEN_TEST_POSTGRES_URL`) — 35 приёмочных тестов + юниты без новых провалов.
- Известный pre-existing flake (НЕ регрессия, не чинить в ветке): `tests/test_qt_harness.py::test_failure_capture_produces_artifacts` падает, если в том же процессе pytest до него отработал `tests/test_ldap_http.py` (кольцевой журнал модульного уровня, введён в `807bf54`; воспроизводится на main до ветки).
- Ручная приёмка (на машине пользователя, сервер 127.0.0.1:8080): вход на вкладке «Серверная учётная запись» с новым логином/паролем через «Создать и войти» — аккаунт создаётся (не админ), сессия устанавливается, хаб переходит в рабочую область; повторный вход через «Войти»; с рабочей станции в LAN (172.20.0.0/16) «Создать и войти» показывает 403-текст «Создание локальных аккаунтов из внутренней сети запрещено. Используйте доменную учётную запись.»; доменная и локальная вкладки работают как раньше; в Kraken Admin тип обычных аккаунтов — «серверная».
