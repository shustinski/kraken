# Implementation Plan: окно входа Kraken Hub — доменная и локальная учётная запись

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** одно окно входа при запуске хаба с вкладками «Доменная учётная запись» (адрес сервера, статус LDAP, логин из Windows, пароль с сохранением в хранилище) и «Локальная» (как сейчас); флаг `domain_auth` в `/health`; колонка «Тип» в списке аккаунтов Kraken Admin.

**Architecture:** серверная часть аутентификации уже существует (локальная проверка → AD → JIT). Хаб получает статус LDAP из `/health`, а доменный вход идёт по существующему `configure_remote_local` (POST /auth/sessions); учётные данные сохраняются существующим механизмом программных аккаунтов (QSettings + Windows keyring), автоподключение при следующих запусках делает штатный `_connect_saved_server`. Диалог входа перестраивается на `QTabWidget`; Qt-независимая логика выносится в тестируемые помощники.

**Tech Stack:** Python 3.11+, PyQt6 (хаб, админ), FastAPI (сервер), urllib (HTTP-клиент хаба), pytest. Новых зависимостей нет.

**Spec:** `docs/superpowers/specs/2026-11-11-domain-login-window-design.md`

## Global Constraints

- UI-тексты — на русском (конвенция репозитория).
- Вкладка по умолчанию — **«Доменная учётная запись»** (решение пользователя); локальная — вторая вкладка.
- Доменная вкладка показывается только когда `service.has_accounts` (диалог первого запуска — без изменений).
- `configure_remote_local` (`src/kraken_hub/composition/catalog.py`) и её закреплённый тест `tests/test_no_auto_registration.py` **не менять**.
- Тесты не требуют сети, реального AD или реального сервера (фейки/monkeypatch).
- Коммит после каждой задачи; коммиты — на текущей ветке, как в репозитории (короткие заголовки, стиль существующих).
- Проверка после задач: `C:\code\kraken\.venv\Scripts\python.exe -m pytest <файлы тестов> -v` и `C:\code\kraken\.venv\Scripts\python.exe -m ruff check <изменённые файлы>`.

## Review Focus

1. **Сервер недоступен / опечатка в адресе** — статус «сервер недоступен» появляется не позже ~3 с; «Войти» немедленно отклоняет попытку с понятным сообщением; диалог не блокируется, локальная вкладка доступна. → тесты: Task 2 (URLError/таймаут → `unreachable`), Task 3 (reject-путь).
2. **Сервер отвечает, но LDAP выключен** (актуальное состояние локальной машины до появления `ldap-bind.secret`) — статус «доменная авторизация не включена», «Войти» отклоняется без обращения к паролю. → тест: Task 3 `test_domain_submit_rejects_when_ldap_disabled`.
3. **GitLab-токен с тем же логином** — upsert не перетирает секрет `gitlab`-аккаунта; создаётся отдельный `local`-аккаунт. → тест: Task 2.
4. **Windows Hello отменено / нет сохранённых локальных данных** после успешного доменного входа — диалог переключается на вкладку «Локальная» с подсказкой, серверная сессия уже установлена; закрытие диалога ничего не теряет (пароль сохранён — следующий старт подключится). → не юнит-тестится (нативный диалог нельзя симулировать); ветка «нет данных» покрывается дизайном `_try_local_autologin` (любое исключение → None); живой сценарий проверить приёмочно.
5. **Доменный пароль изменён после сохранения** — автоподключение при старте падает с существующим предупреждением «Сохранённое подключение … не восстановлено»; обход: Настройки → Учетная запись — удалить/добавить аккаунт с новым паролем и перезапустить хаб. Ограничение v1 зафиксировано; отдельного кода не требуется.

---

### Task 1: Сервер — флаг `domain_auth` в `/api/v1/health`

**Files:**
- Modify: `src/kraken_server/interface/http/routes/auth.py` (функция `health()`, строки ~29-35)
- Test: `tests/test_ldap_http.py` (дописать)

**Interfaces:**
- Consumes: `ctx.domain_auth` уже есть в замыкании `register()` (строка 22); `create_app(..., domain_auth=broker)` — существующий параметр (см. `test_domain_login_end_to_end`).
- Produces: поле `domain_auth: bool` в JSON-ответе `GET /api/v1/health` (`True`, когда broker настроен, иначе `False`). Task 2 читает его через `domain_server_status`.

- [ ] **Step 1: Write the failing tests**

В `tests/test_ldap_http.py` (паттерны — соседние тесты того же файла, фикстура `database`):

```python
def test_health_reports_domain_auth_enabled(database):
    services, accounts, _owner, _config = database
    fake = FakeLdap({"e2e": _ldap_identity("e2e", display="E2E User", email="e2e@inme")})
    broker = make_broker(accounts, services.identities, fake, PostgresPerformerStore(services.engine))
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=HybridSessionResolver(accounts, services.identities, None,
                                                                          performers=PostgresPerformerStore(services.engine)),
                                   project_access_mode="acl", domain_auth=broker))
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["domain_auth"] is True


def test_health_reports_domain_auth_disabled(database):
    services, accounts, _owner, _config = database
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=HybridSessionResolver(accounts, services.identities, None,
                                                                          performers=PostgresPerformerStore(services.engine)),
                                   project_access_mode="acl"))
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["domain_auth"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_ldap_http.py::test_health_reports_domain_auth_enabled tests/test_ldap_http.py::test_health_reports_domain_auth_disabled -v`
Expected: FAIL — ключ `domain_auth` отсутствует в ответе (`KeyError` / `assert False is True`-подобное).

- [ ] **Step 3: Implement the flag**

В `health()` (после строки с `status["blob_gateway"]`):

```python
    status["domain_auth"] = domain_auth is not None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_ldap_http.py -v`
Expected: PASS (все тесты файла, включая существующие).

- [ ] **Step 5: Commit**

```bash
git add src/kraken_server/interface/http/routes/auth.py tests/test_ldap_http.py
git commit -m "feat(server): report domain_auth flag in /api/v1/health"
```

---

### Task 2: Хаб — Qt-независимые помощники доменного входа

**Files:**
- Create: `src/kraken_hub/presentation/qt/parts/domain_login.py`
- Test: `tests/test_domain_login_helpers.py` (создать)

**Interfaces:**
- Consumes: `RemoteHttpClient` (`src/kraken_hub/infrastructure/remote/client.py`), `RemoteHttpClient.request(method, path, *, token, payload=None, headers=None, timeout=None)` (бросает `RemoteServerError(message, *, status, ...)`; сетевые ошибки — `status=None`); `load_program_accounts/save_program_accounts/program_settings/program_secret_store/ProgramAccount` из `kraken_hub.presentation.qt.preferences_dialog`; `KeyringSecretStore` (`get/set/delete`).
- Produces (использует Task 3, импорты в `login.py`):
  - Константы `STATUS_ENABLED = "enabled"`, `STATUS_DISABLED = "disabled"`, `STATUS_UNREACHABLE = "unreachable"`.
  - `domain_server_status(url: str, *, timeout: float = 3.0, http_factory: Callable[[str], Any] | None = None) -> str` — один из трёх статусов.
  - `upsert_program_account(username: str, password: str, *, display_name: str = "", settings: QSettings | None = None, secrets: KeyringSecretStore | None = None) -> ProgramAccount` — обновляет только `local`-аккаунт с тем же логином (без учёта регистра) либо добавляет новый; результат всегда `is_default=True`; `gitlab`-аккаунты не трогаются.
  - `domain_login_error_text(exc: BaseException) -> str` — `RemoteServerError` с `status is None` → `"Сервер недоступен"`; иначе `str(exc)` (detail сервера).

- [ ] **Step 1: Write the failing tests**

Создать `tests/test_domain_login_helpers.py`. В начале: `pytest.importorskip("PyQt6")`. Класс `_MemorySecrets` и `_settings(path)` — скопировать паттерн из `tests/test_program_account_secrets.py` (стр. 16-31). Фейковый клиент:

```python
class _FakeHealthClient:
    def __init__(self, payload=None, error=None):
        self._payload, self._error = payload, error
    def request(self, method, path, *, token, payload=None, headers=None, timeout=None):
        assert (method, path) == ("GET", "/api/v1/health")
        if self._error is not None:
            raise self._error
        return self._payload


def _status(url, client, **kw):
    return domain_server_status(url, http_factory=lambda base: client, **kw)
```

Тесты:

```python
def test_status_enabled_when_server_reports_domain_auth():
    assert _status("http://127.0.0.1:8080", _FakeHealthClient({"status": "ok", "domain_auth": True})) == "enabled"

def test_status_disabled_when_domain_auth_off_or_missing():
    assert _status("http://127.0.0.1:8080", _FakeHealthClient({"status": "ok", "domain_auth": False})) == "disabled"
    assert _status("http://127.0.0.1:8080", _FakeHealthClient({"status": "ok"})) == "disabled"

def test_status_unreachable_on_server_or_network_errors():
    from kraken_hub.application.remote_error import RemoteServerError
    assert _status("http://127.0.0.1:9", _FakeHealthClient(error=RemoteServerError("degraded", status=503))) == "unreachable"
    assert _status("http://127.0.0.1:9", _FakeHealthClient(error=RemoteServerError("network", status=None))) == "unreachable"
    assert _status("http://127.0.0.1:9", _FakeHealthClient(error=OSError("refused"))) == "unreachable"

def test_status_unreachable_on_empty_or_malformed_url():
    assert domain_server_status("", http_factory=lambda base: _FakeHealthClient({"domain_auth": True})) == "unreachable"
    assert domain_server_status("not a url", http_factory=lambda base: _FakeHealthClient({"domain_auth": True})) == "unreachable"
```

Upsert (временный ini + `_MemorySecrets`):

```python
def test_upsert_creates_default_local_account(tmp_path):
    settings, secrets = _settings(tmp_path / "hub.ini"), _MemorySecrets()
    account = upsert_program_account("shustinsky.m.a", "pw1", display_name="Mark", settings=settings, secrets=secrets)
    assert account.provider == "local" and account.is_default
    assert secrets.get(account.account_id) == b"pw1"
    loaded = load_program_accounts(settings, secrets)
    assert [a.username for a in loaded] == ["shustinsky.m.a"]

def test_upsert_updates_existing_local_account_password(tmp_path):
    settings, secrets = _settings(tmp_path / "hub.ini"), _MemorySecrets()
    first = upsert_program_account("mark", "old", settings=settings, secrets=secrets)
    upsert_program_account("MARK", "new", settings=settings, secrets=secrets)
    loaded = load_program_accounts(settings, secrets)
    assert [a.username for a in loaded] == ["mark"]
    assert loaded[0].secret == "new"
    assert loaded[0].account_id == first.account_id

def test_upsert_keeps_gitlab_account_with_same_login(tmp_path):
    settings, secrets = _settings(tmp_path / "hub.ini"), _MemorySecrets()
    save_program_accounts([ProgramAccount("gl-1", "gitlab", "shustinsky.m.a", "Mark", "glpat-1", False)], settings, secrets)
    upsert_program_account("shustinsky.m.a", "adpw", settings=settings, secrets=secrets)
    loaded = load_program_accounts(settings, secrets)
    gl = next(a for a in loaded if a.provider == "gitlab")
    local = next(a for a in loaded if a.provider == "local")
    assert gl.secret == "glpat-1" and local.secret == "adpw" and local.is_default
```

Маппинг ошибок:

```python
def test_error_text_maps_status_and_network():
    from kraken_hub.application.remote_error import RemoteServerError
    assert domain_login_error_text(RemoteServerError("Доменная авторизация временно недоступна", status=503)) == "Доменная авторизация временно недоступна"
    assert domain_login_error_text(RemoteServerError("x", status=None)) == "Сервер недоступен"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_domain_login_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError` / импорт не найден.

- [ ] **Step 3: Implement `domain_login.py`**

Модуль без Qt-виджетов (только QSettings-типы в сигнатурах). Ключевые решения, которые не вытекают из тестов:

- `domain_server_status`: нормализовать `url = url.strip().rstrip("/")`; пустой или не начинается с `http://`/`https://` → сразу `STATUS_UNREACHABLE`. `client = (http_factory or RemoteHttpClient)(url)`; `payload = client.request("GET", "/api/v1/health", token="", timeout=timeout)`. `isinstance(payload, Mapping) and payload.get("domain_auth") is True` → `STATUS_ENABLED`, иначе → `STATUS_DISABLED`. Любой `Exception` (в т.ч. `RemoteServerError` любой `status`, `OSError`) → `STATUS_UNREACHABLE`.
- `upsert_program_account`: `source = settings or program_settings()`, `store = secrets or program_secret_store()`; `accounts = load_program_accounts(source, store)`; поиск первого `provider == "local"` и `username.casefold() == target.username.casefold()`; замена на `ProgramAccount(тот же account_id, "local", исходный username аккаунта, display_name or старый display_name, password, is_default=True)` (username сохраняется в том виде, как был записан ранее — регистр ввода при обновлении не меняет его) либо добавление `ProgramAccount(str(uuid4()), "local", username, display_name or username, password, is_default=True)`; `save_program_accounts(accounts, source, store)`; вернуть upserted-аккаунт.
- `domain_login_error_text`: `RemoteServerError` (импорт из `kraken_hub.application.remote_error`): `exc.status is None` → `"Сервер недоступен"`, иначе `str(exc)`; прочее → `str(exc)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_domain_login_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Ruff + commit**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m ruff check src/kraken_hub/presentation/qt/parts/domain_login.py tests/test_domain_login_helpers.py`
Expected: нет новых замечаний (база репозитория не ухудшается).

```bash
git add src/kraken_hub/presentation/qt/parts/domain_login.py tests/test_domain_login_helpers.py
git commit -m "feat(hub): domain login helpers — server status, account upsert, error text"
```

---

### Task 3: Хаб — окно входа на вкладках + тестовый каркас

**Files:**
- Modify: `src/kraken_hub/presentation/qt/parts/login.py` (диалог `loginDialog`; диалог `initialAccountDialog` не трогать)
- Modify: `tests/helpers/qt_harness.py` (`_sign_in`, строки ~606-643)
- Test: `tests/test_hub_login_window.py` (создать)

**Interfaces:**
- Consumes: Task 2 — `domain_server_status`, `upsert_program_account`, `domain_login_error_text`, константы статусов (импортировать в `login.py` так, чтобы тесты могли monkeypatch `kraken_hub.presentation.qt.parts.login.<имя>`); `program_server_url/save_program_server_url` из `preferences_dialog`; `service.configure_remote_local(base_url=, username=, password=)` и `service.login(username, password)` / `service.has_accounts`; `RemoteServerError` из `kraken_hub.application.remote_error`; паттерн Windows-верификации из `development_session._autofill_credentials` (`windows_credentials.credentials_available/load_credentials/credential_target`, `windows_hello.verify_windows_identity`).
- Produces (object-name контракты, использует тестовый каркас и Task 4-ревью):
  - `QTabWidget` objectName `loginTabs`; вкладка 0 — «Доменная учётная запись», вкладка 1 — «Локальная»; **текущая по умолчанию — 0**.
  - Доменная вкладка: `loginDomainServer` (QLineEdit, предзаполнен из `program_server_url()`), `loginDomainStatus` (QLabel), `loginDomainUsername` (QLineEdit, предзаполнен `os.environ.get("USERNAME", "")`), `loginDomainPassword` (QLineEdit, Password echo), `loginDomainSubmit` (**QPushButton**, текст «Войти», `setAutoDefault(False)` — НЕ QDialogButtonBox).
  - Локальная вкладка: текущие виджеты без изменений (`loginUsername`, `loginPassword`, `loginAutofill`, QDialogButtonBox с Ok «Войти») + `loginLocalHint` (QLabel, скрыт, текст «Доменный вход выполнен. Завершите локальный вход.»).
  - Диалог закрывается с сессией в обоих путях; `_login(parent, service) -> DesktopSession | None` — сигнатура не меняется.

- [ ] **Step 1: Update the Qt harness (failing state)**

В `tests/helpers/qt_harness.py`: импорт `QTabWidget` (в блок PyQt6-импортов, строки ~62-70) и в `_sign_in` — внутри `poll()`, до заполнения полей:

```python
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        if tabs is not None:
            tabs.setCurrentIndex(1)  # локальная вкладка — путь существующих тестов
```

Причина: вкладка по умолчанию теперь доменная; тесты, гоняющие локальный вход, должны явно выбирать локальную. QDialogButtonBox в диалоге один (в локальной вкладке), поэтому клик по кнопке остаётся рабочим.

Run (проверка, что каркас ещё не сломан до изменения диалога — тесты должны проходить и сейчас):
`C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_qt_harness.py -v`
Expected: PASS.

- [ ] **Step 2: Write the failing dialog tests**

Создать `tests/test_hub_login_window.py` (паттерн запуска/поиска модального диалога — `qt_harness._sign_in` и `tests/test_qt_harness.py`; фикстура `qapp`):

```python
def test_login_window_domain_tab_default_and_local_login_completes(qapp, tmp_path, monkeypatch, service_with_account):
    """service_with_account: локальный аккаунт уже создан (create_initial_account)."""
    import kraken_hub.presentation.qt.parts.login as login_parts
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "disabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    from PyQt6.QtCore import QTimer
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QDialog, QTabWidget
    seen = {}
    timer = QTimer(); timer.setInterval(25)
    def poll():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QDialog) and dialog.objectName() == "loginDialog" and not seen:
            tabs = dialog.findChild(QTabWidget, "loginTabs")
            seen["tabs"] = tabs
            seen["default"] = tabs.currentIndex()
            seen["domain_username"] = dialog.findChild(QLineEdit, "loginDomainUsername").text()
            seen["submit_kind"] = type(dialog.findChild(QPushButton, "loginDomainSubmit")).__name__
            tabs.setCurrentIndex(1)
            dialog.findChild(QLineEdit, "loginUsername").setText("mark")
            dialog.findChild(QLineEdit, "loginPassword").setText("pw")
            box = dialog.findChild(QDialogButtonBox)
            QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
    timer.timeout.connect(poll); timer.start()
    try:
        session = login_parts._login(None, service_with_account)
    finally:
        timer.stop()
    assert session is not None
    assert seen["default"] == 0
    assert seen["tabs"].tabText(0) == "Доменная учётная запись"
    assert seen["domain_username"] == os.environ.get("USERNAME", "")
    assert seen["submit_kind"] == "QPushButton"
```

Вторая проверка — reject-путь (LDAP выключен):

```python
def test_domain_submit_rejects_when_ldap_disabled(qapp, tmp_path, monkeypatch, service_with_account):
    """Клик по loginDomainSubmit при status 'disabled' — предупреждение, диалог открыт, локальный вход завершает сессию."""
    # тот же каркас; после предупреждения (QMessageBox, кнопка OK — паттерн _dialog_buttons из qt_harness)
    # диалог должен оставаться открытым; затем локальный вход → сессия.
    # assert: текст предупреждения содержит "доменная авторизация не включена"
```

Фикстура `service_with_account` — модульный `conftest`-уровневый помощник в этом файле: собрать сервис как в `qt_harness` (`build_workspace_service` из `kraken_hub.composition.catalog`, `tmp_path`-хранилище) и вызвать `service.create_initial_account(username="mark", display_name="Mark", password="pw")`. Ориентир — то, как `tests/test_qt_harness.py` готовит сервис для `_sign_in`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_hub_login_window.py -v`
Expected: FAIL — `loginTabs` не найден (диалог ещё без вкладок).

- [ ] **Step 4: Implement the tabbed dialog**

Перестроить только ветку `loginDialog` в `login.py` (ветка `initialAccountDialog` — как есть):

Структура диалога:

```python
    dialog = QDialog(parent)
    dialog.setObjectName("loginDialog")
    dialog.setWindowTitle("Вход в Kraken")
    layout = QVBoxLayout(dialog)
    tabs = QTabWidget(dialog)
    tabs.setObjectName("loginTabs")
    tabs.addTab(_build_domain_tab(), "Доменная учётная запись")
    tabs.addTab(_build_local_tab(service), "Локальная")
    layout.addWidget(tabs)
```

`_build_domain_tab()` — страница с QFormLayout: `loginDomainServer` (placeholder `http://127.0.0.1:8080`, текст `program_server_url()`), строка статуса `loginDomainStatus`, `loginDomainUsername` (текст `os.environ.get("USERNAME", "")`), `loginDomainPassword` (Password), кнопка `loginDomainSubmit`.

`_build_local_tab(service)` — текущее содержимое ветки loginDialog (заголовок «Локальная сессия рабочей станции», метка хранилища `loginDataDirectory`, `loginUsername`, `loginPassword`, QDialogButtonBox, `loginAutofill`) + `loginLocalHint` (`setVisible(False)`).

Ссылки на виджеты вкладок в `_login` берутся через `findChild` по objectName (единожды, после сборки `tabs`) — так же, как тесты находят их.

Привязка поведения (в `_login`, где доступны `service` и `dialog`):

```python
    def refresh_status(force: bool = False) -> str:
        url = server.text().strip().rstrip("/")
        if not url:
            status_label.setText("Укажите адрес сервера")
            return STATUS_UNREACHABLE
        state = domain_server_status(url)
        status_label.setText({STATUS_ENABLED: "Доменная авторизация включена",
                              STATUS_DISABLED: "Доменная авторизация не включена",
                              STATUS_UNREACHABLE: "Сервер недоступен"}[state])
        return state
```

- перед `dialog.exec()`: если `program_server_url()` не пуст — `refresh_status()` (блокирует ≤3 с), иначе метка «Укажите адрес сервера».
- `server.editingFinished` → `refresh_status()`.
- `submit.clicked` (синхронно, как существующий локальный поток):
  1. пустой URL → `QMessageBox.warning("Укажите адрес сервера")`; неверный префикс → «Адрес сервера должен начинаться с http:// или https://»; return.
  2. `state = refresh_status()` (принудительно); `unreachable` → warning «Сервер недоступен»; `disabled` → warning «На сервере не включена доменная авторизация»; return (пароль не используется).
  3. пустой логин/пароль → warning; return.
  4. `service.configure_remote_local(base_url=url, username=username, password=entered)`:
     - `ValueError` → (URL уже валиден) 401: warning «Неверный доменный пароль», `password.clear()`, return.
     - `RemoteServerError` → warning `domain_login_error_text(exc)`, return.
  5. Успех: `upsert_program_account(username, entered)` (display_name по умолчанию = username; серверное отображаемое имя придёт с сессией), `save_program_server_url(url)`.
  6. Локальная часть: `_try_local_autologin(dialog, service)` —
     ```python
     def _try_local_autologin(dialog, service):
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
     ```
     - есть сессия → `domain_session = session`; `password.clear()`; `dialog.accept()`.
     - нет → `hint.setVisible(True)`, `tabs.setCurrentIndex(1)`, `username_local.setFocus()`.

- Основной цикл:

```python
    domain_session = None
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
```

Импорт `RemoteServerError` — в функции (как другие тяжёлые импорты в этом модуле), чтобы не тянуть сеть-модули на старте.

- [ ] **Step 5: Run new tests + harness-dependent suite**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_hub_login_window.py tests/test_qt_harness.py -v`
Expected: PASS.

Run (приёмочные тесты, гоняющие реальный диалог через обновлённый каркас):
`C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_hub_manager_app_parts/ -v`
Expected: PASS (известный предсуществующий шум: 6 WinError-145 в `tests/test_admin_project_workflows.py` — не в этом каталоге).

- [ ] **Step 6: Ruff + commit**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m ruff check src/kraken_hub/presentation/qt/parts/login.py tests/helpers/qt_harness.py tests/test_hub_login_window.py`
Expected: нет новых замечаний.

```bash
git add src/kraken_hub/presentation/qt/parts/login.py tests/helpers/qt_harness.py tests/test_hub_login_window.py
git commit -m "feat(hub): tabbed login window with domain (LDAP) sign-in"
```

---

### Task 4: Kraken Admin — колонка «Тип» в списке аккаунтов

**Files:**
- Modify: `src/kraken_admin/application/accounts.py` (добавить функцию)
- Modify: `src/kraken_admin/presentation/qt/window.py` (`_build_accounts_tab` строки ~160-171, `_fill_accounts` строки ~260-276)
- Test: `tests/test_admin_app.py` (поправить существующие ассерты, добавить новый тест)

**Interfaces:**
- Consumes: `account_rows` уже отдаёт `auth_provider` (стр. 140); `LocalAccount` — dataclass, 6-й позиционный аргумент `auth_provider`.
- Produces: `account_type_label(auth_provider: object) -> str` — `"доменная"` при `"ldap"`, иначе `"локальная"`. Таблица: 5 колонок, заголовки `("Отображаемое имя", "Логин", "Тип", "Состояние", "Создано")`, «Тип» — колонка 2.

- [ ] **Step 1: Write the failing tests**

В `tests/test_admin_app.py`:

1. Поправить существующий `test_admin_window_lists_accounts_projects_and_revokes_maintainer`: `columnCount() == 5`; `horizontalHeaderItem(2).text() == "Тип"`; сдвинуть прежние ассерты «Создано» с колонки 3 на 4; добавить `assert window.accounts.item(0, 2).text() == "локальная"`.
2. Новый тест (паттерн — соседние тесты этого файла: `LocalAccountStore` + `window._accounts = store` + `_fill_accounts`):

```python
def test_admin_window_marks_ldap_accounts(qapp, tmp_path: Path) -> None:
    store = LocalAccountStore(tmp_path / "accounts.sqlite3", ScryptPasswordHasher())
    local = store.create_account("mark", "Mark", "pw")
    ldap = LocalAccount(str(uuid4()), "shustinsky.m.a", "Шустинский М.", True,
                        "2026-11-11T00:00:00+00:00", "ldap")
    window = ...  # как в соседнем тесте
    window._accounts = store
    monkeypatch store.list_accounts -> lambda include_disabled=True: (ldap, local)
    window._fill_accounts()
    assert window.accounts.item(0, 2).text() == "доменная"
    assert window.accounts.item(1, 2).text() == "локальная"
```

3. Юнит-тест маппинга (в тот же файл):

```python
def test_account_type_label():
    from kraken_admin.application.accounts import account_type_label
    assert account_type_label("ldap") == "доменная"
    assert account_type_label("local") == "локальная"
    assert account_type_label(None) == "локальная"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_admin_app.py -v`
Expected: FAIL — `account_type_label` не существует / columnCount 4 ≠ 5.

- [ ] **Step 3: Implement**

`src/kraken_admin/application/accounts.py`:

```python
def account_type_label(auth_provider: object) -> str:
    """Человекочитаемый тип учётной записи для таблиц."""
    return "доменная" if str(auth_provider) == "ldap" else "локальная"
```

`window.py`: `QTableWidget(0, 5, page)`; заголовки `("Отображаемое имя", "Логин", "Тип", "Состояние", "Создано")`; в `_fill_accounts` — колонка 2: `QTableWidgetItem(account_type_label(value["auth_provider"]))`, «Создано» — колонка 4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m pytest tests/test_admin_app.py -v`
Expected: PASS.

- [ ] **Step 5: Ruff + commit**

Run: `C:\code\kraken\.venv\Scripts\python.exe -m ruff check src/kraken_admin/application/accounts.py src/kraken_admin/presentation/qt/window.py tests/test_admin_app.py`

```bash
git add src/kraken_admin/application/accounts.py src/kraken_admin/presentation/qt/window.py tests/test_admin_app.py
git commit -m "feat(admin): show account type (local/domain) in the accounts table"
```

---

## После всех задач (приёмка на живой машине)

1. Перезапустить Kraken Hub; в окне входа: вкладка «Доменная» по умолчанию, статус по `http://127.0.0.1:8080` — «Доменная авторизация не включена» (пока нет `ldap-bind.secret` от ИТ) — «Войти» отклоняет с этим сообщением; вкладка «Локальная» работает как раньше.
2. После появления DPAPI-файла `ldap-bind.secret` и рестарта сервера: статус «включена», доменный вход создаёт JIT-аккаунт, в Kraken Admin он виден в списке с «Тип: доменная».
3. Второй запуск хаба: локальный вход → автоматическое подключение к серверу без повторного ввода доменного пароля.
