# Any-Account Sign-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** после успешного доменного/серверного входа Kraken Hub завершает вход сам — входом в зеркальную локальную учётную запись, без шага «Завершите локальный вход».

**Architecture:** новый сервисный метод `login_or_create_local` на `EmbeddedProjectService` (там же, где `login`/`create_initial_account`) создаёт локальную учётку при её отсутствии и входит; `DualCatalogService` делегирует; финал диалога `complete_remote_sign_in` вызывает его после неудачи `_try_local_autologin`.

**Tech Stack:** Python 3.14, PyQt6 (offscreen-тесты), pytest, sqlite `LocalAccountStore`.

**Spec:** `docs/superpowers/specs/2026-11-17-any-account-login-design.md`

## Global Constraints

- Пользовательские строки диалога — на русском, существующие тексты не менять.
- Новые зависимости запрещены.
- `tests/test_no_auto_registration.py` (запрет авторегистрации на сервере) не трогать.
- Существующий локальный пароль никогда не перезаписывается.
- Тесты гонять `.venv\Scripts\python.exe -m pytest` (в PATH pytest нет), `QT_QPA_PLATFORM=offscreen` уже ставится тестовыми модулями.

## Review Focus

- Удалённый логин недопустим как имя локального аккаунта (пробелы, пустая строка) → `None`, подсказка и вкладка «Локальная» остаются.
- Локальная учётка с таким логином уже есть с другим паролем → `None`, старый пароль продолжает работать.
- Регистронезависимость: «OWNER» и «owner» — одна учётка, дубль не создаётся (`accounts` NOCASE).
- Приоритет сохранённой локальной сессии (`_try_local_autologin`) перед зеркалом сохраняется — вызовы идут в этом порядке.
- Повторный запуск: зеркальная учётка работает как обычная локальная (`service.login(...)` тем же паролем).

---

### Task 1: сервис `login_or_create_local` + делегирование + протокол

**Files:**
- Modify: `src/kraken_hub/composition/parts/revoke_project_role.py` (рядом с `login`, ~line 248)
- Modify: `src/kraken_hub/composition/catalog.py` (рядом с `login`, ~line 562)
- Modify: `src/kraken_hub/application/desktop_operations.py` (после `login`, line 118)
- Test: `tests/test_local_mirror_sign_in.py` (создать)

**Interfaces:**
- Consumes: `LocalAccountStore.get_by_username/create_account`, `EmbeddedProjectService.login` (уже есть).
- Produces: `login_or_create_local(self, username: str, password: str) -> DesktopSession | None` на `EmbeddedProjectService` и `DualCatalogService`; запись в протоколе `DesktopOperations`.

- [ ] **Step 1: написать падающие тесты** `tests/test_local_mirror_sign_in.py`

Пять тестов на `EmbeddedProjectService(tmp_path)` (паттерн — `tests/test_no_auto_registration.py`):

```python
def test_mirror_account_created_and_sign_in_works(...)   # есть только "owner"; login_or_create_local("alice","pw")
                                                          # → session, principal.subject == "alice";
                                                          # service.login("alice","pw") is not None; account_count == 2
def test_existing_local_account_with_matching_password_signs_in_without_duplicate(...)
                                                          # login_or_create_local("OWNER","secret") → session;
                                                          # account_count == 1 (NOCASE, без дубля)
def test_existing_local_account_with_other_password_returns_none(...)
                                                          # login_or_create_local("owner","other") is None;
                                                          # service.login("owner","secret") is not None
def test_invalid_username_returns_none(...)               # login_or_create_local("bad name","pw") is None;
                                                          # account_count == 1
def test_catalog_delegates_to_local(...)                  # DualCatalogService(EmbeddedProjectService(...))
                                                          # .login_or_create_local("alice","pw") → session
```

- [ ] **Step 2: прогнать — убедиться, что падают** (`AttributeError: login_or_create_local`)

Run: `.venv\Scripts\python.exe -m pytest tests/test_local_mirror_sign_in.py -q`

- [ ] **Step 3: реализовать** `login_or_create_local` в `RevokeProjectRoleMixin`
  (`composition/parts/revoke_project_role.py`, сразу после `login`): `try: get_by_username; если None — create_account(username, username, password); except ValueError → None`; затем `return self.login(username, password)`. Docstring: зеркало удалённо проверенного входа; чужой локальный пароль не перезаписывается.

- [ ] **Step 4: делегирование** в `DualCatalogService` (`catalog.py`, сразу после `login`):
  `def login_or_create_local(self, username: str, password: str): return self.local.login_or_create_local(username, password)`

- [ ] **Step 5: протокол**: в `DesktopOperations` после `login` (desktop_operations.py:118):
  `def login_or_create_local(self, username: str, password: str) -> UserSession | None: ...`

- [ ] **Step 6: тесты зелёные**

Run: `.venv\Scripts\python.exe -m pytest tests/test_local_mirror_sign_in.py -q`

- [ ] **Step 7: коммит** `git add src/kraken_hub/composition/parts/revoke_project_role.py src/kraken_hub/composition/catalog.py src/kraken_hub/application/desktop_operations.py tests/test_local_mirror_sign_in.py` + `git commit -m "feat(hub): login_or_create_local mirrors a verified remote sign-in into a local account"`

---

### Task 2: финал диалога завершает вход без вкладки «Локальная»

**Files:**
- Modify: `src/kraken_hub/presentation/qt/parts/login.py` (`complete_remote_sign_in`, ~line 312)
- Modify: `tests/test_hub_login_window.py` (`test_server_sign_in_success_completes_session`, `test_server_sign_in_works_when_domain_auth_disabled`; новый тест)

**Interfaces:**
- Consumes: `service.login_or_create_local(username, password)` (Task 1); существующие `_try_local_autologin`, `_remember_credentials`, `upsert_program_account`.
- Produces: поведение диалога: успех удалённого входа закрывает диалог с сессией, если локальная учётка создана/найдена; иначе — прежняя подсказка и вкладка «Локальная».

- [ ] **Step 1: переписать два существующих теста на новое ожидание + добавить новый**

- `test_server_sign_in_success_completes_session`: после клика «Войти» (server-user/server-pw) диалог закрывается сразу — убрать фазу `expect_handoff`; `session is not None`; `calls["configure"/"upsert"/"url"]` как раньше; `service_with_account.login("server-user", "server-pw") is not None` (зеркало создано); monkeypatch `login_parts._remember_credentials` — зафиксировать вызов с `(server-user, server-pw)`.
- `test_server_sign_in_works_when_domain_auth_disabled`: то же упрощение (диалог закрывается, предупреждений нет).
- Новый `test_server_sign_in_keeps_local_tab_when_local_account_rejects_password`: логин `mark` (локальная учётка есть), серверный пароль `not-local-pw`; `_try_local_autologin` → None; ожидание: `loginLocalHint` виден, `tabs.currentIndex() == 2`, диалог открыт; затем локальный вход `mark`/`pw` завершает сессию (паттерн фаз из текущего `test_server_sign_in_success_completes_session`).

- [ ] **Step 2: прогнать — новые/обновлённые ожидания падают**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hub_login_window.py -q -k server_sign_in`

- [ ] **Step 3: реализовать в `complete_remote_sign_in`** (`login.py`): после `_try_local_autologin` при `session is None` — `session = service.login_or_create_local(entered_username, entered_password)`; при успехе `session` — `_remember_credentials(entered_username, entered_password, data_directory=service.data_dir)` (только на этом пути: авологиновый путь уже имеет сохранённые данные и не должен их перезаписать удалённым паролем); далее общий финал (`password_field.clear()`, `dialog.accept()`, return). Комментарий на русском: удалённо проверенного входа достаточно, зеркальная локальная учётка создаётся при отсутствии.

- [ ] **Step 4: весь файл диалога зелёный**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hub_login_window.py tests/test_hub_login_wiring.py tests/test_no_auto_registration.py -q`

- [ ] **Step 5: линт**

Run: `.venv\Scripts\python.exe -m ruff check src/kraken_hub tests/test_local_mirror_sign_in.py tests/test_hub_login_window.py`

- [ ] **Step 6: коммит** `git add src/kraken_hub/presentation/qt/parts/login.py tests/test_hub_login_window.py` + `git commit -m "feat(hub): a verified server/domain sign-in completes the hub login via a mirrored local account"`
