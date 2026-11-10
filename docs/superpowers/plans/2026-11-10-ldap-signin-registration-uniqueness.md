# Direct LDAP (AD) Sign-in, Explicit Registration, Login Uniqueness, Profile Groundwork — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kraken Server verifies domain (AD/LDAP) credentials at login time and JIT-provisions accounts; local self-registration becomes explicit and is blocked from the LAN; logins are unique across all identity sources; principals gain profile fields (login/avatar/status) with a minimal API.

**Architecture:** A stdlib-testable `LdapClient` (service-account search + user bind) verifies credentials server-side, login-time only. A `DomainAuthBroker` in the server composition routes each login to the local hash path or the LDAP path (strictly separate providers) and JIT-provisions `server_accounts` + `Principal(provider=ldap)` + linked performer. Login-claim semantics (`principals.login_key` unique) enforce global login uniqueness with corporate identity winning collisions. The desktop loses implicit auto-registration and gains an explicit "create on server" action.

**Tech Stack:** Python 3.14, FastAPI, SQLAlchemy Core + Alembic (PostgreSQL), `ldap3` (new dep, pure Python), PyQt6 (desktop), pytest.

**Spec:** `docs/superpowers/specs/2026-11-10-ldap-signin-registration-uniqueness-design.md`

## Global Constraints

- New dependency is `ldap3` ONLY (add to the `server` extra in `pyproject.toml`). No other new packages.
- ruff: line-length 120, target py314. Run `uv run ruff check` on changed files.
- User-facing strings are Russian, matching existing tone (see `routes/auth.py`, `preferences_dialog.py`).
- The LDAP bind password never appears in TOML, environment variables, logs, or the DB — DPAPI-protected secret file only (`protect_secret`/`unprotect_secret` pattern from `kraken_server/configuration.py`).
- AD is consulted **only at login**. No authenticated request may call the LDAP client.
- Providers are strictly separate: a local account is never verified against AD and an LDAP account never against the stored hash.
- One new Alembic revision, parent `20261029_0011` (the current single head).
- The existing GitLab provider, its client, and its tests are untouched.
- Integration tests use the `database` fixture from `tests/test_admin_project_workflows.py` (requires `KRAKEN_TEST_POSTGRES_URL`; skips otherwise). Unit tests are pure Python.
- `LocalAccount` (in `src/kraken_manager/infrastructure/auth/local.py`) is frozen with fields `(account_id, username, display_name, enabled, created_at)` — new fields must be appended with defaults so positional construction keeps working.

## Review Focus

Failure modes the spec implies but which need an explicit pinning test (added to the owning task):

- **LDAP injection / filter breaking via username** — a login containing `)`, `*`, `\`, or NUL must be escaped before filtering. → Task 2, `test_verify_escapes_username_in_filter`.
- **AD outage must not break existing sessions** — after a successful domain login, AD going down must not invalidate the issued token. → Task 7, `test_existing_session_survives_ad_outage`.
- **Admin-disabling an LDAP account blocks domain logins** — `set_enabled(False)` must gate the LDAP path too. → Task 6, `test_disabled_ldap_account_cannot_login`.
- **A local user whose login was claimed by a domain user keeps working** (no lockout in the reverse direction). → Task 6, `test_reverse_interleave_local_login_keeps_working`.
- **Concurrent/second JIT of the same domain user must not crash on the unique `username_key`** — the create-then-authenticate race re-fetches and still returns a session. → Task 6, `test_second_jit_for_same_domain_user_is_idempotent`.

---

### Task 1: Domain — `PrincipalProvider.LDAP`, profile fields, `login_key`

**Files:**
- Modify: `src/kraken_manager/domain/identity.py:24-156`
- Test: `tests/test_ldap_principal.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PrincipalProvider.LDAP = "ldap"`.
  - `Principal` gains optional fields `login: str | None = None`, `avatar_url: str | None = None`, `avatar_source: str | None = None` (only `"ldap"` or `"manual"`), `status_text: str | None = None` (all `None`-defaulted, appended after `system_roles`).
  - `Principal.login_key -> str | None` property: `None if self.login is None else self.login.casefold()`.
  - `Principal.ldap(*, issuer: str, subject: str, display_name: str, principal_id: PrincipalId | str | None = None, email: str | None = None, login: str | None = None, avatar_url: str | None = None, status_text: str | None = None) -> Principal` — `login` defaults to `subject`; `issuer` is required (validated like GITLAB).
  - `Principal.external_key` for LDAP: `f"ldap:{self.subject.casefold()}"` (LOCAL/GITLAB unchanged).

- [ ] **Step 1: Write the failing tests** in `tests/test_ldap_principal.py`:

```python
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.identity import Principal, PrincipalProvider


def test_ldap_principal_requires_issuer():
    try:
        Principal.ldap(subject="ivan", display_name="Ivan")
    except DomainValidationError:
        pass
    else:
        raise AssertionError("issuer is required for ldap principals")


def test_ldap_external_key_is_casefolded_login():
    p = Principal.ldap(issuer="ldap://dc:389", subject="Ivan.P", display_name="Ivan")
    assert p.external_key == "ldap:ivan.p"
    assert p.login == "Ivan.P"
    assert p.login_key == "ivan.p"


def test_profile_fields_validation():
    p = Principal.ldap(issuer="i", subject="u", display_name="U", status_text="x" * 256)
    # must raise DomainValidationError (status_text max 255)
```

(Write all three as real asserts: `pytest.raises(DomainValidationError)` for the validation cases; a valid principal with `avatar_url="https://x/y.png"`, `avatar_source="ldap"`, `login_key` property checks; an invalid `avatar_source="bogus"` raises.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ldap_principal.py -v`
Expected: FAIL (`LDAP` not in `PrincipalProvider`, `Principal.ldap` missing).

- [ ] **Step 3: Implement** in `identity.py`:

- `LDAP = "ldap"` in `PrincipalProvider`.
- New fields appended to the dataclass with `None` defaults; in `__post_init__`: LDAP requires `issuer` (same rule as GITLAB); `login` if not None → `require_non_empty(..., maximum=128)`; `avatar_url` if not None → `require_non_empty(..., maximum=2048)`; `avatar_source` if not None must be in `{"ldap", "manual"}`; `status_text` if not None must have `len <= 255` (empty string allowed).
- `login_key` property; `ldap` factory; `external_key` branch for LDAP.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ldap_principal.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kraken_manager/domain/identity.py tests/test_ldap_principal.py
git commit -m "feat(identity): ldap provider and principal profile fields"
```

---

### Task 2: `LdapClient` transport (`kraken_manager/infrastructure/auth/ldap.py`)

**Files:**
- Create: `src/kraken_manager/infrastructure/auth/ldap.py`
- Modify: `pyproject.toml` (`server` extra: add `"ldap3>=2.9"`)
- Test: `tests/test_ldap_client.py` (new)

**Interfaces:**
- Consumes: nothing from other tasks (self-contained module).
- Produces:
  - `DEFAULT_USER_FILTER = "(&(objectClass=user)(objectCategory=person)(sAMAccountName={username}))"`
  - `@dataclass(frozen=True, slots=True) LdapIdentity: username: str; display_name: str; email: str | None; avatar_url: str | None`
  - `class LdapUnavailable(RuntimeError)`, `class LdapAuthenticationError(RuntimeError)`
  - `resolve_ldap_tls(url: str, *, start_tls: bool, allow_insecure: bool) -> str` → `"ssl" | "starttls" | "plain"`; `ValueError` for plain `ldap://` without `start_tls`/`allow_insecure` and for unknown schemes.
  - Transport seam (protocol, defined in the module):
    ```python
    class LdapTransport(Protocol):
        def connect(self) -> None: ...            # service-account bind; LdapAuthenticationError on reject, LdapUnavailable on network/TLS failure
        def search_user(self, search_filter: str) -> tuple[str, dict[str, Any]] | None: ...  # (dn, attrs); None if no match; LdapUnavailable if ambiguous/error
        def user_bind(self, dn: str, password: str) -> bool: ...
        def close(self) -> None: ...
    ```
  - `class LdapClient`:
    ```python
    def __init__(self, url: str, *, base_dn: str, bind_dn: str, bind_password: str,
                 user_filter: str = DEFAULT_USER_FILTER, start_tls: bool = False,
                 allow_insecure: bool = False, ca_file: str | None = None,
                 timeout: float = 5.0, transport: LdapTransport | None = None) -> None
    issuer: str  # == url, consumed by the broker for Principal.ldap(issuer=...)
    def verify(self, username: str, password: str) -> LdapIdentity | None
    ```
  - `Ldap3Transport` — the real `ldap3`-based implementation of the seam (built when `transport is None` in `verify()`; uses `ldap3.Server/Connection/Tls`, `receive_timeout=timeout`, `filter_escape` for the username).

- [ ] **Step 1: Write the failing tests** in `tests/test_ldap_client.py`. `pytest.importorskip("ldap3")` is NOT required for the client tests (the fake transport is pure Python); the real `Ldap3Transport` construction is tested only behind `pytest.importorskip("ldap3")`.

```python
class FakeTransport:
    def __init__(self, *, service_ok=True, entries=(), user_bind_ok=True, error_on=None):
        self.service_ok, self.entries, self.user_bind_ok = service_ok, entries, user_bind_ok
        self.error_on = error_on  # "search" -> raise LdapUnavailable
        self.filters: list[str] = []
        self.user_binds: list[tuple[str, str]] = []
    def connect(self):
        if not self.service_ok:
            raise LdapAuthenticationError("service bind rejected")
    def search_user(self, f):
        self.filters.append(f)
        if self.error_on == "search":
            raise LdapUnavailable("ambiguous")
        return self.entries[0] if len(self.entries) == 1 else (None if not self.entries else _AMBIGUOUS)
    def user_bind(self, dn, password):
        self.user_binds.append((dn, password))
        return self.user_bind_ok
    def close(self):
        pass
```

Tests (write as real test functions):
- `test_resolve_ldap_tls_matrix` — `ldaps://dc:636` → `"ssl"`; `ldap://dc:389` + `start_tls=True` → `"starttls"`; `ldap://dc:389` + `allow_insecure=True` → `"plain"`; `ldap://dc:389` with neither → `pytest.raises(ValueError)`; `http://x` → `pytest.raises(ValueError)`.
- `test_verify_success_returns_identity` — entries `("CN=Ivan,DC=ad,DC=inme", {"sAMAccountName": "ivan", "displayName": "Ivan Petrov", "mail": "ivan@inme", "thumbnailPhoto": None})` → `LdapIdentity(username="ivan", display_name="Ivan Petrov", email="ivan@inme", avatar_url=None)`.
- `test_verify_wrong_password_returns_none` — `user_bind_ok=False` → `None`; assert the user bind was attempted with the found DN.
- `test_verify_unknown_user_returns_none` — `entries=()` → `None`; `user_binds == []`.
- `test_verify_ambiguous_search_raises_unavailable` — two entries → `pytest.raises(LdapUnavailable)`.
- `test_verify_service_bind_rejected_raises_authentication_error` — `service_ok=False` → `pytest.raises(LdapAuthenticationError)`.
- `test_verify_transport_error_propagates_and_closes` — fake whose `search_user` raises `LdapUnavailable` → `verify` raises the same; the fake records that `close()` was called (no leaked connections).
- `test_verify_escapes_username_in_filter` — username `a)(&*` → the recorded filter contains `sAMAccountName=a\29\28\26\2a` (i.e. escaped via `ldap3.utils.conv.filter_escape` — for the pure test, implement escaping in a module-level helper `_escape(value)` that the client calls, and assert on its output; the real transport and the client both go through it).
- `test_invalid_email_becomes_none` — `mail="not-an-email"` (no `@`) → `email is None`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ldap_client.py -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement** `ldap.py`:

- `verify()`: escape username → build filter via `self.user_filter.format(username=escaped)` → `transport.connect()` → `try: found = transport.search_user(f); ok = transport.user_bind(dn, password) if found else False  finally: transport.close()` → map attributes: `display_name = attrs.get("displayName") or attrs.get("cn") or username`; `email` only if it contains `@`; `avatar_url` stays `None` for AD (binary `thumbnailPhoto` is not converted in v1 — URL storage is a future feature).
- Constructor validates the URL scheme via `resolve_ldap_tls` (raises `ValueError` early on plain-without-opt-in).
- `Ldap3Transport`: straightforward `ldap3` wrapping per the protocol comments (builder decides exact ldap3 calls; behavior and exception mapping are pinned by the tests).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ldap_client.py -v`
Expected: PASS.

- [ ] **Step 5: Add the dependency** to `pyproject.toml` `server` extra (`"ldap3>=2.9",`), then:

Run: `uv sync --extra server --extra postgres` (or the repo's standard sync command)
Expected: resolves cleanly.

- [ ] **Step 6: Commit**

```bash
git add src/kraken_manager/infrastructure/auth/ldap.py pyproject.toml uv.lock tests/test_ldap_client.py
git commit -m "feat(auth): ldap client with injectable transport and typed errors"
```

---

### Task 3: Schema — migration, store MetaData, `LocalAccount.auth_provider`

**Files:**
- Create: `migrations/versions/20261110_0012_identity_logins_and_profiles.py`
- Modify: `src/kraken_manager/infrastructure/postgres/account_store.py:16-31` (`_account_tables`), `:87-96` (`_account`)
- Modify: `src/kraken_manager/infrastructure/postgres/identity_store.py:23-49` (`_identity_tables`), `:69-80` (`_principal`)
- Modify: `src/kraken_manager/infrastructure/auth/local.py:80-94` (`LocalAccount`)
- Test: `tests/test_ldap_schema.py` (new)

**Interfaces:**
- Consumes: `Principal` fields from Task 1.
- Produces:
  - `principals` columns: `login` (text, null), `login_key` (text, null, partial unique index `WHERE login_key IS NOT NULL`), `avatar_url` (text, null), `avatar_source` (text, null), `status_text` (text, null), `profile_updated_at` (timestamptz, null).
  - `server_accounts` column: `auth_provider` (text, not null, server_default `"local"`).
  - `LocalAccount` gains `auth_provider: str = "local"` (appended field).
  - `PostgresIdentityAclStore._principal(row)` populates `login`, `avatar_url`, `avatar_source`, `status_text` from the row (store-only columns `avatar_source`/`profile_updated_at` are NOT part of the domain except `avatar_source`, which Task 1 added).

- [ ] **Step 1: Write the failing tests** in `tests/test_ldap_schema.py` (use `pytest_plugins = ["tests.test_admin_project_workflows"]` and the `database` fixture):

```python
def test_new_account_defaults_to_local_provider(database):
    services, accounts, _owner, _config = database
    account = accounts.create_account("prov-check", "Prov Check", "pw")
    assert accounts.get_account(account.account_id).auth_provider == "local"


def test_login_key_unique_index_rejects_duplicates(database):
    services, accounts, _owner, _config = database
    from kraken_manager.domain.identity import Principal
    from kraken_manager.domain.common import PrincipalId
    a = Principal.local(subject="dup-user", display_name="A", principal_id="11111111-1111-1111-1111-111111111111", login="dup-user")
    services.identities.save(a)
    b = Principal.local(subject="dup-user-2", display_name="B", principal_id="22222222-2222-2222-2222-222222222222", login="DUP.user")
    import sqlalchemy as sa
    with pytest.raises(sa.exc.IntegrityError):
        services.identities.save(b)


def test_migration_backfills_local_principal_logins(tmp_path, monkeypatch):
    # Create a throwaway schema, upgrade to the PREVIOUS head, insert one
    # server_accounts row + matching principals row (old shape, no login),
    # upgrade to head, assert login/login_key are backfilled.
    # Follow the schema/alembic pattern of the `database` fixture
    # (tests/test_admin_project_workflows.py:28-63), but pin
    # command.upgrade(cfg, "20261029_0011") before seeding and
    # command.upgrade(cfg, "head") after.
```

(Write the backfill test fully: seed `INSERT INTO server_accounts (account_id, username, username_key, display_name, password_hash, enabled, failed_attempts, created_at) VALUES (gen_random_uuid(), 'backfill.user', 'backfill.user', 'BF', 'x', true, 0, now())` and `INSERT INTO principals (principal_id, external_key, provider, subject, display_name, enabled, system_roles) VALUES (SELECT account_id, 'local:backfill.user', 'local', 'backfill.user', 'BF', true, '[]' FROM server_accounts WHERE username='backfill.user')`; after `head` assert the principal row has `login='backfill.user'`, `login_key='backfill.user'`.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ldap_schema.py -v`
Expected: FAIL (no `auth_provider` attribute / no `login` column). If `KRAKEN_TEST_POSTGRES_URL` is unset the tests SKIP — set it (local Postgres) and re-run; a skip is not a pass.

- [ ] **Step 3: Implement**

- `LocalAccount`: append `auth_provider: str = "local"` after `created_at`.
- `account_store.py`: add `sa.Column("auth_provider", sa.Text, nullable=False, server_default="local")` to `server_accounts`; `_account()` passes `auth_provider=str(row["auth_provider"] or "local")`.
- `identity_store.py`: add the six columns to `principals` in `_identity_tables()`; `_principal()` reads them (missing → `None`).
- Migration file (revision `20261110_0012`, `down_revision = "20261029_0011"`), `upgrade()`:
  1. `op.add_column` the six `principals` columns.
  2. `op.create_index("ux_principals_login_key", "principals", ["login_key"], unique=True, postgresql_where=sa.text("login_key IS NOT NULL"))`.
  3. `op.add_column("server_accounts", sa.Column("auth_provider", sa.Text(), nullable=False, server_default="local"))`.
  4. Backfill via `op.execute(sa.text(...))`:
     ```sql
     UPDATE principals p
        SET login = a.username, login_key = a.username_key
      FROM server_accounts a
     WHERE p.provider = 'local'
       AND p.external_key = 'local:' || a.username
       AND p.login_key IS NULL
       AND NOT EXISTS (
           SELECT 1 FROM principals q
            WHERE q.login_key = a.username_key AND q.principal_id <> p.principal_id);
     ```
  5. Safety-net audit rows for skipped conflicts:
     ```sql
     INSERT INTO administration_audit (actor_id, action, target_account_id, details, recorded_at)
     SELECT NULL, 'login_claim_conflict_backfill', p.principal_id,
            jsonb_build_object('login', a.username), now()
      FROM server_accounts a
      JOIN principals p ON p.external_key = 'local:' || a.username
     WHERE p.login_key IS NULL
       AND EXISTS (SELECT 1 FROM principals q
                    WHERE q.login_key = a.username_key AND q.principal_id <> p.principal_id);
     ```
  `downgrade()`: drop the index, drop the six columns, drop `auth_provider`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ldap_schema.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add migrations/versions/20261110_0012_identity_logins_and_profiles.py src/kraken_manager/infrastructure tests/test_ldap_schema.py
git commit -m "feat(schema): principal logins/profiles and account auth_provider"
```

---

### Task 4: Account store — LDAP accounts, verifier-based authenticate, session issuing

**Files:**
- Modify: `src/kraken_manager/infrastructure/postgres/account_store.py:98-126` (`create_account`), `:164-175` (`reset_password`), `:177-225` (`authenticate`)
- Test: `tests/test_ldap_account_store.py` (new)

**Interfaces:**
- Consumes: Task 3 schema (`auth_provider` column).
- Produces (all on `PostgresAccountStore`):
  - `create_account(username, display_name, password, *, actor_id: str | None = None, auth_provider: str = "local") -> LocalAccount` (existing signature extended).
  - `create_ldap_account(username: str, display_name: str) -> LocalAccount` — inserts `auth_provider="ldap"`, `password_hash = self.hasher.hash(secrets.token_urlsafe(32))` (structurally unusable), `enabled=True`.
  - `authenticate(username, password, *, lifetime: timedelta = timedelta(hours=12), verifier: Callable[[str, str], bool] | None = None) -> LocalSession | None` — `verifier(stored_hash, password)` replaces `self.hasher.verify` when provided; all existing lockout behavior unchanged.
  - `issue_session(username: str, *, lifetime: timedelta = timedelta(hours=12)) -> LocalSession | None` — same as `authenticate` without any password check; still honors `enabled` and `blocked_until`; resets `failed_attempts` on success. Used only right after an external (LDAP) verification.
  - `record_audit(*, action: str, target_account_id: str | None = None, details: dict[str, object] | None = None, actor_id: str | None = None) -> None` — public wrapper that opens its own transaction and inserts an `administration_audit` row.

- [ ] **Step 1: Write the failing tests** in `tests/test_ldap_account_store.py` (`database` fixture):

```python
def test_create_ldap_account_hash_is_unusable(database):
    services, accounts, _owner, _config = database
    account = accounts.create_ldap_account("ldap.user", "LDAP User")
    assert account.auth_provider == "ldap"
    assert accounts.authenticate("ldap.user", "anything") is None
    ok = accounts.authenticate("ldap.user", "anything", verifier=lambda stored, pw: True)
    assert ok is not None


def test_authenticate_verifier_counts_failures_and_blocks(database):
    services, accounts, _owner, _config = database
    accounts.create_ldap_account("lock.user", "Lock User")
    for _ in range(5):
        assert accounts.authenticate("lock.user", "x", verifier=lambda s, p: False) is None
    # 6th attempt is inside the lockout window even with a good verifier
    assert accounts.authenticate("lock.user", "x", verifier=lambda s, p: True) is None


def test_issue_session_respects_enabled(database):
    services, accounts, _owner, _config = database
    accounts.create_ldap_account("iss.user", "Iss User")
    assert accounts.issue_session("iss.user") is not None
    accounts.set_enabled(accounts.get_by_username("iss.user").account_id, False)
    assert accounts.issue_session("iss.user") is None


def test_record_audit_persists(database):
    services, accounts, _owner, _config = database
    accounts.record_audit(action="test.action", details={"k": "v"})
    rows = accounts.administration_audit(limit=10)
    assert any(r["action"] == "test.action" and r["details"] == {"k": "v"} for r in rows)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ldap_account_store.py -v`
Expected: FAIL (`create_ldap_account` etc. missing).

- [ ] **Step 3: Implement** in `account_store.py` per the Interfaces block. Refactor the shared row-fetch/lockout/session-insert core of `authenticate()` into a private helper used by both `authenticate` and `issue_session` to avoid duplicating the lockout logic.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ldap_account_store.py tests/test_local_auth.py -v`
Expected: PASS (new tests + existing local-auth regression).

- [ ] **Step 5: Commit**

```bash
git add src/kraken_manager/infrastructure/postgres/account_store.py tests/test_ldap_account_store.py
git commit -m "feat(accounts): ldap-backed accounts and verifier-based authenticate"
```

---

### Task 5: Identity store — login claims, profile updates

**Files:**
- Modify: `src/kraken_manager/infrastructure/postgres/identity_store.py:111-132` (`save`)
- Test: `tests/test_ldap_identity_store.py` (new)

**Interfaces:**
- Consumes: Task 1 (`Principal.login`, `login_key`, `avatar_url`, `avatar_source`, `status_text`), Task 3 (columns).
- Produces (all on `PostgresIdentityAclStore`):
  - `save(principal)` — INSERT carries all fields including `login`, `login_key` (`principal.login_key`), `avatar_url`, `avatar_source`, `status_text`, `profile_updated_at`. On conflict (update) it touches ONLY: `external_key`, `provider`, `issuer`, `subject`, `display_name`, `email`, `enabled`, `system_roles`, `login`, `login_key` — never the avatar/status columns.
  - `find_by_login(login_key: str) -> Principal | None` — lookup by `login_key` (callers pass `login.casefold()`).
  - `sync_avatar(principal_id: PrincipalId, avatar_url: str | None) -> None` — `UPDATE ... SET avatar_url=:u, avatar_source='ldap', profile_updated_at=now() WHERE principal_id=:id AND COALESCE(avatar_source,'') <> 'manual'`.
  - `set_manual_avatar(principal_id: PrincipalId, avatar_url: str) -> None` — unconditional; sets `avatar_source='manual'`, `profile_updated_at`.
  - `set_status(principal_id: PrincipalId, status_text: str) -> None` — sets `status_text`, `profile_updated_at`.

- [ ] **Step 1: Write the failing tests** in `tests/test_ldap_identity_store.py` (`database` fixture; helper to build principals with fixed UUIDs):

```python
def test_save_and_find_by_login_is_case_insensitive(database):
    services, _accounts, _owner, _config = database
    p = Principal.local(subject="Case.User", display_name="CU", principal_id=_uuid(1), login="Case.User")
    services.identities.save(p)
    found = services.identities.find_by_login("case.user")
    assert found is not None and str(found.id) == str(p.id)


def test_sync_avatar_respects_manual_source(database):
    services, _a, _o, _c = database
    pid = _uuid(2)
    services.identities.save(Principal.local(subject="av.user", display_name="AV", principal_id=pid))
    services.identities.set_manual_avatar(pid, "https://me/photo.png")
    services.identities.sync_avatar(pid, "https://ad/photo.png")
    assert services.identities.get(pid).avatar_url == "https://me/photo.png"


def test_sync_avatar_updates_when_not_manual(database):
    services, _a, _o, _c = database
    pid = _uuid(3)
    services.identities.save(Principal.local(subject="av2.user", display_name="AV2", principal_id=pid))
    services.identities.sync_avatar(pid, "https://ad/photo.png")
    assert services.identities.get(pid).avatar_url == "https://ad/photo.png"


def test_set_status(database):
    services, _a, _o, _c = database
    pid = _uuid(4)
    services.identities.save(Principal.local(subject="st.user", display_name="ST", principal_id=pid))
    services.identities.set_status(pid, "на встрече")
    assert services.identities.get(pid).status_text == "на встрече"


def test_save_conflict_does_not_clobber_avatar(database):
    services, _a, _o, _c = database
    pid = _uuid(5)
    first = Principal.local(subject="cw.user", display_name="CW", principal_id=pid, avatar_url="https://x/1.png", avatar_source="ldap")
    services.identities.save(first)
    services.identities.save(Principal.local(subject="cw.user", display_name="CW renamed", principal_id=pid))
    stored = services.identities.get(pid)
    assert stored.display_name == "CW renamed" and stored.avatar_url == "https://x/1.png"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ldap_identity_store.py -v`
Expected: FAIL (methods missing).

- [ ] **Step 3: Implement** per the Interfaces block.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ldap_identity_store.py tests/test_ldap_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kraken_manager/infrastructure/postgres/identity_store.py tests/test_ldap_identity_store.py
git commit -m "feat(identities): login claims and profile field updates"
```

---

### Task 6: `DomainAuthBroker` + claim-aware local provisioning

**Files:**
- Create: `tests/ldap_fixtures.py` (shared fakes for Tasks 6-9)
- Modify: `src/kraken_server/composition.py` (new `DomainAuthBroker`; `HybridSessionResolver.provision_local_account` at :131-149)
- Test: `tests/test_domain_auth_broker.py` (new)

**Interfaces:**
- Consumes: Tasks 1-5 (`Principal.ldap`, `LdapClient`/`LdapIdentity`, `LdapUnavailable`/`LdapAuthenticationError`, account store methods, identity store methods).
- Produces:
  - `tests/ldap_fixtures.py`:
    ```python
    class FakeLdap:
        """Stand-in for LdapClient in server tests."""
        def __init__(self, identities: dict | None = None, *, verify_error: Exception | None = None):
            self.identities = identities or {}   # username -> LdapIdentity
            self.verify_error = verify_error
            self.calls: list[str] = []
            self.issuer = "ldap://fake:389"
        def verify(self, username, password):
            self.calls.append(username)
            if self.verify_error is not None:
                raise self.verify_error
            return self.identities.get(username)
    ```
  - `DomainAuthBroker` (in `composition.py`):
    ```python
    class DomainAuthBroker:
        def __init__(self, accounts: Any, identities: Any, ldap: Any | None, *, performers: Any | None = None) -> None
        def authenticate(self, username: str, password: str) -> "LocalSession | None": ...
    ```
    Routing (pinned):
    1. `account = accounts.get_by_username(username)` (a `ValueError` from username normalization → return `None`).
    2. **Not found** → `self.ldap is None` → `None`. `identity = self.ldap.verify(username, password)` (`LdapUnavailable`/`LdapAuthenticationError` propagate to the caller). `identity is None` → `None`. `accounts.create_ldap_account(username, identity.display_name)`; on `IntegrityError` (concurrent JIT) re-fetch via `get_by_username` and continue through the "found ldap account" path with `identity` in hand. `session = accounts.issue_session(username)`; `self._provision_ldap(account, identity)`; return `session`.
    3. **Found, `auth_provider == "local"`** → `accounts.authenticate(username, password)` (LDAP client is never touched).
    4. **Found, `auth_provider == "ldap"`** → `identity = self.ldap.verify(...)`; `ok = identity is not None`; `session = accounts.authenticate(username, password, verifier=lambda _stored, _pw: ok)`; if `session` and `identity`: `self._provision_ldap(account, identity)` (refresh path — idempotent).
  - `_provision_ldap(account, identity)`:
    - `principal_id = PrincipalId(str(uuid5(NAMESPACE_URL, f"kraken:ldap:{account.username.casefold()}")))`
    - `principal = Principal.ldap(principal_id=principal_id, issuer=self.ldap.issuer, subject=account.username, display_name=identity.display_name, email=identity.email, login=account.username, avatar_url=identity.avatar_url, system_roles=frozenset(SystemRole(r) for r in self.accounts.global_roles_for(account.account_id)))`
      (system_roles MUST be read from the store — `save()` overwrites the column; same pattern as `provision_local_account`.)
    - Claim check: `holder = self.identities.find_by_login(account.username.casefold())`. If `holder is not None` and `str(holder.id) != str(principal_id)`:
      - `holder.provider is PrincipalProvider.LOCAL` → `self.accounts.set_enabled(str(holder.id), False)` (revokes sessions + audits `account.disabled`) then `self.accounts.record_audit(action="login_conflict_local_disabled", target_account_id=str(holder.id), details={"login": account.username, "ldap_principal": str(principal_id)})`.
      - otherwise → raise `LdapUnavailable("conflicting LDAP principals for one login")` (503 upstream; never a silent rename).
    - `self.identities.save(principal)`; if `identity.avatar_url is not None`: `self.identities.sync_avatar(principal_id, identity.avatar_url)`; if `self.performers is not None`: `ensure_linked_performer(principal, self.performers)`.
  - `HybridSessionResolver.provision_local_account` (reverse direction): before building the principal, `holder = self.identities.find_by_login(account.username.casefold())`; if `holder is not None and str(holder.id) != str(account.account_id) and holder.provider is PrincipalProvider.LDAP` → build the principal with `login=None` (claim yielded) and `self.accounts.record_audit(action="login_claim_conflict_local_skipped", target_account_id=account.account_id, details={"login": account.username, "holder": str(holder.id)})`; otherwise `login=account.username`.

- [ ] **Step 1: Write `tests/ldap_fixtures.py`** (the `FakeLdap` above, plus a `make_broker(accounts, identities, fake, performers)` helper returning `DomainAuthBroker(accounts, identities, fake, performers=performers)`).

- [ ] **Step 2: Write the failing tests** in `tests/test_domain_auth_broker.py` (`database` fixture; `_uuid(n)` helper as in Task 5):

```python
def _ldap_identity(username, display="Ivan Petrov", email=None, avatar=None):
    from kraken_manager.infrastructure.auth.ldap import LdapIdentity
    return LdapIdentity(username, display, email, avatar)


def test_jit_first_domain_login_creates_account_principal_performer(database):
    services, accounts, _owner, _config = database
    fake = FakeLdap({"ivan": _ldap_identity("ivan", email="ivan@inme")})
    performers = PostgresPerformerStore(services.engine)
    broker = make_broker(accounts, services.identities, fake, performers)
    session = broker.authenticate("ivan", "domain-pw")
    assert session is not None and session.account.auth_provider == "ldap"
    account = accounts.get_by_username("ivan")
    principal = services.identities.get(PrincipalId(str(account.account_id)))
    assert principal is None  # ldap principal id is uuid5, NOT the account id
    principal = services.identities.find_by_login("ivan")
    assert principal.provider is PrincipalProvider.LDAP
    assert principal.display_name == "Ivan Petrov" and principal.email == "ivan@inme"
    assert performers.get_by_principal(principal.id) is not None


def test_jit_wrong_password_creates_nothing(database):
    services, accounts, _o, _c = database
    broker = make_broker(accounts, services.identities, FakeLdap({}), None)
    assert broker.authenticate("ghost", "pw") is None
    assert accounts.get_by_username("ghost") is None


def test_broker_without_ldap_returns_none_for_unknown(database):
    services, accounts, _o, _c = database
    broker = make_broker(accounts, services.identities, None, None)
    assert broker.authenticate("anyone", "pw") is None


def test_local_account_never_touches_ldap(database):
    services, accounts, _o, _c = database
    fake = FakeLdap()
    broker = make_broker(accounts, services.identities, fake, None)
    accounts.create_account("local.only", "LO", "pw123")
    assert broker.authenticate("local.only", "pw123") is not None
    assert fake.calls == []


def test_ldap_account_reverified_each_login_and_ad_down_propagates(database):
    services, accounts, _o, _c = database
    fake = FakeLdap({"re.user": _ldap_identity("re.user")})
    broker = make_broker(accounts, services.identities, fake, None)
    assert broker.authenticate("re.user", "ok") is not None
    fake.verify_error = LdapUnavailable("dc down")
    with pytest.raises(LdapUnavailable):
        broker.authenticate("re.user", "ok")


def test_login_conflict_disables_squatting_local_account(database):
    services, accounts, _o, _c = database
    # local account "squatter" already logged in once -> principal holds the login claim
    accounts.create_account("squatter", "Squatter", "pw")
    services.identities.save(Principal.local(subject="squatter", display_name="Squatter",
                                             principal_id=accounts.get_by_username("squatter").account_id,
                                             login="squatter"))
    broker = make_broker(accounts, services.identities, FakeLdap({"squatter": _ldap_identity("squatter")}), None)
    session = broker.authenticate("squatter", "domain-pw")
    assert session is not None
    assert accounts.get_by_username("squatter").enabled is False
    assert any(r["action"] == "login_conflict_local_disabled" for r in accounts.administration_audit(limit=20))
    assert services.identities.find_by_login("squatter").provider is PrincipalProvider.LDAP


def test_reverse_interleave_local_login_keeps_working(database):
    services, accounts, _o, _c = database
    from uuid import NAMESPACE_URL, uuid5
    ldap_pid = str(uuid5(NAMESPACE_URL, "kraken:ldap:late.user"))
    services.identities.save(Principal.ldap(issuer="ldap://fake:389", subject="late.user",
                                            display_name="Late User", principal_id=ldap_pid, login="late.user"))
    accounts.create_account("late.user", "Late Local", "pw")
    resolver = HybridSessionResolver(accounts, services.identities, None)
    session = accounts.authenticate("late.user", "pw")
    assert session is not None
    resolver.provision_local_account(session.account)
    local_principal = services.identities.get(PrincipalId(session.account.account_id))
    assert local_principal.login is None  # claim yielded to the LDAP principal
    assert any(r["action"] == "login_claim_conflict_local_skipped" for r in accounts.administration_audit(limit=20))


def test_disabled_ldap_account_cannot_login(database):
    services, accounts, _o, _c = database
    broker = make_broker(accounts, services.identities, FakeLdap({"off.user": _ldap_identity("off.user")}), None)
    broker.authenticate("off.user", "pw")
    accounts.set_enabled(accounts.get_by_username("off.user").account_id, False)
    assert broker.authenticate("off.user", "pw") is None


def test_second_jit_for_same_domain_user_is_idempotent(database):
    services, accounts, _o, _c = database
    broker = make_broker(accounts, services.identities, FakeLdap({"twice": _ldap_identity("twice")}), None)
    first = broker.authenticate("twice", "pw")
    second = broker.authenticate("twice", "pw")
    assert first is not None and second is not None
    assert accounts.get_by_username("twice").account_id == first.account.account_id
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_domain_auth_broker.py -v`
Expected: FAIL (`DomainAuthBroker` missing).

- [ ] **Step 4: Implement** `DomainAuthBroker` and the `provision_local_account` change per the Interfaces block.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_domain_auth_broker.py tests/test_registered_principal_before_session.py -v`
Expected: PASS (new tests + existing SRV-ROLE-BEFORE-SESSION regression).

- [ ] **Step 6: Commit**

```bash
git add src/kraken_server/composition.py tests/ldap_fixtures.py tests/test_domain_auth_broker.py
git commit -m "feat(server): domain auth broker with JIT provisioning and login claims"
```

---

### Task 7: HTTP wiring — broker in the login route, LAN gate, provider acceptance

**Files:**
- Create: `src/kraken_server/interface/http/ip_gate.py`
- Modify: `src/kraken_server/interface/http/context.py` (`HttpContext` attrs :36-71, `build_http_context` :74-86, `principal` :263-283, `shared_mutation_actor` :323-334, `system_roles` :336-350)
- Modify: `src/kraken_server/interface/http/factory.py:33-59`
- Modify: `src/kraken_server/interface/http/routes/auth.py:29-39` (`create_session`)
- Modify: `src/kraken_server/interface/http/routes/session.py:32`
- Test: `tests/test_ldap_http.py` (new)

**Interfaces:**
- Consumes: Tasks 4-6.
- Produces:
  - `ip_gate.py`: `def is_lan_client(ip: str, cidrs: Sequence[str]) -> bool` — `ipaddress.ip_address(ip)` in any `ipaddress.ip_network(cidr, strict=False)`; unparseable IP or CIDR → `False` (fail-open for registration; a warning log is acceptable). IPv4 and IPv6 both supported.
  - `build_http_context(..., domain_auth: Any | None = None, lan_gate: Callable[[str], bool] | None = None, client_ip: Callable[[Any], str] | None = None, identities: Any | None = None)` — new keyword params appended after `blob_gateway`; `client_ip` defaults to `lambda request: (request.client.host if request.client else "")`; all four exposed on `HttpContext` (new attrs default `None`).
  - `create_app(..., domain_auth=None, lan_gate=None, client_ip=None, identities=None)` — forwards to `build_http_context`.
  - `create_session` route: when `ctx.domain_auth` is not None → `session = domain_auth.authenticate(username, password)`; `LdapUnavailable` → `HTTPException(503, "Доменная авторизация временно недоступна")`; `LdapAuthenticationError` → `HTTPException(503, "Неверные учётные данные доменного аккаунта Kraken")`. Otherwise the existing `account_store.authenticate` path (unchanged — dev/tests without a broker).
  - `register_local_account` route: gains `request: Request` param (via `ctx.Request`); before creation: `if ctx.lan_gate is not None and ctx.lan_gate(ctx.client_ip(request)): raise HTTPException(403, "Создание локальных аккаунтов из внутренней сети запрещено. Используйте доменную учётную запись.")`.
  - Provider acceptance: `shared_mutation_actor` → `elif subject.provider not in {"local", "ldap"}: raise HTTPException(403, ...)`; `system_roles` → `if subject.provider in {"local", "ldap"} and account_store is not None and ...`; `session.py` fallback → `subject.provider if subject.provider in {"local", "gitlab", "ldap"} else "local"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_ldap_http.py`:

```python
# Unit part (no fixture):
def test_is_lan_client_matrix():
    from kraken_server.interface.http.ip_gate import is_lan_client
    assert is_lan_client("10.1.2.3", ["10.0.0.0/8"]) is True
    assert is_lan_client("8.8.8.8", ["10.0.0.0/8", "192.168.0.0/16"]) is False
    assert is_lan_client("192.168.0.1", ["192.168.0.0/16"]) is True
    assert is_lan_client("192.168.1.255", ["192.168.0.0/16"]) is True
    assert is_lan_client("10.0.0.0", ["10.0.0.0/8"]) is True        # network address, strict=False
    assert is_lan_client("2001:db8::1", ["2001:db8::/32"]) is True
    assert is_lan_client("10.1.2.3", []) is False
    assert is_lan_client("not-an-ip", ["10.0.0.0/8"]) is False


# Integration part (`database` fixture + TestClient + FakeLdap, pattern from
# tests/test_registered_principal_before_session.py):
def test_domain_login_end_to_end(database):
    ...
    fake = FakeLdap({"e2e": _ldap_identity("e2e", display="E2E User", email="e2e@inme")})
    broker = make_broker(accounts, services.identities, fake, PostgresPerformerStore(services.engine))
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=resolver, project_access_mode="acl",
                                   domain_auth=broker))
    resp = client.post("/api/v1/auth/sessions", json={"username": "e2e", "password": "pw"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    me = client.get("/api/v1/session", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["display_name"] == "E2E User"


def test_ad_unavailable_returns_503(database):
    # FakeLdap(verify_error=LdapUnavailable("dc down")) -> POST /auth/sessions -> 503


def test_local_login_unchanged_when_broker_present(database):
    # broker with empty FakeLdap; admin/password -> 200; fake.calls == []


def test_registration_blocked_from_lan(database):
    # create_app(..., lan_gate=lambda ip: True) -> POST /auth/accounts -> 403
    # create_app(..., lan_gate=lambda ip: False) -> same POST -> 201


def test_ldap_token_passes_mutation_gate(database):
    # JIT login, grant server_admin to the account (accounts.grant_global_role),
    # POST /api/v1/projects with the ldap token + Idempotency-Key -> 201


def test_existing_session_survives_ad_outage(database):
    # JIT login OK; fake.verify_error = LdapUnavailable(...);
    # GET /api/v1/session with the existing token -> 200
```

(Write all as full tests; `resolver` in the integration tests is `HybridSessionResolver(accounts, services.identities, None, performers=...)` as in the existing registration test.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ldap_http.py -v`
Expected: FAIL (no `ip_gate` module; `create_app` rejects `domain_auth`).

- [ ] **Step 3: Implement** per the Interfaces block (all files listed above).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ldap_http.py tests/test_registered_principal_before_session.py tests/test_auth_policy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kraken_server/interface/http tests/test_ldap_http.py
git commit -m "feat(server): domain login route, LAN registration gate, ldap provider acceptance"
```

---

### Task 8: Registration pre-check + admin CLI (shared `kraken_admin` application functions)

**Files:**
- Modify: `src/kraken_server/interface/http/routes/auth.py:53-82` (`register_local_account`)
- Modify: `src/kraken_admin/application/accounts.py:16-25` (`create_account`), `:93-96` (`reset_password`), `:126-137` (`account_rows`)
- Modify: `src/kraken_admin/presentation/cli/run_local_admin.py:46-54` (`account-create`), `:26-45` (`account-list` output)
- Test: `tests/test_ldap_registration.py` (new)

**Interfaces:**
- Consumes: Tasks 5 (`find_by_login`), 4 (`LocalAccount.auth_provider`), 7 (`ctx.identities`).
- Produces:
  - `register_local_account`: after username/password validation, before `store.create_account`:
    ```python
    identities = ctx.identities
    if identities is not None:
        holder = identities.find_by_login(username.strip().casefold())
        if holder is not None and holder.provider is not PrincipalProvider.LOCAL:
            raise HTTPException(409, f"Логин «{username}» уже занят")
    ```
  - `create_account(store, username, display_name, password, *, identities: object | None = None) -> str` — same pre-check; raises `ValueError(f"Логин «{username}» уже занят доменной учётной записью")`. (Existing `IntegrityError → "Логин уже занят"` behavior for local-local collisions unchanged.)
  - `reset_password`: after `account_by_username`, `if getattr(account, "auth_provider", "local") != "local": raise ValueError("Пароль управляется доменом — сброс пароля недоступен для доменных учётных записей")`.
  - `account_rows`: each row gains `"auth_provider": getattr(account, "auth_provider", "local")`.
  - `run_local_admin.py`: `account-create` builds `PostgresIdentityAclStore(store.engine)` and passes it as `identities=`; `account-list` JSON/text output gains the provider (text: append a `provider` column, default rendering `local` / `ldap`).

- [ ] **Step 1: Write the failing tests** in `tests/test_ldap_registration.py`:

```python
def test_http_registration_rejects_ldap_held_login(database):
    # save Principal.ldap(issuer=..., subject="taken", display_name="T", principal_id=uuid, login="taken")
    # via services.identities.save; TestClient over create_app(services=services,
    # account_store=accounts, identities=services.identities, ...) — identities MUST be
    # passed or the pre-check is disabled;
    # POST /api/v1/auth/accounts {"username": "taken", "password": "pw"} -> 409

def test_http_registration_still_allows_new_login(database):
    # same setup; POST username "free.user" -> 201

def test_cli_create_account_rejects_ldap_held_login(database):
    from kraken_admin.application.accounts import create_account
    services, accounts, _o, _c = database
    # seed ldap principal as above
    with pytest.raises(ValueError, match="уже занят"):
        create_account(accounts, "taken2", "T2", "pw", identities=services.identities)

def test_cli_create_account_without_identities_still_works(database):
    services, accounts, _o, _c = database
    create_account(accounts, "plain.local", "PL", "pw")  # no identities kwarg -> no pre-check

def test_reset_password_refuses_ldap_account(database):
    from kraken_admin.application.accounts import reset_password
    # create an ldap account via accounts.create_ldap_account("rp.user", "RP")
    with pytest.raises(ValueError, match="доменом"):
        reset_password(accounts, "rp.user", "newpw")

def test_account_rows_include_provider(database):
    from kraken_admin.application.accounts import account_rows
    accounts.create_ldap_account("prov.user", "P")
    rows = {r["username"]: r["auth_provider"] for r in account_rows(accounts)}
    assert rows["prov.user"] == "ldap"
    assert rows["admin"] == "local"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ldap_registration.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** per the Interfaces block.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ldap_registration.py tests/test_ldap_http.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kraken_server/interface/http/routes/auth.py src/kraken_admin tests/test_ldap_registration.py
git commit -m "feat(accounts): reject registrations whose login is held by a domain principal"
```

---

### Task 9: Profile API + payloads

**Files:**
- Modify: `src/kraken_server/interface/http/routes/session.py` (new PATCH route)
- Modify: `src/kraken_server/persistent_services_parts/create_project.py:90-120` (`list_principals` payload)
- Test: `tests/test_profile_api.py` (new)

**Interfaces:**
- Consumes: Tasks 5 (`set_status`, `set_manual_avatar`, `find_by_login`), 7 (`ctx.identities`).
- Produces:
  - `PATCH /api/v1/principals/me/profile` — body `{"status_text"?: str, "avatar_url"?: str}`; self-only by construction (target is always the session's own `principal_id`); `ctx.identities is None` → `HTTPException(503, "Identity catalog is not configured")`; validation: `status_text` when present must be a str with `len <= 255`; `avatar_url` when present must be a non-empty str with `len <= 2048` (else `ValidationError`/422 via `ctx`); returns `{"login", "display_name", "avatar_url", "status_text"}` of the updated principal.
  - `list_principals` principal dict gains `"login"`, `"avatar_url"`, `"status_text"` (from the `Principal` fields); the account-fallback dict (accounts without a principal row) gains `"login": account.username, "avatar_url": None, "status_text": None`. `GET /api/v1/session` picks these up automatically via the actor dict.

- [ ] **Step 1: Write the failing tests** in `tests/test_profile_api.py` (`database` fixture + TestClient; the app is built with `create_app(services=services, account_store=accounts, identities=services.identities, project_access_mode="acl")` — `identities` is required for the PATCH route; admin token via `accounts.authenticate("admin", "password")`):

```python
def test_patch_own_profile_is_visible_in_session_and_principals(database):
    # PATCH /api/v1/principals/me/profile {"status_text": "на встрече", "avatar_url": "https://me/1.png"} -> 200
    # GET /api/v1/session -> status_text == "на встрече", avatar_url == "https://me/1.png"
    # GET /api/v1/principals -> the admin item carries login "admin", the same status/avatar

def test_patch_validates_lengths(database):
    # status_text of 256 chars -> 422; avatar_url of 2049 chars -> 422; empty avatar_url -> 422

def test_manual_avatar_survives_ldap_sync(database):
    # admin PATCH avatar (manual); then identities.sync_avatar(admin_pid, "https://ad/x.png")
    # -> identities.get(admin_pid).avatar_url is still the manual one

def test_new_local_registration_payload_has_profile_fields(database):
    # POST /auth/accounts (new user); GET /api/v1/principals -> item has login == username,
    # avatar_url is None, status_text is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_profile_api.py -v`
Expected: FAIL (404 on the PATCH route).

- [ ] **Step 3: Implement** per the Interfaces block.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_profile_api.py tests/test_ldap_http.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kraken_server/interface/http/routes/session.py src/kraken_server/persistent_services_parts/create_project.py tests/test_profile_api.py
git commit -m "feat(server): own-profile PATCH endpoint and profile fields in payloads"
```

---

### Task 10: Desktop — remove implicit auto-registration, explicit "create on server"

**Files:**
- Modify: `src/kraken_hub/composition/catalog.py:306-343` (`configure_remote_local`)
- Modify: `src/kraken_hub/presentation/qt/preferences_dialog.py:285-315` (account page), `:337-343` (`_add_account` neighbor)
- Modify: `src/kraken_hub/presentation/qt/parts/connect_saved_server.py:14-31` (`_server_project_hint`)
- Test: `tests/test_no_auto_registration.py` (new)

**Interfaces:**
- Consumes: nothing server-side (the registration endpoint already exists).
- Produces:
  - `configure_remote_local`: on `RemoteServerError` with `status == 401` → `raise ValueError("Учётная запись не найдена или неверный пароль") from exc` (the `register_local` fallback is DELETED); non-401 re-raises as today.
  - `ProgramPreferencesDialog` account page: new button **"Создать на сервере…"** (next to "Добавить…"); handler `_create_account_on_server`: `_AccountEditorDialog.edit(self)` → if the chosen provider is not `"local"` → warn "Создаётся только локальная учётная запись" and return; `url = self._validated_url()`; empty → warn "Сначала укажите адрес сервера"; then `RemoteServerProjectService.register_local(url, username=account.username, password=account.secret, display_name=account.display_name)`; `RemoteServerError` → `QMessageBox.warning` with the server message (this surfaces the 403 LAN gate and the 409 login-taken errors); success → append the account, `_single_default`, `_refresh_accounts()`, info box "Учётная запись создана на сервере".
  - Account-page hint (preferences_dialog.py:307-311) replaced: "Доменные пользователи вводят доменный логин и пароль — аккаунт создаётся на сервере при первом входе. Локальные учётные записи (пользователи вне локальной сети) создаются кнопкой «Создать на сервере…»."
  - `_server_project_hint` (connect_saved_server.py): replace "Если такой записи ещё нет на сервере, она будет создана." with "Локальная запись создаётся на сервере явно (Настройки → Учетная запись → «Создать на сервере…»). Доменные пользователи входят доменным логином."

- [ ] **Step 1: Write the failing test** in `tests/test_no_auto_registration.py`:

```python
def test_failed_remote_login_does_not_register():
    # Stub http object with .request(method, path, token, payload=None) that
    # raises RemoteServerError("Invalid credentials", 401) for
    # POST /api/v1/auth/sessions and records every call.
    # Drive the fallback logic the same way catalog.configure_remote_local does:
    # build a minimal object exposing the needed surface OR, if existing tests
    # already construct DualCatalogService (search tests/ for DualCatalogService),
    # follow that pattern.
    # Assert: configure_remote_local raises ValueError ("не найдена или неверный")
    # and the stub's recorded paths do NOT include "/api/v1/auth/accounts".
```

(If constructing `DualCatalogService` proves heavy, extract the fallback removal so the test targets `RemoteServerProjectService.authenticate_local` + the new explicit error contract: the test's essential pin is "401 → no POST /auth/accounts call". Follow whatever pattern existing remote tests in `tests/` use for stubbing `RemoteHttpClient`.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_no_auto_registration.py -v`
Expected: FAIL (the 401 still triggers registration today).

- [ ] **Step 3: Implement** per the Interfaces block (delete the fallback, add the button + handler, update both hint texts).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_no_auto_registration.py -v && uv run ruff check src/kraken_hub`
Expected: PASS + clean.

- [ ] **Step 5: Commit**

```bash
git add src/kraken_hub tests/test_no_auto_registration.py
git commit -m "feat(hub): explicit server-side account creation, no implicit auto-registration"
```

---

### Task 11: Server configuration — `[identity]` LDAP keys, secret file, env propagation, composition wiring

**Files:**
- Modify: `src/kraken_server/configuration.py:132-157` (fields), `:160-246` (`load`), `:256-301` (`apply_to_environment`)
- Modify: `src/kraken_server/composition.py:162-253` (`postgresql_composition`)
- Modify: `config/templates/server.production.template.toml`, `config/templates/server.local.template.toml`
- Test: `tests/test_ldap_config.py` (new)

**Interfaces:**
- Consumes: Tasks 2 (`LdapClient`), 6 (`DomainAuthBroker`), 7 (`is_lan_client`, `create_app` kwargs).
- Produces:
  - `ServerConfig` new fields (all defaulted): `ldap_url: str | None`, `ldap_base_dn: str | None`, `ldap_user_filter: str | None`, `ldap_bind_dn: str | None`, `ldap_bind_secret: Path | None`, `ldap_start_tls: bool = False`, `ldap_allow_insecure: bool = False`, `ldap_ca_file: Path | None`, `ldap_timeout: float = 5.0`, `lan_cidrs: tuple[str, ...] = ()`.
  - `load()` — parse from the `[identity]` table; validation: `ldap_url` set but `ldap_base_dn`/`ldap_bind_dn` missing → `ValueError("LDAP identity configuration is incomplete")`; plain `ldap://` without `ldap_start_tls`/`ldap_allow_insecure` → `ValueError`; every `lan_cidrs` entry must parse via `ipaddress.ip_network(entry, strict=False)` → `ValueError(f"Invalid lan_cidrs entry: {entry}")`. `ldap_bind_secret` resolves like other paths (relative to the config dir).
  - `apply_to_environment()` — when `ldap_url` set: `KRAKEN_LDAP_URL`, `KRAKEN_LDAP_BASE_DN`, `KRAKEN_LDAP_USER_FILTER` (only if set), `KRAKEN_LDAP_BIND_DN`, `KRAKEN_LDAP_BIND_SECRET_FILE` (the secret FILE PATH — the password itself never enters the environment), `KRAKEN_LDAP_START_TLS`/`KRAKEN_LDAP_ALLOW_INSECURE` (`"1"`/`"0"`), `KRAKEN_LDAP_CA_FILE` (only if set), `KRAKEN_LDAP_TIMEOUT`; always: `KRAKEN_LAN_CIDRS` = JSON list (possibly `[]`). When `ldap_url` unset: pop all `KRAKEN_LDAP_*` (blob-gateway pattern, configuration.py:291-301).
  - `postgresql_composition()` — after `accounts`/`identities`/`performers` are built:
    ```python
    ldap_url = os.environ.get("KRAKEN_LDAP_URL", "").strip()
    domain_auth = None
    if ldap_url:
        from kraken_manager.infrastructure.auth.ldap import LdapClient
        from kraken_server.configuration import unprotect_secret
        bind_secret_file = Path(os.environ.get("KRAKEN_LDAP_BIND_SECRET_FILE", ""))
        if not bind_secret_file.is_file():
            raise RuntimeError("KRAKEN_LDAP_BIND_SECRET_FILE is configured but the secret file is missing")
        ldap_client = LdapClient(
            ldap_url,
            base_dn=os.environ["KRAKEN_LDAP_BASE_DN"],
            bind_dn=os.environ["KRAKEN_LDAP_BIND_DN"],
            bind_password=unprotect_secret(bind_secret_file),
            user_filter=os.environ.get("KRAKEN_LDAP_USER_FILTER", "") or None,  # None -> LdapClient default
            start_tls=os.environ.get("KRAKEN_LDAP_START_TLS") == "1",
            allow_insecure=os.environ.get("KRAKEN_LDAP_ALLOW_INSECURE") == "1",
            ca_file=os.environ.get("KRAKEN_LDAP_CA_FILE") or None,
            timeout=float(os.environ.get("KRAKEN_LDAP_TIMEOUT", "5")),
        )
        domain_auth = DomainAuthBroker(accounts, identities, ldap_client, performers=performers)
    import json as _json
    lan_gate = None
    cidrs = _json.loads(os.environ.get("KRAKEN_LAN_CIDRS", "[]"))
    if cidrs:
        from kraken_server.interface.http.ip_gate import is_lan_client
        lan_gate = lambda ip: is_lan_client(ip, cidrs)
    ```
    and the returned dict gains `"domain_auth": domain_auth, "lan_gate": lan_gate, "identities": identities`.
  - Templates: commented `[identity]` block with the real values as examples (`ldap_url = "ldap://DC.ad.inme:389"`, `ldap_base_dn = "DC=ad,DC=inme"`, `ldap_bind_dn = "CN=kraken,CN=Users,DC=ad,DC=inme"`, `ldap_start_tls = true`, `lan_cidrs = [...]`) and the note that the bind password lives in a DPAPI secret file, never in TOML.

- [ ] **Step 1: Write the failing tests** in `tests/test_ldap_config.py` (pure unit, `tmp_path` TOML files; `ServerConfig.load` requires the mandatory sections — copy the minimal valid TOML shape used by existing configuration tests if any exist, else build a minimal file with `[server]`, `[database]`, `[storage]`):

```python
def _toml(tmp_path, identity_extra: str) -> Path:  # minimal valid server.toml + identity block

def test_load_parses_ldap_section(tmp_path):
    # -> ldap_url/base_dn/bind_dn/start_tls/allow_insecure/ca_file/timeout/lan_cidrs parsed

def test_load_rejects_incomplete_ldap(tmp_path):
    # ldap_url set, ldap_base_dn missing -> ValueError("incomplete")

def test_load_rejects_plain_without_starttls(tmp_path):
    # ldap://dc:389, no start_tls/allow_insecure -> ValueError

def test_load_rejects_bad_cidr(tmp_path):
    # lan_cidrs = ["10.0.0.0/33"] -> ValueError

def test_apply_to_environment_sets_and_pops(monkeypatch):
    # with ldap configured: KRAKEN_LDAP_* + KRAKEN_LAN_CIDRS present,
    #   KRAKEN_LDAP_BIND_SECRET_FILE == str(path) and the password value is NOT in any env var
    # with ldap unset: all KRAKEN_LDAP_* popped
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ldap_config.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** per the Interfaces block.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ldap_config.py tests/test_ldap_http.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kraken_server/configuration.py src/kraken_server/composition.py config/templates tests/test_ldap_config.py
git commit -m "feat(server): ldap identity configuration, secret file, composition wiring"
```

---

### Task 12: Full validation pass

**Files:**
- None new (verification only; fix whatever surfaces).

- [ ] **Step 1: Run the auth-adjacent and new test suites**

Run: `uv run pytest tests/test_local_auth.py tests/test_auth_policy.py tests/test_gitlab_oidc.py tests/test_identity_ldap_principal.py tests/test_ldap_client.py tests/test_ldap_schema.py tests/test_ldap_account_store.py tests/test_ldap_identity_store.py tests/test_domain_auth_broker.py tests/test_ldap_http.py tests/test_ldap_registration.py tests/test_profile_api.py tests/test_no_auto_registration.py tests/test_ldap_config.py tests/test_registered_principal_before_session.py -v`
Expected: all PASS (PG tests skip only if `KRAKEN_TEST_POSTGRES_URL` is unset — in that case set it and re-run; a skip is not a pass).

- [ ] **Step 2: Lint the changed files**

Run: `uv run ruff check src/kraken_manager src/kraken_server src/kraken_admin src/kraken_hub migrations tests`
Expected: no new findings on touched files (pre-existing findings elsewhere are out of scope — do not fix them).

- [ ] **Step 3: Diff review**

Run: `git diff --stat` against the branch start; verify: no unrelated files touched; `gitlab.py`, the GitLab provider paths, and the desktop SQLite mirrors (`accounts.sqlite3`/`identity.sqlite3` schema) are unchanged; the bind password appears in no TOML, env assignment, or log call.

- [ ] **Step 4: Final commit (only if fixes were needed in Steps 1-3)**

```bash
git add -A
git commit -m "chore: validation fixes for ldap sign-in plan"
```
