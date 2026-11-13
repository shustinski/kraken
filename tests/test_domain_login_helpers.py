from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QSettings

from kraken_hub.presentation.qt.parts.domain_login import (
    domain_login_error_text,
    domain_server_status,
    upsert_program_account,
)
from kraken_hub.presentation.qt.preferences_dialog import (
    ProgramAccount,
    load_program_accounts,
    save_program_accounts,
)


class _MemorySecrets:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def set(self, key: str, value: bytes) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


def _settings(path) -> QSettings:
    return QSettings(str(path), QSettings.Format.IniFormat)


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


def test_error_text_maps_status_and_network():
    from kraken_hub.application.remote_error import RemoteServerError
    assert domain_login_error_text(RemoteServerError("Доменная авторизация временно недоступна", status=503)) == "Доменная авторизация временно недоступна"
    assert domain_login_error_text(RemoteServerError("x", status=None)) == "Сервер недоступен"
