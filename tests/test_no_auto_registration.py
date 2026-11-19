"""A failed (401) server login must not implicitly register a local account."""

from __future__ import annotations

import pytest

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.infrastructure.remote import client as remote_client
from kraken_hub.infrastructure.remote.client import RemoteHttpClient, RemoteServerError


class _RecordingHttp(RemoteHttpClient):
    """Transport stub: rejects every login with 401 and records request paths."""

    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.calls: list[tuple[str, str]] = []

    def request(self, method, path, *, token, payload=None, headers=None, timeout=None):
        self.calls.append((method.upper(), path))
        if path == "/api/v1/auth/sessions":
            raise RemoteServerError("Invalid credentials", status=401)
        raise RemoteServerError(f"unexpected {method} {path}", status=500)


def test_failed_remote_login_does_not_register(tmp_path, monkeypatch) -> None:
    http = _RecordingHttp("http://127.0.0.1:8080")
    # authenticate_local/register_local build their own transport via
    # register_local._remote_http_client, which reads this module attribute.
    monkeypatch.setattr(remote_client, "RemoteHttpClient", lambda base_url: http)

    catalog = DualCatalogService(EmbeddedProjectService(tmp_path / "local"))

    with pytest.raises(ValueError, match="не найдена или неверный") as excinfo:
        catalog.configure_remote_local(
            base_url="http://127.0.0.1:8080",
            username="alice",
            password="wrong",
        )

    assert isinstance(excinfo.value.__cause__, RemoteServerError)
    assert excinfo.value.__cause__.status == 401
    assert http.calls == [("POST", "/api/v1/auth/sessions")]
    assert "/api/v1/auth/accounts" not in [path for _method, path in http.calls]
