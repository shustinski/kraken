"""An active remote login remains usable without a saved default account."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kraken_hub.presentation.qt.parts.connect_program_account import ConnectProgramAccountMixin
from kraken_hub.presentation.qt import preferences_dialog


def _controller(url: str) -> ConnectProgramAccountMixin:
    controller = ConnectProgramAccountMixin()
    controller.service = SimpleNamespace(
        supports_server_connection=True,
        server_connection=SimpleNamespace(base_url=url),
    )
    return controller


def test_current_server_session_needs_no_saved_default(monkeypatch) -> None:
    monkeypatch.setattr(preferences_dialog, "default_program_account", lambda _settings: None)
    monkeypatch.setattr(preferences_dialog, "program_server_url", lambda _settings: "")
    _controller("http://127.0.0.1:8080")._connect_program_account()


def test_different_saved_server_still_requires_an_account(monkeypatch) -> None:
    monkeypatch.setattr(preferences_dialog, "default_program_account", lambda _settings: None)
    monkeypatch.setattr(preferences_dialog, "program_server_url", lambda _settings: "http://127.0.0.1:8081")
    with pytest.raises(ValueError, match="учётную запись"):
        _controller("http://127.0.0.1:8080")._connect_program_account()
