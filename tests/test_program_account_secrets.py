from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QSettings

from kraken_hub.preferences_dialog import (
    ACCOUNTS_KEY,
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


def test_program_account_secrets_stay_out_of_qsettings(tmp_path) -> None:
    settings = _settings(tmp_path / "hub.ini")
    secrets = _MemorySecrets()
    save_program_accounts(
        [
            ProgramAccount("acc-1", "local", "alice", "Alice", "s3cret", True),
            ProgramAccount("acc-2", "gitlab", "bob", "Bob", "glpat-1", False),
        ],
        settings,
        secrets,  # type: ignore[arg-type]
    )
    settings.sync()
    stored = json.loads(str(settings.value(ACCOUNTS_KEY)))
    assert all("secret" not in item for item in stored)
    raw = (tmp_path / "hub.ini").read_text(encoding="utf-8")
    assert "s3cret" not in raw
    assert "glpat-1" not in raw
    loaded = load_program_accounts(settings, secrets)  # type: ignore[arg-type]
    assert [account.secret for account in loaded] == ["s3cret", "glpat-1"]

    save_program_accounts(loaded[:1], settings, secrets)  # type: ignore[arg-type]
    assert "acc-2" not in secrets.values
    assert secrets.get("acc-1") == b"s3cret"


def test_legacy_plaintext_secret_is_migrated_out_of_qsettings(tmp_path) -> None:
    settings = _settings(tmp_path / "legacy.ini")
    settings.setValue(
        ACCOUNTS_KEY,
        json.dumps(
            [
                {
                    "account_id": "legacy",
                    "provider": "local",
                    "username": "admin",
                    "display_name": "Admin",
                    "secret": "plain-password",
                    "is_default": True,
                }
            ]
        ),
    )
    secrets = _MemorySecrets()
    loaded = load_program_accounts(settings, secrets)  # type: ignore[arg-type]
    settings.sync()
    assert loaded[0].secret == "plain-password"
    assert secrets.get("legacy") == b"plain-password"
    raw = (tmp_path / "legacy.ini").read_text(encoding="utf-8")
    assert "plain-password" not in raw
