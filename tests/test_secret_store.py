from __future__ import annotations

import sys
from types import SimpleNamespace

from kraken_hub.secret_store import KeyringSecretStore


def test_keyring_secret_store_round_trip_and_missing_delete(monkeypatch) -> None:
    values: dict[tuple[str, str], str] = {}

    class PasswordDeleteError(Exception):
        pass

    def get_password(service: str, key: str) -> str | None:
        return values.get((service, key))

    def set_password(service: str, key: str, value: str) -> None:
        values[(service, key)] = value

    def delete_password(service: str, key: str) -> None:
        try:
            del values[(service, key)]
        except KeyError as exc:
            raise PasswordDeleteError from exc

    fake_keyring = SimpleNamespace(
        get_password=get_password,
        set_password=set_password,
        delete_password=delete_password,
        errors=SimpleNamespace(PasswordDeleteError=PasswordDeleteError),
    )
    monkeypatch.setitem(sys.modules, "keyring", fake_keyring)
    store = KeyringSecretStore("Kraken.Tests")

    assert store.get("review") is None
    store.set("review", b"\x00private-key\xff")
    assert store.get("review") == b"\x00private-key\xff"
    assert values[("Kraken.Tests", "review")] != "\x00private-key\xff"
    store.delete("review")
    assert store.get("review") is None
    store.delete("review")
