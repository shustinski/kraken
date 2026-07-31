"""OS-backed secrets used by the Desktop composition root."""

from __future__ import annotations

import base64


class KeyringSecretStore:
    """Small bytes adapter over the installed system keyring backend."""

    def __init__(self, service_name: str = "Kraken.ProjectManager") -> None:
        self.service_name = service_name

    @staticmethod
    def _keyring():
        import keyring

        return keyring

    def get(self, key: str) -> bytes | None:
        encoded = self._keyring().get_password(self.service_name, str(key))
        if encoded is None:
            return None
        return base64.b64decode(encoded.encode("ascii"), validate=True)

    def set(self, key: str, value: bytes) -> None:
        if not isinstance(value, bytes):
            raise TypeError("Secret value must be bytes")
        encoded = base64.b64encode(value).decode("ascii")
        self._keyring().set_password(self.service_name, str(key), encoded)

    def delete(self, key: str) -> None:
        keyring = self._keyring()
        try:
            keyring.delete_password(self.service_name, str(key))
        except keyring.errors.PasswordDeleteError:
            return


__all__ = ["KeyringSecretStore"]
