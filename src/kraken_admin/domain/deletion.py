"""Proof that an administrator confirmed removal of a stored file."""

from __future__ import annotations

# The rest of local deletion reports refusal as this same conflict.
from kraken_server.services import ConflictError


class DeletionAuthorization:
    """Proof that this deletion attempt was confirmed or auto-confirmed."""

    __slots__ = ("_granted",)

    def __init__(self, granted: bool) -> None:
        self._granted = granted

    @classmethod
    def grant(
        cls,
        *,
        name_matches: bool,
        auto_confirm: bool,
        policy_enabled: bool,
    ) -> DeletionAuthorization:
        if auto_confirm and not policy_enabled:
            raise ConflictError("Автоподтверждение удалений выключено")
        if name_matches or auto_confirm:
            return cls(True)
        raise ConflictError("Удаление файлов требует подтверждения администратора")

    def require(self) -> None:
        if not self._granted:
            raise ConflictError("Удаление файлов требует подтверждения администратора")
