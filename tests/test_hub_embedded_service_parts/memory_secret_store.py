"""test_hub_embedded_service: memory secret store."""
from __future__ import annotations


class MemorySecretStoreMixin:
    @staticmethod
    def _memory_secret_store():
        class MemorySecretStore:
            def __init__(self) -> None:
                self.values: dict[str, bytes] = {}

            def get(self, key: str) -> bytes | None:
                return self.values.get(key)

            def set(self, key: str, value: bytes) -> None:
                self.values[key] = value

        return MemorySecretStore()
