"""services: command context."""
from __future__ import annotations
from dataclasses import dataclass

class ConflictError(RuntimeError):
    pass

class NotFoundError(KeyError):
    pass

class ValidationError(ValueError):
    pass

class ForbiddenError(PermissionError):
    pass

@dataclass(frozen=True, slots=True)
class CommandContext:
    actor_id: str
    idempotency_key: str
    expected_revision: int | None
