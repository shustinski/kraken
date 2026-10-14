"""app: session principal."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SessionPrincipal:
    principal_id: str
    provider: str
    access_token: str
