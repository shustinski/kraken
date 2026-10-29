"""Authenticated caller of an HTTP or websocket request."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SessionPrincipal:
    principal_id: str
    provider: str
    access_token: str
