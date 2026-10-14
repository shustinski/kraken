"""remote_client: principal server payload."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from kraken_manager.domain.identity import (
    Principal,
    PrincipalProvider,
    SystemRole,
)

def _principal_from_server_payload(payload: Mapping[str, Any]) -> Principal:
    """Map a Kraken Server principal/session document into the domain Principal."""
    provider_raw = str(payload.get("provider", "local") or "local")
    issuer = None if payload.get("issuer") is None else str(payload["issuer"])
    if provider_raw == PrincipalProvider.GITLAB.value and not issuer:
        # Development memory backends historically emitted gitlab without an OIDC issuer.
        provider_raw = PrincipalProvider.LOCAL.value
    return Principal(
        id=str(payload.get("principal_id") or payload.get("id") or ""),
        provider=PrincipalProvider(provider_raw),
        subject=str(payload.get("subject", "")),
        issuer=issuer,
        display_name=str(payload.get("display_name", "")),
        email=None if payload.get("email") is None else str(payload["email"]),
        active=bool(payload.get("active", True)),
        system_roles=frozenset(
            SystemRole(str(role)) for role in payload.get("system_roles", ())
        ),
    )
