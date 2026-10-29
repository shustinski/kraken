"""Current authenticated session."""

from __future__ import annotations

from typing import Any

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    Depends = ctx.Depends
    principal = ctx.principal
    system_roles = ctx.system_roles

    @app.get(f"{API_PREFIX}/session")
    def current_session(subject: SessionPrincipal = Depends(principal)) -> dict[str, Any]:
        actor = next(
            (
                item
                for item in backend.list_principals(include_inactive=True)
                if str(item.get("principal_id")) == subject.principal_id
            ),
            None,
        )
        if actor is None:
            return {
                "principal_id": subject.principal_id,
                "provider": (subject.provider if subject.provider in {"local", "gitlab"} else "local"),
                "subject": subject.principal_id,
                "issuer": None,
                "display_name": subject.principal_id,
                "email": None,
                "active": True,
                "system_roles": sorted(system_roles(subject)),
            }
        payload = dict(actor)
        payload["system_roles"] = sorted(system_roles(subject))
        return payload
