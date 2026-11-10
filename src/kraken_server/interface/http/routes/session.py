"""Current authenticated session and the principal's own profile."""

from __future__ import annotations

from typing import Any

from kraken_manager.domain.common import PrincipalId, validate_uuid
from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import ValidationError

MAX_STATUS_TEXT_LENGTH = 255
MAX_AVATAR_URL_LENGTH = 2048


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    Depends = ctx.Depends
    principal = ctx.principal
    system_roles = ctx.system_roles
    HTTPException = ctx.HTTPException
    identities = ctx.identities

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
                "provider": (subject.provider if subject.provider in {"local", "gitlab", "ldap"} else "local"),
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

    @app.patch(f"{API_PREFIX}/principals/me/profile")
    def update_own_profile(payload: dict[str, Any], subject: SessionPrincipal = Depends(principal)) -> dict[str, Any]:
        """Update the caller's own status text and manual avatar.

        Self-only by construction: the target is always the session's own
        principal id. The identity store is the only writer; AD is never
        consulted on an authenticated path.
        """
        if identities is None:
            raise HTTPException(status_code=503, detail="Identity catalog is not configured")
        status_text = payload.get("status_text")
        if status_text is not None and (
            not isinstance(status_text, str) or len(status_text) > MAX_STATUS_TEXT_LENGTH
        ):
            raise ValidationError(
                f"status_text must be a string of at most {MAX_STATUS_TEXT_LENGTH} characters"
            )
        avatar_url = payload.get("avatar_url")
        if avatar_url is not None and (
            not isinstance(avatar_url, str) or not avatar_url or len(avatar_url) > MAX_AVATAR_URL_LENGTH
        ):
            raise ValidationError(
                f"avatar_url must be a non-empty string of at most {MAX_AVATAR_URL_LENGTH} characters"
            )
        try:
            principal_id = PrincipalId(validate_uuid(subject.principal_id, field="principal_id"))
        except (TypeError, ValueError):
            principal_id = None
        if principal_id is None or identities.get(principal_id) is None:
            # The session resolves but its identity row does not exist: a data
            # anomaly, reported cleanly instead of failing with a 500.
            raise HTTPException(status_code=404, detail="Profile principal row is missing")
        if status_text is not None:
            identities.set_status(principal_id, status_text)
        if avatar_url is not None:
            identities.set_manual_avatar(principal_id, avatar_url)
        updated = identities.get(principal_id)
        if updated is None:
            raise HTTPException(status_code=404, detail="Profile principal row is missing")
        return {
            "login": updated.login,
            "display_name": updated.display_name,
            "avatar_url": updated.avatar_url,
            "status_text": updated.status_text,
        }
