"""Health, local sessions and ordinary account registration."""

from __future__ import annotations

from typing import Any

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.services import ValidationError


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    account_store = ctx.account_store
    blob_gateway = ctx.blob_gateway
    HTTPException = ctx.HTTPException
    problem = ctx.problem

    @app.get(f"{API_PREFIX}/health")
    def health() -> Any:
        status = backend.health()
        if status.get("status") != "ok":
            return problem(503, "service.degraded", "Service degraded", str(status.get("detail", "")))
        status["blob_gateway"] = "enabled" if blob_gateway is not None else "proxy"
        return status

    @app.post(f"{API_PREFIX}/auth/sessions")
    def create_session(payload: dict[str, Any]) -> dict[str, Any]:
        if account_store is None:
            return {"provider": "development", "access_token": str(payload.get("username", "developer"))}
        try:
            session = account_store.authenticate(str(payload.get("username", "")), str(payload.get("password", "")))
        except ValueError:
            session = None
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return local_session_payload(session)

    def local_session_payload(session: Any) -> dict[str, Any]:
        return {
            "provider": "local",
            "access_token": session.token,
            "expires_at": session.expires_at,
            "principal": {
                "id": session.account.account_id,
                "username": session.account.username,
                "display_name": session.account.display_name,
            },
        }

    @app.post(f"{API_PREFIX}/auth/accounts", status_code=201)
    def register_local_account(payload: dict[str, Any]) -> dict[str, Any]:
        """Create an ordinary account, or sign in when the same password is already registered.

        The account receives no server administrator role. It may create its own
        projects and sees only projects its Kraken roles allow.
        """
        store = require_account_store()
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        display_name = str(payload.get("display_name", "")).strip() or username
        if not username or not password:
            raise ValidationError("Username and password are required")
        try:
            store.create_account(username, display_name, password)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        except Exception as exc:
            if exc.__class__.__name__ != "IntegrityError":
                raise
            session = store.authenticate(username, password)
            if session is None:
                raise HTTPException(status_code=401, detail="Invalid credentials") from exc
            return local_session_payload(session)
        session = store.authenticate(username, password)
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return local_session_payload(session)

    def require_account_store() -> Any:
        if account_store is None or not hasattr(account_store, "list_accounts"):
            raise HTTPException(status_code=503, detail="Server account administration is not configured")
        return account_store
