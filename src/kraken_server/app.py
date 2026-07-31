"""FastAPI application factory; all adapter selection happens here."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from .services import (
    CommandContext,
    ConflictError,
    ForbiddenError,
    InMemoryServerServices,
    NotFoundError,
    ServerServices,
    ValidationError,
)


API_PREFIX = "/api/v1"


@dataclass(frozen=True, slots=True)
class SessionPrincipal:
    principal_id: str
    provider: str
    access_token: str


def create_app(
    *,
    services: ServerServices | None = None,
    account_store: Any | None = None,
    session_resolver: Callable[[str], SessionPrincipal | None] | None = None,
    live_gitlab_verifier: Callable[[SessionPrincipal], bool] | None = None,
    development: bool = False,
    connection_hub: Any | None = None,
    outbox_publisher: Any | None = None,
) -> Any:
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install Kraken with the 'server' extra to run Kraken Server") from exc

    if not development and services is None:
        raise RuntimeError("Production Kraken Server requires an injected persistent ServerServices adapter")
    if not development and account_store is None and session_resolver is None:
        raise RuntimeError("Production Kraken Server requires an authenticated session resolver")
    backend = services or InMemoryServerServices()
    from .outbox import ConnectionHub

    hub = connection_hub or ConnectionHub()
    app = FastAPI(title="Kraken Server", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.connection_hub = hub
    app.state.outbox_publisher = outbox_publisher

    @app.on_event("startup")
    async def _startup() -> None:
        hub.set_loop(asyncio.get_running_loop())
        if outbox_publisher is not None:
            outbox_publisher.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if outbox_publisher is not None:
            outbox_publisher.stop()

    def problem(status: int, code: str, title: str, detail: str, instance: str | None = None) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            media_type="application/problem+json",
            content={
                "type": f"urn:kraken:problem:{code}",
                "title": title,
                "status": status,
                "detail": detail,
                "instance": instance,
                "code": code,
            },
        )

    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return problem(404, "resource.not_found", "Resource not found", str(exc), str(request.url.path))

    @app.exception_handler(ValidationError)
    async def invalid(request: Request, exc: ValidationError) -> JSONResponse:
        return problem(422, "command.invalid", "Invalid command", str(exc), str(request.url.path))

    @app.exception_handler(ConflictError)
    async def conflict(request: Request, exc: ConflictError) -> JSONResponse:
        return problem(409, "revision.conflict", "Revision conflict", str(exc), str(request.url.path))

    @app.exception_handler(ForbiddenError)
    async def forbidden(request: Request, exc: ForbiddenError) -> JSONResponse:
        return problem(403, "authorization.denied", "Permission denied", str(exc), str(request.url.path))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        code = "auth.required" if exc.status_code == 401 else "request.rejected"
        return problem(exc.status_code, code, "Request rejected", str(exc.detail), str(request.url.path))

    @app.exception_handler(RequestValidationError)
    async def request_invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        return problem(422, "request.invalid", "Invalid request", str(exc), str(request.url.path))

    def principal(authorization: str | None = Header(default=None)) -> SessionPrincipal:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authentication required")
        token = authorization.removeprefix("Bearer ").strip()
        if session_resolver is not None:
            resolved = session_resolver(token)
            if resolved is None:
                raise HTTPException(status_code=401, detail="Session expired or revoked")
            return resolved
        if account_store is not None:
            account = account_store.resolve_session(token)
            if account is None:
                raise HTTPException(status_code=401, detail="Session expired or revoked")
            return SessionPrincipal(str(account.account_id), "local", token)
        if development and token:
            return SessionPrincipal(token, "development", token)
        raise HTTPException(status_code=401, detail="Authentication required")

    def shared_mutation_actor(subject: SessionPrincipal = Depends(principal)) -> SessionPrincipal:
        if development and subject.provider == "development":
            return subject
        if subject.provider != "gitlab":
            raise HTTPException(status_code=403, detail="Shared mutations require a GitLab principal")
        if live_gitlab_verifier is None or not live_gitlab_verifier(subject):
            raise HTTPException(status_code=503, detail="Live GitLab identity verification failed")
        return subject

    def command_context(
        actor: SessionPrincipal,
        idempotency_key: str | None,
        if_match: str | None,
        *,
        revision_required: bool,
    ) -> CommandContext:
        if not idempotency_key:
            raise ValidationError("Idempotency-Key is required")
        revision = None
        if if_match is not None:
            try:
                revision = int(if_match.strip('"'))
            except ValueError as exc:
                raise ValidationError("If-Match must contain an integer revision") from exc
        if revision_required and revision is None:
            raise ValidationError("If-Match is required")
        return CommandContext(actor.principal_id, idempotency_key, revision)

    @app.get(f"{API_PREFIX}/health")
    async def health() -> Any:
        status = backend.health()
        if status.get("status") != "ok":
            return problem(503, "service.degraded", "Service degraded", str(status.get("detail", "")))
        return status

    @app.post(f"{API_PREFIX}/auth/sessions")
    async def create_session(payload: dict[str, Any]) -> dict[str, Any]:
        if account_store is None:
            return {"provider": "development", "access_token": str(payload.get("username", "developer"))}
        try:
            session = account_store.authenticate(
                str(payload.get("username", "")), str(payload.get("password", ""))
            )
        except ValueError:
            session = None
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {
            "provider": "local",
            "access_token": session.token,
            "expires_at": session.expires_at,
            "principal": {
                "id": session.account.account_id,
                "display_name": session.account.display_name,
            },
        }

    @app.get(f"{API_PREFIX}/projects")
    async def list_projects(_: SessionPrincipal = Depends(principal)) -> dict[str, Any]:
        return {"items": backend.list_projects()}

    @app.post(f"{API_PREFIX}/projects", status_code=201)
    async def create_project(
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.create_project(
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=False),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}")
    async def get_project(project_id: str, _: SessionPrincipal = Depends(principal)) -> dict[str, Any]:
        return backend.get_project(project_id)

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}")
    async def rename_project(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.rename_project(
            project_id,
            str(payload.get("name", "")),
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/archive")
    async def archive_project(
        project_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.archive_project(
            project_id, command_context(actor, idempotency_key, if_match, revision_required=True)
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/restore")
    async def restore_project(
        project_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.restore_project(
            project_id, command_context(actor, idempotency_key, if_match, revision_required=True)
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/layers")
    async def list_layers(project_id: str, _: SessionPrincipal = Depends(principal)) -> dict[str, Any]:
        return {"items": backend.list_layers(project_id)}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/acl/{{principal_id}}")
    async def project_roles(
        project_id: str,
        principal_id: str,
        _: SessionPrincipal = Depends(principal),
    ) -> dict[str, Any]:
        return backend.project_roles(project_id, principal_id)

    @app.put(f"{API_PREFIX}/projects/{{project_id}}/acl/{{principal_id}}/{{role}}")
    async def assign_project_role(
        project_id: str,
        principal_id: str,
        role: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.assign_project_role(
            project_id,
            principal_id,
            role,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.delete(f"{API_PREFIX}/projects/{{project_id}}/acl/{{principal_id}}/{{role}}")
    async def revoke_project_role(
        project_id: str,
        principal_id: str,
        role: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.revoke_project_role(
            project_id,
            principal_id,
            role,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers", status_code=201)
    async def create_layer(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.create_layer(
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/rename")
    async def rename_layer(
        project_id: str,
        layer_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.rename_layer(
            project_id,
            layer_id,
            str(payload.get("name", "")),
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/reorder")
    async def reorder_layer(
        project_id: str,
        layer_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.reorder_layer(
            project_id,
            layer_id,
            int(payload.get("order", -1)),
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/reorder")
    async def reorder_layers(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return {
            "items": backend.reorder_layers(
                project_id,
                payload,
                command_context(actor, idempotency_key, None, revision_required=False),
            )
        }

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/archive")
    async def archive_layer(
        project_id: str,
        layer_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.archive_layer(
            project_id,
            layer_id,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/representations")
    async def list_representations(
        project_id: str,
        layer_id: str,
        _: SessionPrincipal = Depends(principal),
    ) -> dict[str, Any]:
        return {"items": backend.list_representations(project_id, layer_id)}

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/representations",
        status_code=201,
    )
    async def create_representation(
        project_id: str,
        layer_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.create_representation(
            project_id,
            layer_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/representations/{{representation_id}}")
    async def update_representation(
        project_id: str,
        layer_id: str,
        representation_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.update_representation(
            project_id, layer_id, representation_id, payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/viewport")
    async def viewport(
        project_id: str,
        layer_id: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int = 0,
        _: SessionPrincipal = Depends(principal),
    ) -> dict[str, Any]:
        return backend.matrix_viewport(project_id, layer_id=layer_id, x1=x1, y1=y1, x2=x2, y2=y2, lod=lod)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/history")
    async def history(
        project_id: str,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        _: SessionPrincipal = Depends(principal),
    ) -> dict[str, Any]:
        return backend.history(project_id, cursor=cursor, limit=limit)

    @app.websocket(f"{API_PREFIX}/ws")
    async def websocket(websocket: WebSocket) -> None:
        authorization = websocket.headers.get("authorization", "")
        token = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
        if session_resolver is not None:
            valid = session_resolver(token) is not None
        elif account_store is not None:
            valid = account_store.resolve_session(token) is not None
        else:
            valid = development and bool(token)
        if not valid:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        hub.register(websocket)
        await websocket.send_json({"type": "connected", "api_version": "v1"})
        try:
            while True:
                message = await websocket.receive_json()
                kind = str(message.get("type", ""))
                if kind == "ping":
                    await websocket.send_json({"type": "pong"})
                elif kind == "subscribe":
                    project_id = message.get("project_id")
                    hub.subscribe(
                        websocket,
                        project_id=None if project_id is None else str(project_id),
                        catalog=bool(message.get("catalog")),
                    )
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "project_id": None if project_id is None else str(project_id),
                            "catalog": bool(message.get("catalog")),
                        }
                    )
                elif kind == "unsubscribe":
                    project_id = message.get("project_id")
                    hub.unsubscribe(
                        websocket,
                        project_id=None if project_id is None else str(project_id),
                        catalog=bool(message.get("catalog")),
                    )
                    await websocket.send_json(
                        {
                            "type": "unsubscribed",
                            "project_id": None if project_id is None else str(project_id),
                            "catalog": bool(message.get("catalog")),
                        }
                    )
        except WebSocketDisconnect:
            hub.unregister(websocket)
            return
        finally:
            hub.unregister(websocket)

    return app


__all__ = ["API_PREFIX", "SessionPrincipal", "create_app"]
