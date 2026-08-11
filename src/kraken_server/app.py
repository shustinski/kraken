"""FastAPI application factory; all adapter selection happens here."""

# FastAPI intentionally declares dependency providers in endpoint defaults.
# ruff: noqa: B008

import asyncio
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    agent_token_store: Any | None = None,
    agent_gateway: Any | None = None,
) -> Any:
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse, StreamingResponse
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

    def problem(
        status: int,
        code: str,
        title: str,
        detail: str,
        instance: str | None = None,
        **extensions: Any,
    ) -> JSONResponse:
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
                **extensions,
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
        detail = str(exc)
        revision_match = re.search(r"Expected .* revision (\d+), found (\d+)", detail)
        segments = [segment for segment in request.url.path.split("/") if segment]
        entity_kind = "project"
        entity_id = next(
            (segments[index + 1] for index, value in enumerate(segments[:-1]) if value == "projects"),
            "",
        )
        for resource, kind in (
            ("artifacts", "artifact_series"),
            ("notes", "note"),
            ("reviews", "review_batch"),
            ("jobs", "plugin_job"),
            ("layers", "layer"),
            ("representations", "representation"),
        ):
            if resource in segments:
                index = segments.index(resource)
                entity_kind = kind
                if index + 1 < len(segments):
                    entity_id = segments[index + 1]
        return problem(
            409,
            "revision.conflict",
            "Revision conflict",
            detail,
            str(request.url.path),
            entity_kind=entity_kind,
            entity_id=entity_id,
            actual_revision=(
                None if revision_match is None else int(revision_match.group(2))
            ),
        )

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

    def has_project_access(project_id: str, subject: SessionPrincipal) -> bool:
        if development and subject.provider == "development":
            return True
        roles = backend.project_roles(project_id, subject.principal_id).get("roles", ())
        if roles:
            return True
        actor = next(
            (
                item
                for item in backend.list_principals(include_inactive=False)
                if str(item.get("principal_id")) == subject.principal_id
            ),
            None,
        )
        return actor is not None and "administrator" in actor.get("system_roles", ())

    def project_reader(
        project_id: str,
        subject: SessionPrincipal = Depends(principal),
    ) -> SessionPrincipal:
        if has_project_access(project_id, subject):
            return subject
        raise ForbiddenError("Project membership is required")

    def agent_identity(authorization: str | None = Header(default=None)) -> Any:
        if agent_token_store is None or not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Agent authentication required")
        identity = agent_token_store.resolve(authorization.removeprefix("Bearer ").strip())
        if identity is None:
            raise HTTPException(status_code=401, detail="Agent token is invalid or revoked")
        return identity

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

    def extract_review_archive(archive: Path, destination: Path) -> None:
        total = 0
        with zipfile.ZipFile(archive) as package:
            entries = package.infolist()
            if len(entries) > 250_000:
                raise ValidationError("Review package has too many entries")
            for entry in entries:
                relative = PurePosixPath(entry.filename)
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or "\\" in entry.filename
                    or any(":" in part for part in relative.parts)
                    or stat.S_ISLNK(entry.external_attr >> 16)
                ):
                    raise ValidationError(f"Unsafe review archive member: {entry.filename}")
                if entry.file_size > 2 * 1024**3:
                    raise ValidationError(f"Review archive member is too large: {entry.filename}")
                if (
                    entry.file_size >= 1024 * 1024
                    and entry.file_size / max(1, entry.compress_size) > 200
                ):
                    raise ValidationError(
                        f"Review archive compression ratio is unsafe: {entry.filename}"
                    )
                total += entry.file_size
                if total > 16 * 1024**3:
                    raise ValidationError("Review archive is too large")
                target = destination.joinpath(*relative.parts)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                observed = 0
                with package.open(entry) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        observed += len(chunk)
                        if observed > entry.file_size:
                            raise ValidationError(
                                f"Review archive member expanded unexpectedly: {entry.filename}"
                            )
                        output.write(chunk)
                if observed != entry.file_size:
                    raise ValidationError(
                        f"Review archive member size differs: {entry.filename}"
                    )

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
    async def list_projects(
        include_archived: bool = False,
        subject: SessionPrincipal = Depends(principal),
    ) -> dict[str, Any]:
        items = backend.list_projects(include_archived=include_archived)
        return {
            "items": [
                item
                for item in items
                if has_project_access(str(item["project_id"]), subject)
            ]
        }

    @app.get(f"{API_PREFIX}/principals")
    async def list_principals(
        include_inactive: bool = False,
        _: SessionPrincipal = Depends(principal),  # noqa: B008 - FastAPI dependency declaration
    ) -> dict[str, Any]:
        return {
            "items": backend.list_principals(
                include_inactive=include_inactive,
            )
        }

    @app.get(f"{API_PREFIX}/performers")
    async def list_performers(
        include_archived: bool = False,
        _: SessionPrincipal = Depends(principal),
    ) -> dict[str, Any]:
        return {
            "items": backend.list_performers(include_archived=include_archived)
        }

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
    async def get_project(project_id: str, _: SessionPrincipal = Depends(project_reader)) -> dict[str, Any]:
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
    async def list_layers(
        project_id: str,
        include_archived: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {
            "items": backend.list_layers(
                project_id, include_archived=include_archived
            )
        }

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/acl/{{principal_id}}")
    async def project_roles(
        project_id: str,
        principal_id: str,
        _: SessionPrincipal = Depends(project_reader),
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
        include_archived: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {
            "items": backend.list_representations(
                project_id, layer_id, include_archived=include_archived
            )
        }

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
        include_missing: bool = True,
        representation_id: list[str] = Query(default=[]),
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return backend.matrix_viewport(
            project_id,
            layer_id=layer_id,
            representation_ids=representation_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            lod=lod,
            include_missing=include_missing,
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/frames")
    async def frame_states(
        project_id: str,
        layer_id: str,
        representation_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        project = backend.get_project(project_id)
        viewport = backend.matrix_viewport(
            project_id,
            layer_id=layer_id,
            representation_ids=(representation_id,),
            x1=1,
            y1=1,
            x2=int(project["width"]),
            y2=int(project["height"]),
            lod=0,
            include_missing=False,
        )
        return {"items": viewport["cells"], "revision": viewport["revision"]}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/history")
    async def history(
        project_id: str,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return backend.history(project_id, cursor=cursor, limit=limit)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/statistics")
    async def statistics(
        project_id: str,
        start: datetime,
        end: datetime,
        timezone: str = "UTC",
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValidationError("Statistics timestamps must include a timezone")
        try:
            report_timezone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValidationError(f"Unknown statistics timezone: {timezone}") from exc
        return backend.statistics(
            project_id, start=start, end=end, timezone=report_timezone
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/analyses/karakal", status_code=201)
    async def publish_karakal_analysis(
        project_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        payload = dict(await request.json())
        payload.setdefault("run_id", str(uuid4()))
        return backend.publish_karakal_analysis(
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/pipeline-actions", status_code=201)
    async def append_pipeline_event(
        project_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.append_pipeline_event(
            project_id,
            dict(await request.json()),
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts")
    async def list_artifacts(
        project_id: str,
        layer_id: str | None = None,
        representation_id: str | None = None,
        frame_id: str | None = None,
        include_archived: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {
            "items": backend.list_artifact_series(
                project_id,
                layer_id=layer_id,
                representation_id=representation_id,
                frame_id=frame_id,
                include_archived=include_archived,
            )
        }

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/artifacts", status_code=201)
    async def create_artifact_series(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return backend.create_artifact_series(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}")
    async def mutate_artifact_series(
        project_id: str,
        series_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.mutate_artifact_series(
            project_id,
            series_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/versions")
    async def list_artifact_versions(
        project_id: str,
        series_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_artifact_versions(project_id, series_id)}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/revision")
    async def artifact_stream_revision(
        project_id: str,
        series_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, int]:
        return {"revision": backend.artifact_stream_revision(project_id, series_id)}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/active")
    async def active_artifact_version(
        project_id: str,
        series_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"item": backend.get_active_artifact_version(project_id, series_id)}

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/versions",
        status_code=201,
    )
    async def upload_artifact_version(
        project_id: str,
        series_id: str,
        request: Request,
        filename: str,
        media_type: str = "application/octet-stream",
        sha256: str = "",
        parent_version_id: str | None = None,
        content_length: int | None = Header(default=None, alias="Content-Length"),
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        context = command_context(actor, idempotency_key, if_match, revision_required=True)
        if re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
            raise ValidationError("A valid SHA-256 digest is required")
        sha256 = sha256.lower()
        received = 0
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as staged:
            async for chunk in request.stream():
                received += len(chunk)
                staged.write(chunk)
            if content_length is None or received != content_length:
                raise ValidationError(
                    "Content-Length is required and must match the uploaded blob size"
                )
            staged.seek(0)

            def chunks() -> Any:
                while True:
                    chunk = staged.read(1024 * 1024)
                    if not chunk:
                        return
                    yield chunk

            return backend.add_managed_artifact_version(
                project_id,
                series_id,
                {
                    "filename": filename,
                    "media_type": media_type,
                    "sha256": sha256,
                    "parent_version_id": parent_version_id,
                },
                chunks(),
                context,
            )

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/external",
        status_code=201,
    )
    async def add_external_artifact_version(
        project_id: str,
        series_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.add_external_artifact_version(
            project_id,
            series_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/versions/{{version_id}}/metadata")
    async def artifact_version_metadata(
        project_id: str,
        version_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return backend.get_artifact_version(project_id, version_id)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/versions/{{version_id}}")
    async def download_artifact_version(
        project_id: str,
        version_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> Any:
        metadata = backend.get_artifact_version(project_id, version_id)
        return StreamingResponse(
            backend.iter_artifact_bytes(project_id, version_id),
            media_type=str(metadata.get("media_type", "application/octet-stream")),
            headers={"Content-Disposition": f'attachment; filename="{metadata.get("filename", version_id)}"'},
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/notes")
    async def list_notes(
        project_id: str,
        layer_id: str | None = None,
        frame_id: str | None = None,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_notes(project_id, layer_id=layer_id, frame_id=frame_id)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/notes", status_code=201)
    async def create_note(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return backend.create_note(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}/notes/{{note_id}}")
    async def revise_note(
        project_id: str,
        note_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.revise_note(
            project_id,
            note_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/reviews")
    async def list_reviews(
        project_id: str,
        active_only: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_review_batches(project_id, active_only=active_only)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/reviews", status_code=201)
    async def create_review(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.create_review_batch(
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}")
    async def mutate_review(
        project_id: str,
        batch_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.mutate_review_batch(
            project_id,
            batch_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}/package")
    async def export_review_package(
        project_id: str,
        batch_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> Any:
        staging = Path(tempfile.mkdtemp(prefix="kraken-review-export-"))
        package_dir = staging / "package"
        archive = staging / "review.zip"
        try:
            issued = backend.export_review_package(
                project_id,
                batch_id,
                str(package_dir),
                command_context(actor, idempotency_key, if_match, revision_required=True),
            )
            with zipfile.ZipFile(
                archive, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True
            ) as output:
                for source in sorted(package_dir.rglob("*")):
                    if source.is_file():
                        output.write(source, source.relative_to(package_dir).as_posix())
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        def chunks() -> Any:
            try:
                with archive.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        yield chunk
            finally:
                shutil.rmtree(staging, ignore_errors=True)

        return StreamingResponse(
            chunks(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="review-{batch_id}.zip"',
                "X-Kraken-Review-Revision": str(issued.get("revision", "")),
            },
        )

    async def receive_review_return(
        *,
        project_id: str,
        batch_id: str,
        request: Request,
        actor: SessionPrincipal,
        idempotency_key: str | None,
        if_match: str | None,
        commit: bool,
    ) -> dict[str, Any]:
        staging = Path(tempfile.mkdtemp(prefix="kraken-review-return-"))
        archive = staging / "return.zip"
        source = staging / "package"
        try:
            size = 0
            with archive.open("xb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 16 * 1024**3:
                        raise ValidationError("Review return archive is too large")
                    output.write(chunk)
            source.mkdir()
            extract_review_archive(archive, source)
            context = command_context(
                actor, idempotency_key, if_match, revision_required=True
            )
            if commit:
                return backend.commit_review_return(
                    project_id, batch_id, str(source), context
                )
            return backend.inspect_review_return(
                project_id, batch_id, str(source), context
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}/return/preflight"
    )
    async def review_return_preflight(
        project_id: str,
        batch_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return await receive_review_return(
            project_id=project_id,
            batch_id=batch_id,
            request=request,
            actor=actor,
            idempotency_key=idempotency_key,
            if_match=if_match,
            commit=False,
        )

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}/return/commit"
    )
    async def commit_review_return(
        project_id: str,
        batch_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return await receive_review_return(
            project_id=project_id,
            batch_id=batch_id,
            request=request,
            actor=actor,
            idempotency_key=idempotency_key,
            if_match=if_match,
            commit=True,
        )

    @app.post(f"{API_PREFIX}/agent/lease")
    async def lease_agent_job(
        payload: dict[str, Any],
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        requested = frozenset(str(item) for item in payload.get("capabilities", ()))
        if requested and not requested.issubset(agent.capabilities):
            raise HTTPException(status_code=403, detail="Requested capability is not granted to this agent")
        effective = agent if not requested else type(agent)(agent.token_id, agent.name, requested)
        lease = agent_gateway.lease(effective, seconds=int(payload.get("lease_seconds", 60)))
        return {"job": lease}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/jobs")
    async def list_project_jobs(
        project_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_plugin_jobs(project_id)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/jobs", status_code=202)
    async def submit_project_job(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return backend.submit_plugin_job(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/jobs/{{job_id}}/cancel")
    async def cancel_project_job(
        project_id: str,
        job_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.cancel_plugin_job(
            project_id,
            job_id,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/heartbeat")
    async def heartbeat_agent_job(
        job_id: str,
        payload: dict[str, Any],
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        return {
            "lease_until": agent_gateway.heartbeat(
                job_id, agent, seconds=int(payload.get("lease_seconds", 60))
            )
        }

    @app.get(f"{API_PREFIX}/agent/jobs/{{job_id}}/inputs/{{version_id}}")
    async def agent_job_input(
        job_id: str,
        version_id: str,
        agent: Any = Depends(agent_identity),
    ) -> Any:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        manifest = agent_gateway.manifest(job_id, agent)
        if version_id not in {str(item.artifact_version_id) for item in manifest.inputs}:
            raise HTTPException(status_code=403, detail="Artifact is not an input of the leased job")
        metadata = backend.get_artifact_version(str(manifest.project_id), version_id)
        return StreamingResponse(
            backend.iter_artifact_bytes(str(manifest.project_id), version_id),
            media_type=str(metadata.get("media_type", "application/octet-stream")),
            headers={"X-Kraken-Filename": str(metadata.get("filename", version_id))},
        )

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/publications")
    async def publish_agent_result(
        job_id: str,
        payload: dict[str, Any],
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        return {"accepted": agent_gateway.publish(job_id, agent, payload)}

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/outputs/{{output_id}}")
    async def upload_agent_output(
        job_id: str,
        output_id: str,
        request: Request,
        sha256: str,
        content_length: int | None = Header(default=None, alias="Content-Length"),
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        if re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
            raise ValidationError("A valid SHA-256 digest is required")
        sha256 = sha256.lower()
        received = 0
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as staged:
            async for chunk in request.stream():
                received += len(chunk)
                staged.write(chunk)
            if content_length is None or received != content_length:
                raise ValidationError(
                    "Content-Length is required and must match the uploaded blob size"
                )
            staged.seek(0)

            def chunks() -> Any:
                while True:
                    chunk = staged.read(1024 * 1024)
                    if not chunk:
                        return
                    yield chunk

            return agent_gateway.upload_output(
                job_id,
                output_id,
                agent,
                chunks(),
                expected_sha256=sha256,
            )

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/complete")
    async def complete_agent_job(
        job_id: str,
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        imported = backend.import_agent_result(job_id, agent)
        agent_gateway.finish(job_id, agent, failed=False)
        return {"state": "completed", "imported": imported}

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/fail")
    async def fail_agent_job(
        job_id: str,
        payload: dict[str, Any],
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        error = str(payload.get("error", ""))
        failed = backend.fail_agent_job(job_id, error)
        agent_gateway.finish(job_id, agent, failed=True, error=error)
        return {"state": "failed", "imported": failed}

    @app.websocket(f"{API_PREFIX}/ws")
    async def websocket(websocket: WebSocket) -> None:
        authorization = websocket.headers.get("authorization", "")
        token = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
        session: SessionPrincipal | None = None
        if session_resolver is not None:
            session = session_resolver(token)
        elif account_store is not None:
            account = account_store.resolve_session(token)
            if account is not None:
                session = SessionPrincipal(str(account.account_id), "local", token)
        else:
            if development and token:
                session = SessionPrincipal(token, "development", token)
        if session is None:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        hub.register(websocket)
        await websocket.send_json({"type": "connected", "api_version": "v1"})
        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(), timeout=10.0
                    )
                except TimeoutError:
                    if session_resolver is not None:
                        still_valid = session_resolver(token) is not None
                    elif account_store is not None:
                        still_valid = account_store.resolve_session(token) is not None
                    else:
                        still_valid = development and bool(token)
                    if not still_valid:
                        await websocket.close(code=4401)
                        return
                    continue
                kind = str(message.get("type", ""))
                if kind == "ping":
                    await websocket.send_json({"type": "pong"})
                elif kind == "subscribe":
                    project_id = message.get("project_id")
                    if project_id is not None and not has_project_access(
                        str(project_id), session
                    ):
                        await websocket.send_json(
                            {
                                "type": "error",
                                "code": "authorization.denied",
                                "project_id": str(project_id),
                            }
                        )
                        continue
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
