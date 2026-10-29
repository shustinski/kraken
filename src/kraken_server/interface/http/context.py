"""Build the HTTP dependencies once, then let route modules close over them.

Nested FastAPI callables keep their annotations evaluated at definition time.
A postponed-annotation import would turn ``Request`` into a query parameter.
"""

import asyncio
import os
import re
import stat
import zipfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import anyio

from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import (
    CommandContext,
    ConflictError,
    ForbiddenError,
    InMemoryServerServices,
    NotFoundError,
    ServerServices,
    ValidationError,
)


class HttpContext:
    """Dependencies one FastAPI app closes over. Routes only read this object."""

    def __init__(self) -> None:
        self.app: Any = None
        self.backend: Any = None
        self.development: bool = False
        self.account_store: Any = None
        self.session_resolver: Callable[..., Any] | None = None
        self.live_gitlab_verifier: Callable[..., Any] | None = None
        self.access_mode: str = "acl"
        self.hub: Any = None
        self.heavy_operations: Any = None
        self.agent_token_store: Any = None
        self.agent_gateway: Any = None
        self.blob_gateway: Any = None
        self.Body: Any = None
        self.Depends: Any = None
        self.HTTPException: Any = None
        self.Header: Any = None
        self.Query: Any = None
        self.Request: Any = None
        self.StreamingResponse: Any = None
        self.WebSocket: Any = None
        self.WebSocketDisconnect: Any = None
        self.run_in_threadpool: Any = None
        self.principal: Any = None
        self.shared_mutation_actor: Any = None
        self.project_reader: Any = None
        self.command_context: Any = None
        self.agent_identity: Any = None
        self.desktop_identity: Any = None
        self.has_project_access: Any = None
        self.transfer_identity: Any = None
        self.transfer_payload: Any = None
        self.transfer_ticket_context: Any = None
        self.extract_review_archive: Any = None
        self.problem: Any = None
        self.system_roles: Any = None


def build_http_context(
    services: ServerServices | None,
    account_store: Any | None,
    session_resolver: Callable[[str], SessionPrincipal | None] | None,
    live_gitlab_verifier: Callable[[SessionPrincipal], bool] | None,
    project_access_mode: str | None,
    development: bool,
    connection_hub: Any | None,
    outbox_publisher: Any | None,
    agent_token_store: Any | None,
    agent_gateway: Any | None,
    blob_gateway: Any | None,
) -> HttpContext:
    try:
        from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
        from fastapi.concurrency import run_in_threadpool
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install Kraken with the 'server' extra to run Kraken Server") from exc

    if not development and services is None:
        raise RuntimeError("Production Kraken Server requires an injected persistent ServerServices adapter")
    if not development and account_store is None and session_resolver is None:
        raise RuntimeError("Production Kraken Server requires an authenticated session resolver")
    backend = services or InMemoryServerServices()
    access_mode = (project_access_mode or os.environ.get("KRAKEN_PROJECT_ACCESS_MODE", "acl")).strip().lower()
    if access_mode not in {"acl", "trusted_network"}:
        raise RuntimeError("KRAKEN_PROJECT_ACCESS_MODE must be 'acl' or 'trusted_network'")
    from kraken_server.outbox import ConnectionHub

    hub = connection_hub or ConnectionHub()
    heavy_operations = anyio.CapacityLimiter(2)

    @asynccontextmanager
    async def _lifespan(_app: Any):  # type: ignore[no-untyped-def]
        hub.set_loop(asyncio.get_running_loop())
        from anyio.to_thread import current_default_thread_limiter

        pool_size = int(os.environ.get("KRAKEN_DB_POOL_SIZE", "10"))
        max_overflow = int(os.environ.get("KRAKEN_DB_MAX_OVERFLOW", "10"))
        current_default_thread_limiter().total_tokens = max(pool_size + max_overflow, 1)
        if outbox_publisher is not None:
            outbox_publisher.start()
        if blob_gateway is not None:
            await run_in_threadpool(blob_gateway.start)
        pipeline_stop = asyncio.Event()

        async def pipeline_loop():
            import logging
            while not pipeline_stop.is_set():
                try:
                    await run_in_threadpool(backend.run_pipeline_cycle)
                    if hasattr(backend, "run_analysis_cycle"):
                        await run_in_threadpool(backend.run_analysis_cycle)
                    if hasattr(backend, "run_stitch_cycle"):
                        await run_in_threadpool(backend.run_stitch_cycle)
                except Exception:
                    logging.getLogger(__name__).exception("Pipeline scheduler cycle failed")
                try:
                    await asyncio.wait_for(pipeline_stop.wait(), timeout=1.0)
                except TimeoutError:
                    pass

        pipeline_task = (asyncio.create_task(pipeline_loop())
                         if hasattr(backend, "run_pipeline_cycle") else None)
        try:
            yield
        finally:
            pipeline_stop.set()
            if pipeline_task is not None:
                await pipeline_task
            if outbox_publisher is not None:
                outbox_publisher.stop()
            if blob_gateway is not None:
                blob_gateway.stop()

    app = FastAPI(
        title="Kraken Server",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=_lifespan,
    )
    from kraken_manager.domain.roles import reset_acting_role, set_acting_role

    @app.middleware("http")
    async def _remember_acting_role(request: Request, call_next):  # type: ignore[no-untyped-def]
        token = set_acting_role(request.headers.get("x-kraken-role"))
        try:
            return await call_next(request)
        finally:
            reset_acting_role(token)

    from kraken_server.operation_log import configure_operation_log, record_http_request

    configure_operation_log(os.environ.get("KRAKEN_LOG_LEVEL", "none"))

    @app.middleware("http")
    async def _journal_request(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        record_http_request(request.method, request.url.path, response.status_code)
        return response

    app.state.connection_hub = hub
    app.state.outbox_publisher = outbox_publisher
    if hasattr(backend, "engine"):
        from kraken_server.project_deletion import ProjectOperationMiddleware

        app.add_middleware(ProjectOperationMiddleware, engine=backend.engine)

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
            actual_revision=(None if revision_match is None else int(revision_match.group(2))),
        )

    @app.exception_handler(PermissionError)
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
            return SessionPrincipal(
                str(uuid5(NAMESPACE_URL, f"kraken:development:{token}")),
                "development",
                token,
            )
        raise HTTPException(status_code=401, detail="Authentication required")

    def enforce_acting_role(request: Request, subject: SessionPrincipal) -> None:
        from kraken_manager.domain.roles import CATALOG, action_for_request, current_acting_role

        acting = current_acting_role()
        if not acting:
            return
        action = action_for_request(request.method, request.url.path)
        if not action:
            return
        project_id = ""
        marker = f"{API_PREFIX}/projects/"
        path = request.url.path
        if marker in path:
            project_id = path.split(marker, 1)[1].split("/", 1)[0]
        elif agent_gateway is not None and request.path_params.get("job_id"):
            # Desktop job routes do not contain /projects/{id}. Resolve their
            # project from the authenticated owner's lease before checking roles.
            try:
                _, project_id = agent_gateway.desktop_identity(
                    str(request.path_params["job_id"]), subject.principal_id
                )
            except PermissionError as exc:
                raise ForbiddenError(str(exc)) from exc
        held: set[str] = set()
        if project_id:
            try:
                held = set(backend.project_roles(project_id, subject.principal_id).get("roles", ()))
            except (NotFoundError, ValidationError):
                held = set()
        decision = CATALOG.resolve(
            held_roles=held,
            acting_role=acting,
            action=action,
            is_server_admin=False,
        )
        if not decision.allowed:
            raise ForbiddenError(decision.message)

    def shared_mutation_actor(
        request: Request,
        subject: SessionPrincipal = Depends(principal),
    ) -> SessionPrincipal:
        if not (development and subject.provider == "development"):
            if subject.provider == "gitlab":
                if live_gitlab_verifier is None or not live_gitlab_verifier(subject):
                    raise HTTPException(status_code=503, detail="Live GitLab identity verification failed")
            elif subject.provider != "local":
                raise HTTPException(status_code=403, detail="Unsupported identity provider")
        enforce_acting_role(request, subject)
        return subject

    def system_roles(subject: SessionPrincipal) -> frozenset[str]:
        if subject.provider == "local" and account_store is not None and hasattr(account_store, "global_roles_for"):
            return frozenset(account_store.global_roles_for(subject.principal_id))
        lookup = getattr(backend, "principal_system_roles", None)
        if lookup is not None:
            return frozenset(lookup(subject.principal_id))
        actor = next(
            (
                item
                for item in backend.list_principals(include_inactive=False)
                if str(item.get("principal_id")) == subject.principal_id
            ),
            None,
        )
        return frozenset(() if actor is None else actor.get("system_roles", ()))

    def has_project_access(project_id: str, subject: SessionPrincipal) -> bool:
        if development and subject.provider == "development":
            return True
        if access_mode == "trusted_network":
            return True
        lookup = getattr(backend, "project_ids_for_principal", None)
        if lookup is not None:
            return project_id in set(lookup(subject.principal_id))
        roles = backend.project_roles(project_id, subject.principal_id).get("roles", ())
        return bool(roles)

    def project_reader(
        project_id: str,
        subject: SessionPrincipal = Depends(principal),
    ) -> SessionPrincipal:
        if has_project_access(project_id, subject):
            return subject
        raise ForbiddenError("Project membership is required")

    def desktop_identity(request: Request, subject: SessionPrincipal, job_id: str) -> tuple[Any, str]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        shared_mutation_actor(request, subject)
        try:
            identity, project_id = agent_gateway.desktop_identity(job_id, subject.principal_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not has_project_access(project_id, subject):
            raise ForbiddenError("Project membership is required")
        backend.authorize_desktop_job(project_id, subject.principal_id)
        from dataclasses import replace
        identity = replace(identity, lease_claim=request.headers.get("X-Kraken-Lease-Claim"))
        return identity, project_id

    def agent_identity(request: Request, authorization: str | None = Header(default=None)) -> Any:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Agent authentication required")
        token = authorization.removeprefix("Bearer ").strip()
        identity = None if agent_token_store is None else agent_token_store.resolve(token)
        if identity is None and not token.startswith("kat_") and request.path_params.get("job_id"):
            return desktop_identity(request, principal(authorization), str(request.path_params["job_id"]))[0]
        if identity is None:
            raise HTTPException(status_code=401, detail="Agent token is invalid or revoked")
        from dataclasses import replace
        return replace(identity, lease_claim=request.headers.get("X-Kraken-Lease-Claim"))

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

    def transfer_identity(payload: dict[str, Any]) -> tuple[str, int]:
        digest = str(payload.get("sha256", "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValidationError("A valid SHA-256 digest is required")
        size = payload.get("size_bytes")
        if isinstance(size, bool):
            raise ValidationError("A non-negative blob size is required")
        try:
            size_bytes = int(size)
        except (TypeError, ValueError) as exc:
            raise ValidationError("A non-negative blob size is required") from exc
        if size_bytes < 0:
            raise ValidationError("A non-negative blob size is required")
        return digest, size_bytes

    def transfer_payload(payload: dict[str, Any], *, digest: str, size_bytes: int) -> dict[str, Any]:
        from kraken_server.blob_gateway import artifact_parameters_digest

        parameters = payload.get("parameters", {})
        try:
            artifact_parameters_digest(parameters)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return {
            "filename": str(payload.get("filename", "")),
            "media_type": str(payload.get("media_type", "application/octet-stream")),
            "sha256": digest,
            "size_bytes": size_bytes,
            "parent_version_id": payload.get("parent_version_id"),
            **({"parameters": parameters} if parameters else {}),
        }
    def transfer_ticket_context(
        actor: SessionPrincipal,
        project_id: str,
        series_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, str]:
        from kraken_server.blob_gateway import artifact_parameters_digest

        return {
            "actor": actor.principal_id,
            "project": project_id,
            "series": series_id,
            "idempotency_key": idempotency_key,
            "filename": str(payload["filename"]),
            "media_type": str(payload["media_type"]),
            "parent_version_id": str(payload.get("parent_version_id") or ""),
            **({"parameters_sha256": artifact_parameters_digest(payload["parameters"])}
               if payload.get("parameters") else {}),
        }

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
                if entry.file_size >= 1024 * 1024 and entry.file_size / max(1, entry.compress_size) > 200:
                    raise ValidationError(f"Review archive compression ratio is unsafe: {entry.filename}")
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
                            raise ValidationError(f"Review archive member expanded unexpectedly: {entry.filename}")
                        output.write(chunk)
                if observed != entry.file_size:
                    raise ValidationError(f"Review archive member size differs: {entry.filename}")

    ctx = HttpContext()
    ctx.app = app
    ctx.backend = backend
    ctx.development = development
    ctx.account_store = account_store
    ctx.session_resolver = session_resolver
    ctx.live_gitlab_verifier = live_gitlab_verifier
    ctx.access_mode = access_mode
    ctx.hub = hub
    ctx.heavy_operations = heavy_operations
    ctx.agent_token_store = agent_token_store
    ctx.agent_gateway = agent_gateway
    ctx.blob_gateway = blob_gateway
    ctx.Body = Body
    ctx.Depends = Depends
    ctx.HTTPException = HTTPException
    ctx.Header = Header
    ctx.Query = Query
    ctx.Request = Request
    ctx.StreamingResponse = StreamingResponse
    ctx.WebSocket = WebSocket
    ctx.WebSocketDisconnect = WebSocketDisconnect
    ctx.run_in_threadpool = run_in_threadpool
    ctx.principal = principal
    ctx.shared_mutation_actor = shared_mutation_actor
    ctx.project_reader = project_reader
    ctx.command_context = command_context
    ctx.agent_identity = agent_identity
    ctx.desktop_identity = desktop_identity
    ctx.has_project_access = has_project_access
    ctx.transfer_identity = transfer_identity
    ctx.transfer_payload = transfer_payload
    ctx.transfer_ticket_context = transfer_ticket_context
    ctx.extract_review_archive = extract_review_archive
    ctx.problem = problem
    ctx.system_roles = system_roles
    return ctx
