"""create_app section 2."""
import asyncio
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import anyio
from kraken_server.services import (
    CommandContext,
    ConflictError,
    ForbiddenError,
    InMemoryServerServices,
    NotFoundError,
    ServerServices,
    ValidationError,
)
from kraken_server.app_parts.module_state import API_PREFIX
from kraken_server.app_parts.session_principal import SessionPrincipal

def _create_app_section_2(account_store, development, blob_gateway, Body, Depends, HTTPException, Header, access_mode, app, backend, command_context, has_project_access, heavy_operations, principal, problem, project_reader, run_in_threadpool, shared_mutation_actor):
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

    @app.get(f"{API_PREFIX}/projects")
    def list_projects(
        include_archived: bool = False,
        subject: SessionPrincipal = Depends(principal),
    ) -> dict[str, Any]:
        items = backend.list_projects(include_archived=include_archived)
        if (development and subject.provider == "development") or access_mode == "trusted_network":
            return {"items": items}
        lookup = getattr(backend, "project_ids_for_principal", None)
        if lookup is None:
            return {"items": [item for item in items if has_project_access(str(item["project_id"]), subject)]}
        visible = set(lookup(subject.principal_id))
        return {"items": [item for item in items if str(item["project_id"]) in visible]}

    @app.get(f"{API_PREFIX}/principals")
    def list_principals(
        include_inactive: bool = False,
        _: SessionPrincipal = Depends(principal),
    ) -> dict[str, Any]:
        return {
            "items": backend.list_principals(
                include_inactive=include_inactive,
            )
        }

    @app.get(f"{API_PREFIX}/performers")
    def list_performers(
        include_archived: bool = False,
        _: SessionPrincipal = Depends(principal),
    ) -> dict[str, Any]:
        return {"items": backend.list_performers(include_archived=include_archived)}

    @app.post(f"{API_PREFIX}/projects", status_code=201)
    def create_project(
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
    def get_project(project_id: str, _: SessionPrincipal = Depends(project_reader)) -> dict[str, Any]:
        return backend.get_project(project_id)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/workspace")
    def get_project_workspace(
        project_id: str, _: SessionPrincipal = Depends(project_reader)
    ) -> dict[str, Any]:
        workspace = getattr(backend, "project_workspace", lambda _project_id: None)(project_id)
        if workspace is None:
            raise NotFoundError(project_id)
        return workspace

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}")
    def rename_project(
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
    def archive_project(
        project_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.archive_project(
            project_id, command_context(actor, idempotency_key, if_match, revision_required=True)
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/deletion-requests", status_code=202)
    def request_project_deletion(
        project_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        payload: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        reason = "" if not payload else payload.get("reason", "")
        return backend.request_project_deletion(project_id, actor.principal_id, reason)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/deletion-requests")
    def project_deletion_requests(
        project_id: str, _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.project_deletion_requests(project_id)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/restore")
    def restore_project(
        project_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.restore_project(
            project_id, command_context(actor, idempotency_key, if_match, revision_required=True)
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/layers")
    def list_layers(
        project_id: str,
        include_archived: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_layers(project_id, include_archived=include_archived)}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/acl/{{principal_id}}")
    def project_roles(
        project_id: str,
        principal_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return backend.project_roles(project_id, principal_id)

    @app.put(f"{API_PREFIX}/projects/{{project_id}}/acl/{{principal_id}}/{{role}}")
    def assign_project_role(
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
    def revoke_project_role(
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
    def create_layer(
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

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/files")
    def get_layer_files(
        project_id: str,
        layer_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        binding = getattr(backend, "layer_file_binding", lambda *_args: None)(project_id, layer_id)
        if binding is None:
            raise NotFoundError(layer_id)
        return binding

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/scan")
    def scan_layer_source(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
    ) -> dict[str, Any]:
        directory = payload.get("directory")
        if not isinstance(directory, str) or not directory.strip():
            raise ValidationError("A source directory is required")
        operation = getattr(backend, "scan_layer_source", None)
        if operation is None:
            raise HTTPException(status_code=503, detail="Server source scanning is unavailable")
        return operation(project_id, directory, CommandContext(actor.principal_id, str(uuid4()), None))

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/external", status_code=201)
    def create_external_layer(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        operation = getattr(backend, "create_external_layer", None)
        if operation is None:
            raise HTTPException(status_code=503, detail="Сервер не принимает файловые слои")
        return operation(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/import", status_code=201)
    async def import_workspace_layer(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        operation = getattr(backend, "import_workspace_layer", None)
        if operation is None:
            raise HTTPException(status_code=503, detail="Сервер не принимает импорт слоя")
        async with heavy_operations:
            return await run_in_threadpool(
                operation,
                project_id,
                payload,
                command_context(actor, idempotency_key, None, revision_required=False),
            )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/rename")
    def rename_layer(
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
    return extract_review_archive, transfer_ticket_context
