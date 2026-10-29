"""Project catalog, lifecycle and access control."""

from __future__ import annotations

from typing import Any

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import NotFoundError


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    development = ctx.development
    access_mode = ctx.access_mode
    Body = ctx.Body
    Depends = ctx.Depends
    Header = ctx.Header
    principal = ctx.principal
    shared_mutation_actor = ctx.shared_mutation_actor
    project_reader = ctx.project_reader
    command_context = ctx.command_context
    has_project_access = ctx.has_project_access

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
