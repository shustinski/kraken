"""Plugin job submission and agent lease heartbeat."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import ConflictError, NotFoundError


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    agent_gateway = ctx.agent_gateway
    Depends = ctx.Depends
    HTTPException = ctx.HTTPException
    Header = ctx.Header
    shared_mutation_actor = ctx.shared_mutation_actor
    project_reader = ctx.project_reader
    command_context = ctx.command_context
    agent_identity = ctx.agent_identity
    desktop_identity = ctx.desktop_identity

    @app.post(f"{API_PREFIX}/agent/lease")
    def lease_agent_job(
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

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/jobs/{{job_id}}/desktop-lease")
    def lease_desktop_job(
        project_id: str,
        job_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
    ) -> dict[str, Any]:
        identity, actual_project_id = desktop_identity(request, actor, job_id)
        if actual_project_id != project_id:
            raise NotFoundError("Job was not found in the project")
        lease = agent_gateway.lease(identity, job_id=job_id, seconds=60)
        if lease is None:
            raise ConflictError("The interactive job is already running or has finished")
        backend.start_desktop_job(
            project_id, job_id, command_context(actor, f"desktop-start:{job_id}", None, revision_required=False)
        )
        return {"job": lease}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/jobs")
    def list_project_jobs(
        project_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_plugin_jobs(project_id)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/jobs", status_code=202)
    def submit_project_job(
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
    def cancel_project_job(
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
    def heartbeat_agent_job(
        job_id: str,
        payload: dict[str, Any],
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        return {"lease_until": agent_gateway.heartbeat(
            job_id, agent, seconds=int(payload.get("lease_seconds", 60)), telemetry=payload.get("telemetry"),
        )}
