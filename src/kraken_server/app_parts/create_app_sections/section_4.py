"""create_app section 4."""
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

def _create_app_section_4(agent_gateway, blob_gateway, Depends, HTTPException, Header, Request, StreamingResponse, agent_identity, app, backend, command_context, desktop_identity, extract_review_archive, heavy_operations, project_reader, run_in_threadpool, shared_mutation_actor, transfer_identity, transfer_payload, transfer_ticket_context):
    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/uploads/complete",
        status_code=201,
    )
    def complete_artifact_upload(
        project_id: str,
        series_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        if blob_gateway is None:
            raise HTTPException(status_code=503, detail="Blob Gateway is not configured")
        context = command_context(actor, idempotency_key, if_match, revision_required=True)
        try:
            claims = blob_gateway.signer.verify(str(payload.get("upload_token", "")), operation="upload")
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        digest, size_bytes = transfer_identity(payload)
        command_payload = transfer_payload(payload, digest=digest, size_bytes=size_bytes)
        expected_context = transfer_ticket_context(
            actor, project_id, series_id, command_payload, context.idempotency_key
        )
        if claims.get("digest") != digest or claims.get("size") != size_bytes or claims.get("ctx") != expected_context:
            raise HTTPException(status_code=401, detail="Blob Gateway ticket does not match completed upload")
        return backend.register_managed_artifact_upload(project_id, series_id, command_payload, context)

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/external",
        status_code=201,
    )
    def add_external_artifact_version(
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
    def artifact_version_metadata(
        project_id: str,
        version_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return backend.get_artifact_version(project_id, version_id)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/versions/{{version_id}}")
    def download_artifact_version(
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

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/versions/{{version_id}}/download-ticket")
    def artifact_download_ticket(
        project_id: str,
        version_id: str,
        actor: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        if blob_gateway is None:
            return {"mode": "proxy"}
        metadata = backend.get_artifact_version(project_id, version_id)
        stored_blob = metadata.get("blob")
        if not isinstance(stored_blob, dict):
            raise ValidationError("External artifact versions do not have managed bytes")
        digest = str(stored_blob.get("sha256", ""))
        size_bytes = int(stored_blob.get("size_bytes", -1))
        return blob_gateway.ticket(
            "download",
            digest,
            size_bytes,
            context={"actor": actor.principal_id, "project": project_id, "version": version_id},
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/notes")
    def list_notes(
        project_id: str,
        layer_id: str | None = None,
        frame_id: str | None = None,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_notes(project_id, layer_id=layer_id, frame_id=frame_id)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/notes", status_code=201)
    def create_note(
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
    def revise_note(
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
    def list_reviews(
        project_id: str,
        active_only: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_review_batches(project_id, active_only=active_only)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/reviews", status_code=201)
    def create_review(
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
    def mutate_review(
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
        def build_archive() -> tuple[Path, Path, dict[str, Any]]:
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
                with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as output:
                    for source in sorted(package_dir.rglob("*")):
                        if source.is_file():
                            output.write(source, source.relative_to(package_dir).as_posix())
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            return staging, archive, issued

        async with heavy_operations:
            staging, archive, issued = await run_in_threadpool(build_archive)

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
            context = command_context(actor, idempotency_key, if_match, revision_required=True)

            def apply_return() -> dict[str, Any]:
                source.mkdir()
                extract_review_archive(archive, source)
                if commit:
                    return backend.commit_review_return(project_id, batch_id, str(source), context)
                return backend.inspect_review_return(project_id, batch_id, str(source), context)

            async with heavy_operations:
                return await run_in_threadpool(apply_return)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}/return/preflight")
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

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}/return/commit")
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

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/work-orders")
    def list_work_orders(project_id: str, _: SessionPrincipal = Depends(project_reader)) -> dict[str, Any]:
        return {"items": backend.list_work_orders(project_id)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/work-orders", status_code=201)
    def issue_work_order(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return backend.issue_work_order(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/work-orders/{{order_id}}/cancel")
    def cancel_work_order(
        project_id: str,
        order_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return backend.cancel_work_order(
            project_id,
            order_id,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/work-orders/{{order_id}}/complete")
    def complete_work_order(
        project_id: str,
        order_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return backend.complete_work_order(
            project_id,
            order_id,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/heartbeat")
    def heartbeat_agent_job(
        job_id: str,
        payload: dict[str, Any],
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        return {"lease_until": agent_gateway.heartbeat(job_id, agent, seconds=int(payload.get("lease_seconds", 60)))}
