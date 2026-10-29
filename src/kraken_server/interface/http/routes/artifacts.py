"""Artifact series and managed blob transfer."""

from __future__ import annotations

import re
import tempfile
from typing import Any

from fastapi import Request

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import ValidationError


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    blob_gateway = ctx.blob_gateway
    Depends = ctx.Depends
    HTTPException = ctx.HTTPException
    Header = ctx.Header
    StreamingResponse = ctx.StreamingResponse
    run_in_threadpool = ctx.run_in_threadpool
    shared_mutation_actor = ctx.shared_mutation_actor
    project_reader = ctx.project_reader
    command_context = ctx.command_context
    transfer_identity = ctx.transfer_identity
    transfer_payload = ctx.transfer_payload
    transfer_ticket_context = ctx.transfer_ticket_context

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts")
    def list_artifacts(
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
    def create_artifact_series(
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
    def mutate_artifact_series(
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
    def list_artifact_versions(
        project_id: str,
        series_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_artifact_versions(project_id, series_id)}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/revision")
    def artifact_stream_revision(
        project_id: str,
        series_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, int]:
        return {"revision": backend.artifact_stream_revision(project_id, series_id)}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/active")
    def active_artifact_version(
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
                raise ValidationError("Content-Length is required and must match the uploaded blob size")
            staged.seek(0)

            def chunks() -> Any:
                while True:
                    chunk = staged.read(1024 * 1024)
                    if not chunk:
                        return
                    yield chunk

            return await run_in_threadpool(
                backend.add_managed_artifact_version,
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

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/uploads")
    def initiate_artifact_upload(
        project_id: str,
        series_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        if blob_gateway is None:
            return {"mode": "proxy"}
        context = command_context(actor, idempotency_key, if_match, revision_required=True)
        digest, size_bytes = transfer_identity(payload)
        command_payload = transfer_payload(payload, digest=digest, size_bytes=size_bytes)
        existing = backend.prepare_managed_artifact_upload(project_id, series_id, command_payload, context)
        if existing is not None:
            return {"mode": "completed", "artifact": existing}
        return blob_gateway.ticket(
            "upload",
            digest,
            size_bytes,
            context=transfer_ticket_context(actor, project_id, series_id, command_payload, context.idempotency_key),
        )

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
