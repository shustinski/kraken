"""create_app section 5."""
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

def _create_app_section_5(account_store, session_resolver, development, agent_gateway, blob_gateway, Depends, HTTPException, Header, Request, StreamingResponse, WebSocket, WebSocketDisconnect, agent_identity, app, backend, has_project_access, hub, run_in_threadpool, transfer_identity):
    def input_metadata(manifest, version_id):
        from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest
        from kraken_core.plugin_protocol import PluginJobManifestV2
        if isinstance(manifest, AnalysisPartitionJobManifest):
            identifiers = {artifact.artifact_version_id for frame in manifest.frames for artifact in frame.artifacts}
            if version_id not in identifiers:
                raise HTTPException(status_code=403, detail="Artifact is not an input of the leased analysis partition")
            return backend.get_artifact_version(str(manifest.project_id), version_id)
        if isinstance(manifest, PluginJobManifestV2):
            asset = next((item for item in manifest.inputs if item.asset_id == version_id), None)
            if asset is None:
                raise HTTPException(status_code=403, detail="Artifact is not an input of the leased V2 job")
            metadata = backend.get_artifact_version(str(manifest.project_id), version_id)
            if metadata["sha256"] != asset.sha256:
                raise HTTPException(status_code=409, detail="V2 input digest does not match immutable metadata")
            return metadata
        if version_id in {str(item.artifact_version_id) for item in manifest.inputs}:
            return backend.get_artifact_version(str(manifest.project_id), version_id)
        if str(manifest.parameters.get("model_version_id", "")) != version_id:
            raise HTTPException(status_code=403, detail="Artifact is not an input of the leased job")
        from kraken_server.model_inputs import model_input_metadata
        from kraken_manager.application.errors import ConflictError as ModelInputConflict
        try:
            return model_input_metadata(backend, manifest)
        except (ValueError, ModelInputConflict) as exc:
            raise HTTPException(status_code=403, detail="Model artifact does not match the leased input") from exc

    @app.get(f"{API_PREFIX}/agent/jobs/{{job_id}}/inputs/{{version_id}}")
    def agent_job_input(
        job_id: str,
        version_id: str,
        agent: Any = Depends(agent_identity),
    ) -> Any:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        manifest = agent_gateway.manifest(job_id, agent)
        metadata = input_metadata(manifest, version_id)
        return StreamingResponse(
            backend.iter_artifact_bytes(str(manifest.project_id), version_id),
            media_type=str(metadata.get("media_type", "application/octet-stream")),
            headers={"X-Kraken-Filename": str(metadata.get("filename", version_id))},
        )

    @app.get(f"{API_PREFIX}/agent/jobs/{{job_id}}/inputs/{{version_id}}/download-ticket")
    def agent_job_input_ticket(
        job_id: str,
        version_id: str,
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        manifest = agent_gateway.manifest(job_id, agent)
        metadata = input_metadata(manifest, version_id)
        media_type = str(metadata.get("media_type", "application/octet-stream"))
        if blob_gateway is None:
            return {"mode": "proxy", "media_type": media_type}
        stored_blob = metadata.get("blob")
        if not isinstance(stored_blob, dict):
            raise ValidationError("Agent input does not have managed bytes")
        ticket = blob_gateway.ticket(
            "download",
            str(stored_blob.get("sha256", "")),
            int(stored_blob.get("size_bytes", -1)),
            context={"agent": agent.token_id, "claim": agent.lease_claim, "job": job_id, "version": version_id,
                     "project": str(manifest.project_id)},
        )
        return {**ticket, "media_type": media_type}

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/publications")
    def publish_agent_result(
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
                raise ValidationError("Content-Length is required and must match the uploaded blob size")
            staged.seek(0)

            def chunks() -> Any:
                while True:
                    chunk = staged.read(1024 * 1024)
                    if not chunk:
                        return
                    yield chunk

            return await run_in_threadpool(
                agent_gateway.upload_output,
                job_id,
                output_id,
                agent,
                chunks(),
                expected_sha256=sha256,
            )

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/outputs/{{output_id}}/upload-ticket")
    def agent_output_upload_ticket(
        job_id: str,
        output_id: str,
        payload: dict[str, Any],
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        if blob_gateway is None:
            return {"mode": "proxy"}
        manifest = agent_gateway.manifest(job_id, agent)
        digest, size_bytes = transfer_identity(payload)
        return blob_gateway.ticket(
            "upload",
            digest,
            size_bytes,
            context={"agent": agent.token_id, "claim": agent.lease_claim, "job": job_id, "output": output_id,
                     "project": str(manifest.project_id)},
        )

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/outputs/{{output_id}}/upload-complete")
    def complete_agent_output_upload(
        job_id: str,
        output_id: str,
        payload: dict[str, Any],
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None or blob_gateway is None:
            raise HTTPException(status_code=503, detail="Blob Gateway is not configured")
        try:
            claims = blob_gateway.signer.verify(str(payload.get("upload_token", "")), operation="upload")
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        digest, size_bytes = transfer_identity(payload)
        manifest = agent_gateway.manifest(job_id, agent)
        expected_context = {"agent": agent.token_id, "claim": agent.lease_claim, "job": job_id, "output": output_id,
                            "project": str(manifest.project_id)}
        if claims.get("digest") != digest or claims.get("size") != size_bytes or claims.get("ctx") != expected_context:
            raise HTTPException(status_code=401, detail="Blob Gateway ticket does not match Agent output")
        return agent_gateway.register_output(
            job_id,
            output_id,
            agent,
            expected_sha256=digest,
            expected_size=size_bytes,
        )

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/complete")
    def complete_agent_job(
        job_id: str,
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest
        from kraken_core.plugin_protocol import PluginJobManifestV2
        manifest = agent_gateway.manifest(job_id, agent)
        native_analysis = isinstance(manifest, AnalysisPartitionJobManifest)
        native_stitch = isinstance(manifest, PluginJobManifestV2) and manifest.operation == "layer.stitch.v1"
        imported = agent_gateway.finish(
            job_id, agent, failed=False,
            import_result=lambda: (backend.import_analysis_result(job_id, agent) if native_analysis
                                   else backend.import_stitch_agent_result(job_id, agent) if native_stitch
                                   else backend.import_agent_result(job_id, agent)),
        )
        return {"state": "completed", "imported": imported}

    @app.post(f"{API_PREFIX}/agent/jobs/{{job_id}}/fail")
    def fail_agent_job(
        job_id: str,
        payload: dict[str, Any],
        agent: Any = Depends(agent_identity),
    ) -> dict[str, Any]:
        if agent_gateway is None:
            raise HTTPException(status_code=503, detail="Server agent queue is not configured")
        error = str(payload.get("error", ""))
        from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest
        from kraken_core.plugin_protocol import PluginJobManifestV2
        manifest = agent_gateway.manifest(job_id, agent)
        native_analysis = isinstance(manifest, AnalysisPartitionJobManifest)
        native_stitch = isinstance(manifest, PluginJobManifestV2) and manifest.operation == "layer.stitch.v1"
        failed = agent_gateway.finish(
            job_id, agent, failed=True, error=error,
            import_result=lambda: (backend.fail_analysis_job(job_id, error) if native_analysis
                                   else backend.fail_stitch_agent_job(job_id, error) if native_stitch
                                   else backend.fail_agent_job(job_id, error)),
        )
        return {"state": "failed", "imported": failed}

    @app.websocket(f"{API_PREFIX}/ws")
    async def websocket(websocket: WebSocket) -> None:
        authorization = websocket.headers.get("authorization", "")
        token = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""

        def resolve_session() -> SessionPrincipal | None:
            if session_resolver is not None:
                return session_resolver(token)
            if account_store is not None:
                account = account_store.resolve_session(token)
                if account is None:
                    return None
                return SessionPrincipal(str(account.account_id), "local", token)
            if development and token:
                return SessionPrincipal(token, "development", token)
            return None

        session = await run_in_threadpool(resolve_session)
        if session is None:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        hub.register(websocket)
        await websocket.send_json({"type": "connected", "api_version": "v1"})
        try:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
                except TimeoutError:
                    still_valid = await run_in_threadpool(resolve_session) is not None
                    if not still_valid:
                        await websocket.close(code=4401)
                        return
                    continue
                kind = str(message.get("type", ""))
                if kind == "ping":
                    await websocket.send_json({"type": "pong"})
                elif kind == "subscribe":
                    project_id = message.get("project_id")
                    catalog = bool(message.get("catalog"))
                    if project_id is not None and not await run_in_threadpool(
                        has_project_access, str(project_id), session
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
                        catalog=catalog,
                        catalog_visible=(lambda project_id, subject=session: has_project_access(project_id, subject))
                        if catalog
                        else None,
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
