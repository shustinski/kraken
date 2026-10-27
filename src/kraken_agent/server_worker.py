"""Polling worker for the authenticated Kraken Server agent protocol."""

from __future__ import annotations

import hashlib
from dataclasses import replace
import json
import logging
import mimetypes
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import unquote, urlparse
from pathlib import Path
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from kraken_core.analysis_run_protocol import ANALYSIS_PARTITION_JOB_SCHEMA
from kraken_core.model_input import PinnedModelInput
from kraken_core.safe_files import (
    contained_path,
    make_contained_directories,
    open_exclusive_write,
    open_regular_read,
)
from kraken_core.plugin_protocol import (
    PLUGIN_JOB_SCHEMA,
    PLUGIN_JOB_SCHEMA_V2,
    PluginFrameInput,
    PluginJobManifest,
)
from kraken_manager.domain.project import FrameCoordinate
from kraken_manager.domain.workflows import PluginJobManifestV1
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model

from .jobs import DurableJobStore, StagingWorkspace
from .runner import PluginRegistry, SubprocessPluginRunner

LOGGER = logging.getLogger(__name__)


class ServerAgentWorker:
    def __init__(
        self,
        *,
        server_url: str,
        token: str,
        data_dir: Path,
        registry: PluginRegistry,
        lease_seconds: int = 60,
        headers: dict[str, str] | None = None,
    ) -> None:
        parsed_server = urlparse(server_url)
        if parsed_server.scheme != "https" and parsed_server.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ValueError("Kraken Agent requires HTTPS except for a loopback server")
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.headers = dict(headers or {})
        self.data_dir = data_dir.resolve()
        self.registry = registry
        self.lease_seconds = max(30, min(int(lease_seconds), 300))
        self.store = DurableJobStore(self.data_dir / "server-jobs.sqlite3")
        self.runner = SubprocessPluginRunner(
            self.store, self.data_dir / "staging", registry, environment={"KRAKEN_ASSET_TOKEN": token}
        )
        self._stop = threading.Event()
        self._lease_lock = threading.Lock()

    def stop(self) -> None:
        self._stop.set()
        self.runner.stop()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None, *, timeout=30) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.server_url + path,
            data=body,
            method=method,
            headers={
                **self.headers,
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        return {} if not raw else json.loads(raw.decode("utf-8"))

    def _download(
        self,
        path: str,
        destination: Path,
        *,
        expected_sha256: str,
        staging_root: Path | None = None,
    ) -> str:
        destination = destination.absolute()
        root = staging_root if staging_root is not None else Path(destination.anchor)
        parts = destination.relative_to(root).parts
        if not parts or any(part in {"..", "."} for part in parts):
            raise ValueError("Download target escapes staging")
        make_contained_directories(root, parts[:-1])
        destination = contained_path(root, parts)
        try:
            transfer = self._request("GET", path + "/download-ticket")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            transfer = {"mode": "proxy"}
        mode = str(transfer.get("mode", "proxy"))
        url = str(transfer["url"]) if mode == "gateway" else self.server_url + path
        token = str(transfer["token"]) if mode == "gateway" else self.token
        request = urllib.request.Request(
            url,
            headers={**({} if mode == "gateway" else self.headers), "Authorization": f"Bearer {token}"},
        )
        digest = hashlib.sha256()
        created_identity: tuple[int, int] | None = None

        def remove_partial() -> None:
            if created_identity is None:
                return
            # A failed exclusive create owns nothing. Also refuse cleanup if a
            # plugin replaced an ancestor or the file after we created it.
            try:
                checked = contained_path(root, parts)
                current = checked.lstat()
                if (current.st_dev, current.st_ino) == created_identity:
                    checked.unlink()
            except OSError, ValueError:
                LOGGER.warning("Could not safely remove partial input %s", destination)

        try:
            with urllib.request.urlopen(request, timeout=60) as response, open_exclusive_write(destination) as output:
                opened = os.fstat(output.fileno())
                created_identity = (opened.st_dev, opened.st_ino)
                contained_path(root, parts)
                media_type = (
                    str(transfer.get("media_type") or response.headers.get_content_type())
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise ValueError(f"Downloaded input hash mismatch: {destination.name}")
                contained_path(root, parts)
        except Exception:  # noqa: BLE001 - remove a partially downloaded untrusted blob
            remove_partial()
            raise
        if media_type == "application/octet-stream":
            # Older gateways describe raw blob bytes rather than artifact types.
            # Frame staging preserves the original extension; hashes are checked
            # above and the plugin still validates the actual image contents.
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".tif": "image/tiff",
                ".tiff": "image/tiff",
                ".zip": "application/zip",
            }.get(destination.suffix.casefold(), media_type)
        return media_type

    def _upload(self, path: str, source: Path, *, staging_root: Path | None = None) -> None:
        source = source.absolute()
        root = staging_root if staging_root is not None else Path(source.anchor)
        with open_regular_read(source, root=root) as stream:
            self._upload_stream(path, stream)

    def _upload_stream(self, path: str, stream: Any) -> None:
        parsed = urllib.parse.urlsplit(path)
        query = urllib.parse.parse_qs(parsed.query)
        digest = str(query.get("sha256", [""])[0])
        endpoint = parsed.path
        try:
            transfer = self._request(
                "POST",
                endpoint + "/upload-ticket",
                {"sha256": digest, "size_bytes": os.fstat(stream.fileno()).st_size},
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            transfer = {"mode": "proxy"}
        mode = str(transfer.get("mode", "proxy"))
        request = urllib.request.Request(
            str(transfer["url"]) if mode == "gateway" else self.server_url + path,
            data=stream,
            method="PUT" if mode == "gateway" else "POST",
            headers={
                **({} if mode == "gateway" else self.headers),
                "Authorization": f"Bearer {transfer['token'] if mode == 'gateway' else self.token}",
                "Accept": "application/json",
                "Content-Type": "application/octet-stream",
            },
        )
        request.add_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
        if mode == "gateway":
            self._request(
                "POST",
                endpoint + "/upload-complete",
                {
                    "sha256": digest,
                    "size_bytes": os.fstat(stream.fileno()).st_size,
                    "upload_token": str(transfer["token"]),
                },
            )

    def _transport_manifest(self, manifest: PluginJobManifestV1, workspace: StagingWorkspace) -> PluginJobManifest:
        stream_inputs = manifest.parameters.get("input_transport") == "server"
        if stream_inputs and manifest.capability not in {
            "frames.vectorize.v1",
            "frames.dataset.prepare.v1",
            "dataset.model.train.v1",
            "frames.binary-segment.v1",
            "frames.binary-segment.v2",
            "frames.confidence.v1",
        }:
            raise ValueError("This plugin operation does not support server input streams")
        coordinates = {
            str(FrameCoordinate(item.x_start, item.y).frame_id(manifest.project_id)): (item.x_start, item.y)
            for item in manifest.selection.row_ranges
            if item.x_start == item.x_end
        }
        inputs: list[PluginFrameInput] = []
        for item in manifest.inputs:
            coordinate = coordinates.get(str(item.frame_id))
            if coordinate is None:
                coordinate = next(
                    (
                        (candidate.x, candidate.y)
                        for candidate in manifest.selection.iter_coordinates()
                        if str(candidate.frame_id(manifest.project_id)) == str(item.frame_id)
                    ),
                    None,
                )
            if coordinate is None:
                raise ValueError(f"Cannot map input frame {item.frame_id} to a project coordinate")
            target = workspace.path / item.relative_path
            if stream_inputs:
                if item.external_uri is not None:
                    raise ValueError("Server stream jobs require managed server inputs")
                media_types = manifest.parameters.get("input_media_types", {})
                media_type = str(media_types.get(str(item.artifact_version_id)) or "application/octet-stream")
                if media_type == "application/octet-stream":
                    media_type = mimetypes.guess_type(item.relative_path)[0] or media_type
                if media_type == "application/octet-stream":
                    ticket = self._request(
                        "GET", f"/api/v1/agent/jobs/{manifest.job_id}/inputs/{item.artifact_version_id}/download-ticket"
                    )
                    media_type = str(ticket.get("media_type") or media_type)
            elif item.external_uri is None:
                media_type = self._download(
                    f"/api/v1/agent/jobs/{manifest.job_id}/inputs/{item.artifact_version_id}",
                    target,
                    expected_sha256=item.sha256,
                    staging_root=workspace.path,
                )
            else:
                parsed = urlparse(item.external_uri)
                if parsed.scheme != "file":
                    raise ValueError(f"Unsupported external input URI: {item.external_uri}")
                source = Path(unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path))
                if os.name == "nt" and parsed.netloc:
                    source = Path(f"//{parsed.netloc}/{unquote(parsed.path.lstrip('/'))}")
                try:
                    source = source.resolve(strict=True)
                    workspace.stage_file(source, item.relative_path, expected_sha256=item.sha256)
                except OSError as exc:
                    raise FileNotFoundError(
                        f"External input is unavailable to the agent service account: {source}"
                    ) from exc
                media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            inputs.append(
                PluginFrameInput(
                    frame_id=str(item.frame_id),
                    x=coordinate[0],
                    y=coordinate[1],
                    artifact_version_id=str(item.artifact_version_id),
                    sha256=item.sha256,
                    media_type=media_type,
                    relative_path=item.relative_path,
                )
            )
        parameters = encode_model(manifest.parameters)
        parameters.pop("input_media_types", None)
        if stream_inputs:
            parameters["server_url"] = self.server_url
        if parameters.get("execution_host") == "desktop":
            from kraken_manager.infrastructure.plugin.agent_gateway import _stage_model

            parameters = _stage_model(workspace, parameters)
        try:
            created_at = getattr(self.store.get(str(manifest.job_id)).manifest, "created_at", "")
        except KeyError:
            created_at = ""  # Domain V1 has no transport timestamp; do not invent one on each handoff.
        return PluginJobManifest(
            job_id=str(manifest.job_id),
            operation=manifest.capability,
            project_id=str(manifest.project_id),
            layer_id=str(manifest.layer_id),
            actor_id=str(manifest.actor_principal_id),
            target_representation_id=str(manifest.target_representation_id),
            inputs=tuple(inputs),
            parameters=parameters,
            protocol_version=manifest.protocol_version,
            created_at=created_at,
        )

    def _stage_model_input(self, transport, workspace):
        model = PinnedModelInput.from_parameters(getattr(transport, "parameters", {}))
        if model is None:
            return transport
        destination = workspace.resolve(model.relative_path)
        try:
            if workspace.digest(model.relative_path) != model.sha256:
                raise ValueError("Existing staged model does not match the pinned artifact")
            media_type = "application/octet-stream"
        except FileNotFoundError:
            media_type = self._download(
                f"/api/v1/agent/jobs/{transport.job_id}/inputs/{model.version_id}",
                destination,
                expected_sha256=model.sha256,
                staging_root=workspace.path,
            )
        parameters = dict(transport.parameters, model_relative_path=model.relative_path)
        if schema := getattr(transport, "schema", None):
            if schema == PLUGIN_JOB_SCHEMA_V2:
                from kraken_core.plugin_protocol import PluginAsset

                existing = [item for item in transport.inputs if item.asset_id == model.version_id]
                if existing:
                    if (
                        len(existing) != 1
                        or existing[0].role != "model"
                        or existing[0].scope != "layer"
                        or existing[0].relative_path != model.relative_path
                        or existing[0].sha256 != model.sha256
                    ):
                        raise ValueError("Conflicting V2 model asset binding")
                    return replace(transport, parameters=parameters)
                asset = PluginAsset(
                    asset_id=model.version_id,
                    role="model",
                    scope="layer",
                    relative_path=model.relative_path,
                    sha256=model.sha256,
                    media_type=media_type,
                )
                return replace(transport, parameters=parameters, inputs=(*transport.inputs, asset))
        return replace(transport, parameters=parameters)

    def _cancel_lost_lease(self, job_id: str, lost: threading.Event) -> None:
        from kraken_agent.jobs import AgentJobState, JobStateError

        lost.set()
        try:
            job = self.store.get(job_id)
            self.store.transition(job_id, AgentJobState.CANCELLED, expected_revision=job.revision)
        except KeyError:
            # Staging can precede the local queue entry. The caller observes
            # lost again after enqueue; keep the worker usable.
            pass
        except JobStateError:
            LOGGER.info("Job %s already finished when its lease ended", job_id)

    def _heartbeat(self, job_id: str, done: threading.Event, lost: threading.Event, deadline: float) -> None:
        from .telemetry import sample_host

        while not done.is_set():
            if monotonic() >= deadline:
                self._cancel_lost_lease(job_id, lost)
                return
            try:
                telemetry = sample_host()
                sent_at = monotonic()
                if sent_at >= deadline:
                    self._cancel_lost_lease(job_id, lost)
                    return
                self._request(
                    "POST",
                    f"/api/v1/agent/jobs/{job_id}/heartbeat",
                    {
                        "lease_seconds": self.lease_seconds,
                        "telemetry": telemetry,
                    },
                    timeout=min(5, deadline - sent_at),
                )
                # Reserve from send time, not response time, so latency never
                # extends the worker's local authority past its renewal budget.
                deadline = sent_at + self.lease_seconds
            except urllib.error.HTTPError as exc:
                if exc.code not in {403, 409}:
                    LOGGER.exception("Server heartbeat failed for job %s", job_id)
                else:
                    self._cancel_lost_lease(job_id, lost)
                    return
            except OSError, ValueError:
                LOGGER.exception("Server heartbeat failed for job %s", job_id)
            if done.wait(min(max(10, self.lease_seconds // 3), max(0, deadline - monotonic()))):
                return

    def run_once(self) -> bool:
        lease = self._request(
            "POST",
            "/api/v1/agent/lease",
            {
                "capabilities": sorted(self.registry.operations()),
                "lease_seconds": self.lease_seconds,
            },
        ).get("job")
        if not isinstance(lease, dict):
            return False
        try:
            self.run_leased_job(lease)
        except Exception:
            LOGGER.exception("Server job %s failed", lease.get("job_id"))
        return True

    def run_leased_job(self, lease: dict[str, Any]) -> dict[str, Any]:
        """Execute a specific lease, including staging and result publication."""
        with self._lease_lock:
            if self._stop.is_set():
                raise RuntimeError("Agent worker has stopped")
            return self._run_leased_job(lease)

    def _run_leased_job(self, lease: dict[str, Any]) -> dict[str, Any]:
        job_id = str(lease["job_id"])
        claim = lease.get("lease_claim")
        if not isinstance(claim, str) or not claim:
            raise RuntimeError("Server lease has no claim token: update the Server and acquire a new lease")
        remaining = float(self.lease_seconds)
        if lease.get("lease_until") is not None:
            expires = datetime.fromisoformat(str(lease["lease_until"]))
            if expires.tzinfo is None:
                raise ValueError("Server lease expiry requires a timezone")
            remaining = max(0, min(remaining, (expires - datetime.now(UTC)).total_seconds()))
        deadline = monotonic() + remaining
        attempt_key = hashlib.sha256(claim.encode()).hexdigest()
        previous_staging = self.runner.staging_root
        self.runner.staging_root = self.data_dir / "staging" / attempt_key
        previous_claim = self.headers.get("X-Kraken-Lease-Claim")
        previous_environment_claim = self.runner.environment.get("KRAKEN_ASSET_LEASE_CLAIM")
        self.headers["X-Kraken-Lease-Claim"] = claim
        self.runner.environment["KRAKEN_ASSET_LEASE_CLAIM"] = claim
        self.runner.environment["KRAKEN_ASSET_TOKEN"] = self.token
        done = threading.Event()
        lost = threading.Event()
        heartbeat = threading.Thread(target=self._heartbeat, args=(job_id, done, lost, deadline), daemon=True)
        try:
            heartbeat.start()
            raw_manifest = lease["manifest"]
            schema = str(raw_manifest.get("schema", "")) if isinstance(raw_manifest, dict) else ""
            if schema in {ANALYSIS_PARTITION_JOB_SCHEMA, PLUGIN_JOB_SCHEMA_V2, PLUGIN_JOB_SCHEMA}:
                from kraken_agent.protocols import parse_manifest_payload

                transport = parse_manifest_payload(raw_manifest)
                workspace = StagingWorkspace(self.runner.staging_root, job_id)
                workspace.create()
                if schema == ANALYSIS_PARTITION_JOB_SCHEMA:
                    for frame in transport.frames:
                        for artifact in frame.artifacts:
                            destination = workspace.resolve(artifact.relative_path)
                            try:
                                if workspace.digest(artifact.relative_path) != artifact.sha256:
                                    raise ValueError("Existing analysis input does not match the pinned version")
                            except FileNotFoundError:
                                self._download(
                                    f"/api/v1/agent/jobs/{job_id}/inputs/{artifact.artifact_version_id}",
                                    destination,
                                    expected_sha256=artifact.sha256,
                                    staging_root=workspace.path,
                                )
                elif schema == PLUGIN_JOB_SCHEMA_V2:
                    for asset in transport.inputs:
                        destination = workspace.resolve(asset.relative_path)
                        try:
                            if workspace.digest(asset.relative_path) != asset.sha256:
                                raise ValueError("Existing V2 input does not match the pinned version")
                        except FileNotFoundError:
                            self._download(f"/api/v1/agent/jobs/{job_id}/inputs/{asset.asset_id}", destination,
                                expected_sha256=asset.sha256, staging_root=workspace.path)
            else:
                manifest = decode_model(PluginJobManifestV1, raw_manifest)
                workspace = StagingWorkspace(self.runner.staging_root, job_id)
                workspace.create()
                transport = self._transport_manifest(manifest, workspace)
            transport = self._stage_model_input(transport, workspace)
            workspace.write_manifest(transport)
            self.store.enqueue(transport, attempt_key=attempt_key)
            if lost.is_set():
                from kraken_agent.jobs import AgentJobState, JobStateError

                job = self.store.get(job_id)
                try:
                    self.store.transition(job_id, AgentJobState.CANCELLED, expected_revision=job.revision)
                except JobStateError:
                    pass
                return {"imported": {"job": {"state": "cancelled"}}}
            self.runner.run_once(job_id=job_id, cancel_event=lost)
            job = self.store.get(job_id)
            if lost.is_set() or job.state.value == "cancelled":
                return {"imported": {"job": {"state": "cancelled"}}}
            if job.result is None:
                raise RuntimeError(job.error or "Plugin finished without a result")
            result = job.result
            from kraken_core.analysis_run_protocol import AnalysisPartitionResultManifest

            if isinstance(result, AnalysisPartitionResultManifest):
                if result.rendered_map is not None:
                    self._upload(
                        f"/api/v1/agent/jobs/{job_id}/outputs/analysis-map?"
                        + urllib.parse.urlencode({"sha256": result.rendered_map.sha256}),
                        workspace.resolve(result.rendered_map.relative_path), staging_root=workspace.path,
                    )
                if result.bundle is not None:
                    self._upload(
                        f"/api/v1/agent/jobs/{job_id}/outputs/analysis-bundle?"
                        + urllib.parse.urlencode({"sha256": result.bundle.sha256}),
                        workspace.resolve(result.bundle.relative_path),
                        staging_root=workspace.path,
                    )
                self._request("POST", f"/api/v1/agent/jobs/{job_id}/publications", result.to_payload())
                return self._request("POST", f"/api/v1/agent/jobs/{job_id}/complete", {})
            outputs = getattr(result, "outputs", getattr(result, "results", ()))
            for output in outputs:
                relative_path = getattr(output, "relative_path", None)
                sha256 = getattr(output, "sha256", None)
                output_id = getattr(output, "output_id", getattr(output, "asset_id", None))
                if relative_path and sha256 and output_id:
                    self._upload(
                        f"/api/v1/agent/jobs/{job_id}/outputs/{output_id}?"
                        + urllib.parse.urlencode({"sha256": sha256}),
                        workspace.path / relative_path,
                        staging_root=workspace.path,
                    )
            self._request(
                "POST",
                f"/api/v1/agent/jobs/{job_id}/publications",
                result.to_dict(),
            )
            return self._request("POST", f"/api/v1/agent/jobs/{job_id}/complete", {})
        except Exception as exc:  # noqa: BLE001 - plugin and transport failures must fail the lease
            try:
                self._request(
                    "POST",
                    f"/api/v1/agent/jobs/{job_id}/fail",
                    {"error": str(exc)[:10_000]},
                )
            except Exception:  # noqa: BLE001 - preserve the original failure after best-effort reporting
                LOGGER.exception("Could not report failure for server job %s", job_id)
            raise
        finally:
            done.set()
            if heartbeat.is_alive():
                heartbeat.join(timeout=7)
                if heartbeat.is_alive():
                    LOGGER.error("Heartbeat did not stop; retire this worker before another lease")
                    self.stop()
            self.runner.staging_root = previous_staging
            if previous_environment_claim is None:
                self.runner.environment.pop("KRAKEN_ASSET_LEASE_CLAIM", None)
            else:
                self.runner.environment["KRAKEN_ASSET_LEASE_CLAIM"] = previous_environment_claim
            if previous_claim is None:
                self.headers.pop("X-Kraken-Lease-Claim", None)
            else:
                self.headers["X-Kraken-Lease-Claim"] = previous_claim

    def run_forever(self, *, idle_seconds: float = 2.0) -> None:
        self.store.recover_interrupted()
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError:
                worked = False
            if not worked:
                self._stop.wait(idle_seconds)


__all__ = ["ServerAgentWorker"]
