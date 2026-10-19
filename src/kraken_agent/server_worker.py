"""Polling worker for the authenticated Kraken Server agent protocol."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import unquote, urlparse
from pathlib import Path
from typing import Any

from kraken_core.analysis_run_protocol import ANALYSIS_PARTITION_JOB_SCHEMA
from kraken_core.plugin_protocol import (
    PLUGIN_JOB_SCHEMA,
    PLUGIN_JOB_SCHEMA_V2,
    PluginFrameInput,
    PluginJobManifest,
)
from kraken_manager.domain.project import FrameCoordinate
from kraken_manager.domain.workflows import PluginJobManifestV1
from kraken_manager.infrastructure.filesystem._codec import decode_model

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

    def stop(self) -> None:
        self._stop.set()
        self.runner.stop()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
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
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        return {} if not raw else json.loads(raw.decode("utf-8"))

    def _download(self, path: str, destination: Path, *, expected_sha256: str) -> str:
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
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as output:
                media_type = (
                    str(transfer.get("media_type") or response.headers.get_content_type())
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        except Exception:  # noqa: BLE001 - remove a partially downloaded untrusted blob
            destination.unlink(missing_ok=True)
            raise
        if digest.hexdigest() != expected_sha256:
            destination.unlink(missing_ok=True)
            raise ValueError(f"Downloaded input hash mismatch: {destination.name}")
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

    def _upload(self, path: str, source: Path) -> None:
        parsed = urllib.parse.urlsplit(path)
        query = urllib.parse.parse_qs(parsed.query)
        digest = str(query.get("sha256", [""])[0])
        endpoint = parsed.path
        try:
            transfer = self._request(
                "POST",
                endpoint + "/upload-ticket",
                {"sha256": digest, "size_bytes": source.stat().st_size},
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            transfer = {"mode": "proxy"}
        mode = str(transfer.get("mode", "proxy"))
        with source.open("rb") as stream:
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
            request.add_header("Content-Length", str(source.stat().st_size))
            with urllib.request.urlopen(request, timeout=120) as response:
                response.read()
        if mode == "gateway":
            self._request(
                "POST",
                endpoint + "/upload-complete",
                {
                    "sha256": digest,
                    "size_bytes": source.stat().st_size,
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
                )
            else:
                parsed = urlparse(item.external_uri)
                if parsed.scheme != "file":
                    raise ValueError(
                        f"Unsupported external input URI: {item.external_uri}"
                    )
                source = Path(
                    unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path)
                )
                if os.name == "nt" and parsed.netloc:
                    source = Path(
                        f"//{parsed.netloc}/{unquote(parsed.path.lstrip('/'))}"
                    )
                try:
                    source = source.resolve(strict=True)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                except OSError as exc:
                    raise FileNotFoundError(
                        f"External input is unavailable to the agent service account: {source}"
                    ) from exc
                digest = hashlib.sha256()
                with target.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != item.sha256:
                    target.unlink(missing_ok=True)
                    raise ValueError(f"External input hash mismatch: {source}")
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
        parameters = dict(manifest.parameters)
        parameters.pop("input_media_types", None)
        if stream_inputs:
            parameters["server_url"] = self.server_url
        if parameters.get("execution_host") == "desktop":
            from kraken_manager.infrastructure.plugin.agent_gateway import _stage_model

            parameters = _stage_model(workspace, parameters)
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
        )

    def _heartbeat(self, job_id: str, done: threading.Event) -> None:
        while not done.wait(max(10, self.lease_seconds // 3)):
            try:
                self._request("POST", f"/api/v1/agent/jobs/{job_id}/heartbeat", {"lease_seconds": self.lease_seconds})
            except urllib.error.HTTPError as exc:
                if exc.code not in {403, 409}:
                    LOGGER.exception("Server heartbeat failed for job %s", job_id)
                    continue
                from kraken_agent.jobs import AgentJobState, JobStateError

                try:
                    job = self.store.get(job_id)
                    self.store.transition(job_id, AgentJobState.CANCELLED, expected_revision=job.revision)
                except KeyError:
                    self.runner.stop()
                except JobStateError:
                    LOGGER.info("Job %s already finished when its lease ended", job_id)
                return
            except OSError, ValueError:
                LOGGER.exception("Server heartbeat failed for job %s", job_id)

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
        job_id = str(lease["job_id"])
        self.runner.environment["KRAKEN_ASSET_TOKEN"] = self.token
        done = threading.Event()
        heartbeat = threading.Thread(target=self._heartbeat, args=(job_id, done), daemon=True)
        try:
            heartbeat.start()
            raw_manifest = lease["manifest"]
            schema = str(raw_manifest.get("schema", "")) if isinstance(raw_manifest, dict) else ""
            if schema in {ANALYSIS_PARTITION_JOB_SCHEMA, PLUGIN_JOB_SCHEMA_V2, PLUGIN_JOB_SCHEMA}:
                from kraken_agent.protocols import parse_manifest_payload

                transport = parse_manifest_payload(raw_manifest)
                workspace = StagingWorkspace(self.data_dir / "staging", job_id)
                workspace.create()
                workspace.write_manifest(transport)
                self.store.enqueue(transport)
            else:
                manifest = decode_model(PluginJobManifestV1, raw_manifest)
                workspace = StagingWorkspace(self.data_dir / "staging", job_id)
                workspace.create()
                transport = self._transport_manifest(manifest, workspace)
                workspace.write_manifest(transport)
                self.store.enqueue(transport)
            self.runner.run_once()
            job = self.store.get(job_id)
            if job.state.value == "cancelled":
                return {"imported": {"job": {"state": "cancelled"}}}
            if job.result is None:
                raise RuntimeError(job.error or "Plugin finished without a result")
            result = job.result
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
                heartbeat.join(timeout=5)

    def run_forever(self, *, idle_seconds: float = 2.0) -> None:
        self.store.recover_interrupted()
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
                worked = False
            if not worked:
                self._stop.wait(idle_seconds)


__all__ = ["ServerAgentWorker"]
