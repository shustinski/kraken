"""Authenticated loopback control API for Kraken Agent.

This deliberately uses the standard library so the agent can run in the
minimal desktop installation.  The protocol is versioned JSON over HTTP;
WebSocket progress can be added without changing the durable store contract.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kraken_core.plugin_protocol import PluginResultPublicationV2, parse_plugin_job

from .jobs import (
    AgentJob,
    AgentJobState,
    DuplicateCallbackError,
    DurableJobStore,
    JobStateError,
)


AGENT_API_VERSION = "v1"


def _job_payload(job: AgentJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "revision": job.revision,
        "error": job.error,
    }


@dataclass(slots=True)
class AgentControlServer:
    store: DurableJobStore
    token: str
    host: str = "127.0.0.1"
    port: int = 0

    @classmethod
    def create(cls, database: Path | str, *, token: str | None = None) -> "AgentControlServer":
        return cls(DurableJobStore(database), token or secrets.token_urlsafe(32))

    def build_http_server(self) -> ThreadingHTTPServer:
        store = self.store
        expected_token = self.token

        class Handler(BaseHTTPRequestHandler):
            server_version = "KrakenAgent/1"

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status.value)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)

            def _authorized(self) -> bool:
                supplied = self.headers.get("Authorization", "")
                return secrets.compare_digest(supplied, f"Bearer {expected_token}")

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 4 * 1024 * 1024:
                    raise ValueError("Invalid request body size")
                raw = self.rfile.read(length)
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object")
                return payload

            def do_GET(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send(HTTPStatus.UNAUTHORIZED, {"code": "agent.unauthorized"})
                    return
                path = urlparse(self.path).path
                if path == f"/api/{AGENT_API_VERSION}/health":
                    self._send(HTTPStatus.OK, {"status": "ok", "api_version": AGENT_API_VERSION})
                    return
                prefix = f"/api/{AGENT_API_VERSION}/jobs/"
                if path.startswith(prefix):
                    try:
                        job = store.get(path.removeprefix(prefix))
                    except KeyError:
                        self._send(HTTPStatus.NOT_FOUND, {"code": "agent.job_not_found"})
                        return
                    self._send(HTTPStatus.OK, _job_payload(job))
                    return
                self._send(HTTPStatus.NOT_FOUND, {"code": "agent.route_not_found"})

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send(HTTPStatus.UNAUTHORIZED, {"code": "agent.unauthorized"})
                    return
                path = urlparse(self.path).path
                jobs_path = f"/api/{AGENT_API_VERSION}/jobs"
                if path != jobs_path:
                    publication_suffix = "/publications"
                    prefix = jobs_path + "/"
                    if path.startswith(prefix) and path.endswith(publication_suffix):
                        job_id = path[len(prefix) : -len(publication_suffix)]
                        try:
                            payload = self._body()
                            publication = PluginResultPublicationV2.from_dict(payload)
                            if publication.job_id != job_id:
                                raise ValueError("Publication belongs to another job")
                            current = store.get(job_id)
                            if current.state not in {
                                AgentJobState.RUNNING,
                                AgentJobState.WAITING_FOR_USER,
                                AgentJobState.PARTIAL,
                            }:
                                raise JobStateError(
                                    f"Job cannot publish from {current.state.value}"
                                )
                            current, duplicate = store.record_result(
                                publication,
                                callback_key=f"publication:{publication.publication_id}",
                                expected_revision=current.revision,
                            )
                            if publication.final:
                                target = {
                                    "succeeded": AgentJobState.IMPORTING,
                                    "partial": AgentJobState.PARTIAL,
                                    "failed": AgentJobState.FAILED,
                                    "cancelled": AgentJobState.CANCELLED,
                                }[publication.outcome]
                                current = store.transition(
                                    current.job_id,
                                    target,
                                    expected_revision=current.revision,
                                )
                        except KeyError:
                            self._send(HTTPStatus.NOT_FOUND, {"code": "agent.job_not_found"})
                            return
                        except (
                            ValueError,
                            TypeError,
                            DuplicateCallbackError,
                            JobStateError,
                            json.JSONDecodeError,
                        ) as exc:
                            self._send(
                                HTTPStatus.CONFLICT,
                                {"code": "agent.invalid_publication", "detail": str(exc)},
                            )
                            return
                        response = _job_payload(current)
                        response["duplicate"] = duplicate
                        self._send(HTTPStatus.OK, response)
                        return
                    suffixes = {
                        "/cancel": AgentJobState.CANCELLED,
                        "/confirm-partial": AgentJobState.IMPORTING,
                        "/complete-import": AgentJobState.SUCCEEDED,
                    }
                    for suffix, target in suffixes.items():
                        prefix = jobs_path + "/"
                        if path.startswith(prefix) and path.endswith(suffix):
                            job_id = path[len(prefix) : -len(suffix)]
                            try:
                                payload = self._body()
                                job = store.transition(
                                    job_id,
                                    target,
                                    expected_revision=int(payload.get("expected_revision", -1)),
                                )
                            except KeyError:
                                self._send(HTTPStatus.NOT_FOUND, {"code": "agent.job_not_found"})
                                return
                            except (ValueError, JobStateError, json.JSONDecodeError) as exc:
                                self._send(
                                    HTTPStatus.CONFLICT,
                                    {"code": "agent.invalid_transition", "detail": str(exc)},
                                )
                                return
                            self._send(HTTPStatus.OK, _job_payload(job))
                            return
                    self._send(HTTPStatus.NOT_FOUND, {"code": "agent.route_not_found"})
                    return
                try:
                    manifest = parse_plugin_job(self._body())
                    job = store.enqueue(manifest)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._send(HTTPStatus.BAD_REQUEST, {"code": "agent.invalid_manifest", "detail": str(exc)})
                    return
                self._send(HTTPStatus.ACCEPTED, _job_payload(job))

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(server.server_address[1])
        return server


__all__ = ["AGENT_API_VERSION", "AgentControlServer"]
