"""HTTP/WebSocket client for shared projects hosted by kraken-server."""

from __future__ import annotations

import json
import logging
import os
import threading
import hashlib
import mimetypes
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import urllib.parse
from urllib.parse import unquote, urlparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from uuid import uuid4
import zipfile

from kraken_manager.application.ports import StorageProfile
from kraken_manager.application.dto import ReviewReturnCommitResult, ReviewReturnPlan
from kraken_manager.infrastructure.filesystem._codec import decode_model
from kraken_manager.domain.artifacts import ArtifactScope, ArtifactSeries, ArtifactVersion, NoteRevision
from kraken_manager.domain.workflows import (
    PluginJob,
    ReviewBatch,
)
from kraken_manager.infrastructure.review.manifest import manifest_from_json
from kraken_manager.infrastructure.reports import (
    ReportMetrics,
    ReportSeries,
)
from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.identity import (
    Permission,
    Performer,
    Principal,
    PrincipalProvider,
    ProjectRole,
    SystemRole,
    permissions_for_roles,
)
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
    StructureState,
)

from .workspace_service import (
    REMOTE_STORAGE_PROFILE,
    ProjectEventWake,
    project_from_server_dict,
)

LOGGER = logging.getLogger(__name__)


def _review_archive(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for path in sorted(source.rglob("*")):
            metadata = path.lstat()
            attributes = int(getattr(metadata, "st_file_attributes", 0))
            if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400:
                raise ValueError(f"Review package contains a link: {path}")
            if stat.S_ISREG(metadata.st_mode):
                archive.write(path, path.relative_to(source).as_posix())


def _extract_review_archive(archive: Path, destination: Path) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        total = 0
        with zipfile.ZipFile(archive) as package:
            entries = package.infolist()
            if len(entries) > 250_000:
                raise ValueError("Review package has too many entries")
            for entry in entries:
                parts = tuple(part for part in entry.filename.split("/") if part)
                if (
                    not parts
                    or entry.filename.startswith(("/", "\\"))
                    or "\\" in entry.filename
                    or any(part in {".", ".."} or ":" in part for part in parts)
                    or stat.S_ISLNK(entry.external_attr >> 16)
                ):
                    raise ValueError(f"Unsafe review archive member: {entry.filename}")
                total += entry.file_size
                if entry.file_size > 2 * 1024**3 or total > 16 * 1024**3:
                    raise ValueError("Review package exceeds the size limit")
                if (
                    entry.file_size >= 1024 * 1024
                    and entry.file_size / max(1, entry.compress_size) > 200
                ):
                    raise ValueError("Review package compression ratio is unsafe")
                target = staging.joinpath(*parts)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                observed = 0
                with package.open(entry) as input_stream, target.open("xb") as output:
                    while chunk := input_stream.read(1024 * 1024):
                        observed += len(chunk)
                        if observed > entry.file_size:
                            raise ValueError("Review archive member expanded unexpectedly")
                        output.write(chunk)
                if observed != entry.file_size:
                    raise ValueError("Review archive member size differs")
        if destination.exists():
            raise FileExistsError(destination)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


class RemoteServerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        problem: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.problem = dict(problem or {})
        self.code = str(self.problem.get("code", ""))


@dataclass(slots=True)
class RemoteAuth:
    access_token: str
    principal: Principal


class RemoteHttpClient:
    """Minimal stdlib HTTP client for `/api/v1`."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        token: str,
        payload: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        body = None if payload is None else json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = detail
            parsed: Mapping[str, object] | None = None
            try:
                candidate = json.loads(detail)
                parsed = candidate if isinstance(candidate, Mapping) else None
                if parsed is not None:
                    message = str(parsed.get("detail") or parsed.get("title") or detail)
            except (json.JSONDecodeError, TypeError, AttributeError):
                message = detail
            raise RemoteServerError(
                message, status=exc.code, body=detail, problem=parsed
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteServerError(f"Cannot reach Kraken Server at {self.base_url}: {exc.reason}") from exc

    def upload(
        self,
        path: str,
        *,
        token: str,
        source: Path,
        headers: Mapping[str, str],
    ) -> Any:
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/octet-stream",
            **dict(headers),
        }
        try:
            with source.open("rb") as stream:
                request = urllib.request.Request(
                    f"{self.base_url}{path}",
                    data=stream,
                    headers=request_headers,
                    method="POST",
                )
                request.add_header("Content-Length", str(source.stat().st_size))
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
            return None if not raw else json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            parsed: Mapping[str, object] | None = None
            message = detail
            try:
                candidate = json.loads(detail)
                parsed = candidate if isinstance(candidate, Mapping) else None
                if parsed is not None:
                    message = str(parsed.get("detail") or parsed.get("title") or detail)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            raise RemoteServerError(
                message, status=exc.code, body=detail, problem=parsed
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteServerError(f"Cannot reach Kraken Server at {self.base_url}: {exc.reason}") from exc

    def download(
        self,
        path: str,
        *,
        token: str,
        destination: Path,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
    ) -> Path:
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/octet-stream",
            **dict(headers or {}),
        }
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers=request_headers,
            method=method,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response, destination.open("xb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        except urllib.error.HTTPError as exc:
            destination.unlink(missing_ok=True)
            detail = exc.read().decode("utf-8", errors="replace")
            parsed: Mapping[str, object] | None = None
            message = detail
            try:
                candidate = json.loads(detail)
                parsed = candidate if isinstance(candidate, Mapping) else None
                if parsed is not None:
                    message = str(parsed.get("detail") or parsed.get("title") or detail)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            raise RemoteServerError(
                message, status=exc.code, body=detail, problem=parsed
            ) from exc
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return destination


class RemoteSyncClient:
    """Background WebSocket subscriber that emits ProjectEventWake callbacks."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        on_event: Callable[[ProjectEventWake], None],
        on_catalog: Callable[[], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._ws_url = self._to_ws_url(base_url)
        self._token = token
        self._on_event = on_event
        self._on_catalog = on_catalog
        self._on_status = on_status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._ws: Any = None
        self._subscribed: set[str] = set()
        self._catalog = False

    @staticmethod
    def _to_ws_url(base_url: str) -> str:
        if base_url.startswith("https://"):
            return "wss://" + base_url.removeprefix("https://").rstrip("/") + "/api/v1/ws"
        if base_url.startswith("http://"):
            return "ws://" + base_url.removeprefix("http://").rstrip("/") + "/api/v1/ws"
        return base_url.rstrip("/") + "/api/v1/ws"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._status("reconnecting")
        self._thread = threading.Thread(target=self._run, name="kraken-remote-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except (OSError, RuntimeError):
                LOGGER.exception("Failed to close Kraken remote WebSocket")
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._status("offline")

    def _status(self, value: str) -> None:
        if self._on_status is not None:
            self._on_status(value)

    def subscribe_project(self, project_id: str) -> None:
        self._subscribed.add(str(project_id))
        self._send({"type": "subscribe", "project_id": str(project_id)})

    def unsubscribe_project(self, project_id: str) -> None:
        self._subscribed.discard(str(project_id))
        self._send({"type": "unsubscribe", "project_id": str(project_id)})

    def subscribe_catalog(self) -> None:
        self._catalog = True
        self._send({"type": "subscribe", "catalog": True})

    def _send(self, message: Mapping[str, object]) -> None:
        ws = self._ws
        if ws is None:
            return
        with self._send_lock:
            try:
                ws.send(json.dumps(dict(message)))
            except (OSError, RuntimeError):
                LOGGER.exception("Failed to send Kraken remote subscription")

    def _run(self) -> None:
        try:
            from websocket import WebSocketApp  # type: ignore
        except ImportError:
            self._status("offline")
            return

        def on_message(_ws: Any, message: str) -> None:
            try:
                payload = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
            kind = str(payload.get("type", ""))
            if kind == "project_event":
                wake = ProjectEventWake(
                    project_id=str(payload.get("project_id", "")),
                    event_type=str(payload.get("event_type", "")),
                    event_id=str(payload.get("event_id", "")),
                    position=(
                        None
                        if payload.get("position") is None
                        else int(payload["position"])
                    ),
                    revision=(
                        None
                        if payload.get("revision") is None
                        else int(payload["revision"])
                    ),
                    stream_id=str(payload.get("stream_id", "")),
                    entity_kind=str(payload.get("entity_kind", "")),
                    entity_id=str(payload.get("entity_id", "")),
                )
                if wake.project_id:
                    self._on_event(wake)
            elif kind == "catalog_changed" and self._on_catalog is not None:
                self._on_catalog()

        def on_open(ws: Any) -> None:
            self._ws = ws
            if self._catalog:
                self._send({"type": "subscribe", "catalog": True})
            for project_id in list(self._subscribed):
                self._send({"type": "subscribe", "project_id": project_id})
            self._status("synchronized")
            # A reconnect may have missed wake messages. REST snapshots are
            # authoritative, so force a catalog/project refresh on every open.
            if self._on_catalog is not None:
                self._on_catalog()

        def on_close(_ws: Any, *_args: object) -> None:
            self._ws = None
            if not self._stop.is_set():
                self._status("reconnecting")

        while not self._stop.is_set():
            app = WebSocketApp(
                self._ws_url,
                header=[f"Authorization: Bearer {self._token}"],
                on_message=on_message,
                on_open=on_open,
                on_close=on_close,
            )
            try:
                app.run_forever(ping_interval=20, ping_timeout=10)
            except (OSError, RuntimeError):
                LOGGER.exception("Kraken remote WebSocket loop failed")
            if self._stop.wait(2.0):
                break


class RemoteServerProjectService:
    """Shared/PostgreSQL projects via kraken-server REST (+ optional WS sync)."""

    def __init__(
        self,
        base_url: str,
        *,
        auth: RemoteAuth,
        data_dir: Path | str | None = None,
        http: RemoteHttpClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.data_dir = Path(data_dir or Path.home() / ".kraken").resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile = REMOTE_STORAGE_PROFILE
        self._http = http or RemoteHttpClient(self.base_url)
        self._remote_ids: set[str] = set()
        self._sync: RemoteSyncClient | None = None
        self._wake_handlers: list[Callable[[ProjectEventWake], None]] = []
        self._catalog_handlers: list[Callable[[], None]] = []
        self._status_handlers: list[Callable[[str], None]] = []

    @property
    def supports_workspace_roots(self) -> bool:
        return False

    @property
    def supports_live_sync(self) -> bool:
        return True

    def list_storage_profiles(self) -> tuple[StorageProfile, ...]:
        return (self.profile,)

    def list_principals(
        self,
        *,
        include_inactive: bool = False,
    ) -> tuple[Principal, ...]:
        suffix = "?include_inactive=true" if include_inactive else ""
        payload = self._call("GET", f"/api/v1/principals{suffix}")
        values = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            Principal(
                id=str(item.get("principal_id") or item.get("id")),
                provider=PrincipalProvider(str(item.get("provider", "gitlab"))),
                subject=str(item.get("subject", "")),
                issuer=None if item.get("issuer") is None else str(item["issuer"]),
                display_name=str(item.get("display_name", "")),
                email=None if item.get("email") is None else str(item["email"]),
                active=bool(item.get("active", True)),
                system_roles=frozenset(
                    SystemRole(str(role)) for role in item.get("system_roles", ())
                ),
            )
            for item in values
            if isinstance(item, Mapping)
        )

    def list_project_principals(
        self,
        project_id: object,
        *,
        include_inactive: bool = False,
    ) -> tuple[Principal, ...]:
        del project_id
        return self.list_principals(include_inactive=include_inactive)

    def list_performers(
        self, *, include_archived: bool = False
    ) -> tuple[Performer, ...]:
        suffix = "?include_archived=true" if include_archived else ""
        payload = self._call("GET", f"/api/v1/performers{suffix}")
        values = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            decode_model(Performer, item)
            for item in values
            if isinstance(item, Mapping)
        )

    def add_wake_handler(self, handler: Callable[[ProjectEventWake], None]) -> None:
        self._wake_handlers.append(handler)

    def add_catalog_handler(self, handler: Callable[[], None]) -> None:
        self._catalog_handlers.append(handler)

    def add_status_handler(self, handler: Callable[[str], None]) -> None:
        self._status_handlers.append(handler)

    def start_sync(self) -> None:
        if self._sync is not None:
            return

        def on_event(wake: ProjectEventWake) -> None:
            for handler in list(self._wake_handlers):
                handler(wake)

        def on_catalog() -> None:
            for handler in list(self._catalog_handlers):
                handler()

        def on_status(status: str) -> None:
            for handler in list(self._status_handlers):
                handler(status)

        self._sync = RemoteSyncClient(
            self.base_url,
            token=self.auth.access_token,
            on_event=on_event,
            on_catalog=on_catalog,
            on_status=on_status,
        )
        self._sync.subscribe_catalog()
        self._sync.start()

    def stop_sync(self) -> None:
        if self._sync is not None:
            self._sync.stop()
            self._sync = None

    def subscribe_project(self, project_id: object) -> None:
        self.start_sync()
        if self._sync is not None:
            self._sync.subscribe_project(str(project_id))

    def unsubscribe_project(self, project_id: object) -> None:
        if self._sync is not None:
            self._sync.unsubscribe_project(str(project_id))

    def _call(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
        idempotency_key: str | None = None,
        if_match: int | None = None,
    ) -> Any:
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if if_match is not None:
            headers["If-Match"] = str(if_match)
        try:
            return self._http.request(
                method,
                path,
                token=self.auth.access_token,
                payload=payload,
                headers=headers,
            )
        except RemoteServerError as exc:
            if exc.status == 409 and exc.code == "revision.conflict":
                project_id = ""
                parts = [part for part in path.split("?")[0].split("/") if part]
                if "projects" in parts and parts.index("projects") + 1 < len(parts):
                    project_id = parts[parts.index("projects") + 1]
                wake = ProjectEventWake(
                    project_id=project_id,
                    event_type="revision.conflict",
                    event_id="",
                    revision=(
                        None
                        if exc.problem.get("actual_revision") is None
                        else int(exc.problem["actual_revision"])
                    ),
                    entity_kind=str(exc.problem.get("entity_kind", "")),
                    entity_id=str(exc.problem.get("entity_id", "")),
                )
                for handler in list(self._wake_handlers):
                    handler(wake)
                raise RemoteServerError(
                    "Другой пользователь уже сохранил изменение. "
                    "Экран обновлён; повторите текущую операцию.",
                    status=exc.status,
                    body=exc.body,
                    problem=exc.problem,
                ) from exc
            raise

    def list_projects(self, *, include_archived: bool = False) -> tuple[Project, ...]:
        suffix = "?include_archived=true" if include_archived else ""
        payload = self._call("GET", f"/api/v1/projects{suffix}")
        items = payload.get("items", []) if isinstance(payload, Mapping) else []
        projects: list[Project] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            project = project_from_server_dict(item)
            self._remote_ids.add(str(project.id))
            if not include_archived and project.state.value == "archived":
                continue
            projects.append(project)
        return tuple(projects)

    def get_project(
        self,
        project_id: object,
        *,
        as_of: datetime | None = None,
    ) -> Project | None:
        del as_of  # server temporal get is history-based; catalog uses current projection
        try:
            payload = self._call("GET", f"/api/v1/projects/{project_id}")
        except RemoteServerError as exc:
            if exc.status == 404:
                return None
            raise
        if not isinstance(payload, Mapping):
            return None
        project = project_from_server_dict(payload)
        self._remote_ids.add(str(project.id))
        return project

    def create_project(
        self,
        *,
        principal: Principal,
        name: str,
        width: int,
        height: int,
        orientation: GridOrientation,
        idempotency_key: str,
        layer_template: bool = False,
        source_root: Path | str | None = None,
        derived_root: Path | str | None = None,
        storage_profile_id: str | None = None,
    ) -> Project:
        del principal, layer_template, source_root, derived_root
        profile_id = storage_profile_id or self.profile.id
        payload = self._call(
            "POST",
            "/api/v1/projects",
            payload={
                "name": name,
                "width": width,
                "height": height,
                "orientation": orientation.value,
                "storage_profile_id": profile_id,
            },
            idempotency_key=idempotency_key,
        )
        project = project_from_server_dict(payload if isinstance(payload, Mapping) else {})
        self._remote_ids.add(str(project.id))
        return project

    def rename_project(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        idempotency_key: str,
    ) -> Project:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{project.id}",
            payload={"name": name},
            idempotency_key=idempotency_key,
            if_match=project.revision,
        )
        return project_from_server_dict(payload if isinstance(payload, Mapping) else {})

    def archive_project(
        self,
        *,
        principal: Principal,
        project: Project,
        idempotency_key: str,
    ) -> Project:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/archive",
            payload={},
            idempotency_key=idempotency_key,
            if_match=project.revision,
        )
        return project_from_server_dict(payload if isinstance(payload, Mapping) else {})

    def restore_project(
        self,
        *,
        principal: Principal,
        project: Project,
        idempotency_key: str,
    ) -> Project:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/restore",
            payload={},
            idempotency_key=idempotency_key,
            if_match=project.revision,
        )
        return project_from_server_dict(payload if isinstance(payload, Mapping) else {})

    def list_layers(self, project_id: object, *, include_archived: bool = False) -> tuple[Layer, ...]:
        suffix = "?include_archived=true" if include_archived else ""
        payload = self._call("GET", f"/api/v1/projects/{project_id}/layers{suffix}")
        items = payload if isinstance(payload, list) else payload.get("items", []) if isinstance(payload, Mapping) else []
        layers: list[Layer] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            state = StructureState(str(item.get("state", "active")))
            if not include_archived and state is StructureState.ARCHIVED:
                continue
            created = item.get("created_at")
            created_at = (
                datetime.fromisoformat(str(created))
                if created
                else datetime.now().astimezone()
            )
            layers.append(
                Layer(
                    id=str(item.get("layer_id") or item.get("id")),
                    project_id=ProjectId(str(project_id)),
                    name=str(item["name"]),
                    type=LayerType(str(item.get("type", LayerType.METAL.value))),
                    order=int(item.get("order", 0)),
                    state=state,
                    revision=int(item.get("revision", 0)),
                    created_at=created_at,
                )
            )
        return tuple(layers)

    @staticmethod
    def _layer(project_id: object, item: Mapping[str, object]) -> Layer:
        created = item.get("created_at")
        return Layer(
            id=str(item.get("layer_id") or item.get("id")),
            project_id=ProjectId(str(project_id)),
            name=str(item["name"]),
            type=LayerType(str(item.get("type", LayerType.METAL.value))),
            order=int(item.get("order", 0)),
            state=StructureState(str(item.get("state", "active"))),
            revision=int(item.get("revision", 0)),
            created_at=(
                datetime.fromisoformat(str(created))
                if created
                else datetime.now().astimezone()
            ),
        )

    @staticmethod
    def _representation(
        project_id: object,
        layer_id: object,
        item: Mapping[str, object],
    ) -> Representation:
        created = item.get("created_at")
        kind = RepresentationKind(str(item.get("kind", "image")))
        return Representation(
            id=str(item.get("representation_id") or item.get("id")),
            project_id=ProjectId(str(project_id)),
            layer_id=str(layer_id),
            name=str(item["name"]),
            kind=kind,
            purpose=RepresentationPurpose(
                str(
                    item.get(
                        "purpose",
                        "vector" if kind is RepresentationKind.VECTOR else "source",
                    )
                )
            ),
            note=str(item.get("note", "")),
            source=None if item.get("source") is None else str(item["source"]),
            source_image_representation_id=(
                None
                if item.get("source_image_representation_id") is None
                else str(item["source_image_representation_id"])
            ),
            active=bool(item.get("active", False)),
            state=StructureState(str(item.get("state", "active"))),
            revision=int(item.get("revision", 0)),
            created_at=(
                datetime.fromisoformat(str(created))
                if created
                else datetime.now().astimezone()
            ),
        )

    def create_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        idempotency_key: str,
        layer_id: object | None = None,
    ) -> Layer:
        del principal, layer_id
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers",
            payload={"name": name, "type": layer_type.value, "order": order},
            idempotency_key=idempotency_key,
            if_match=project.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid layer")
        return self._layer(project.id, payload)

    def rename_layer(self, *, principal: Principal, project: Project, layer: Layer, name: str, idempotency_key: str) -> Layer:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/{layer.id}/rename",
            payload={"name": name},
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid layer")
        return self._layer(project.id, payload)

    def reorder_layers(
        self,
        *,
        principal: Principal,
        project: Project,
        layers: tuple[Layer, ...] | list[Layer],
        layer_ids: tuple[str, ...],
        idempotency_key: str,
    ) -> tuple[Layer, ...]:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/reorder",
            payload={
                "layer_ids": list(layer_ids),
                "expected_revisions": {
                    str(layer.id): layer.revision for layer in layers
                },
            },
            idempotency_key=idempotency_key,
        )
        values = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            self._layer(project.id, item)
            for item in values
            if isinstance(item, Mapping)
        )

    def archive_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        idempotency_key: str,
    ) -> Layer:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/{layer.id}/archive",
            payload={},
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid layer")
        return self._layer(project.id, payload)

    def list_representations(
        self,
        project_id: object,
        layer_id: object,
        *,
        include_archived: bool = False,
    ) -> tuple[Representation, ...]:
        payload = self._call(
            "GET",
            f"/api/v1/projects/{project_id}/layers/{layer_id}/representations"
            + ("?include_archived=true" if include_archived else ""),
        )
        values = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        representations = tuple(
            self._representation(project_id, layer_id, item)
            for item in values
            if isinstance(item, Mapping)
        )
        if include_archived:
            return representations
        return tuple(
            item for item in representations if item.state is not StructureState.ARCHIVED
        )

    def create_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        kind: RepresentationKind,
        idempotency_key: str,
        note: str = "",
        source: str | None = None,
        source_image_representation_id: object | None = None,
        active: bool = False,
        purpose: RepresentationPurpose = RepresentationPurpose.SOURCE,
    ) -> Representation:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/{layer.id}/representations",
            payload={
                "name": name,
                "kind": kind.value,
                "purpose": (
                    RepresentationPurpose.VECTOR.value
                    if kind is RepresentationKind.VECTOR
                    else purpose.value
                ),
                "note": note,
                "source": source,
                "source_image_representation_id": (
                    None
                    if source_image_representation_id is None
                    else str(source_image_representation_id)
                ),
                "active": active,
            },
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid representation")
        return self._representation(project.id, layer.id, payload)

    def _update_representation(
        self,
        *,
        project: Project,
        layer: Layer,
        representation: Representation,
        operation: Mapping[str, object],
        idempotency_key: str,
    ) -> Representation:
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{project.id}/layers/{layer.id}/representations/{representation.id}",
            payload={
                **operation,
                "expected_representation_revision": representation.revision,
            },
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid representation")
        return self._representation(project.id, layer.id, payload)

    def rename_representation(self, *, principal: Principal, project: Project, layer: Layer, representation: Representation, name: str, idempotency_key: str) -> Representation:
        del principal
        return self._update_representation(project=project, layer=layer, representation=representation, operation={"name": name}, idempotency_key=idempotency_key)

    def update_representation_note(self, *, principal: Principal, project: Project, layer: Layer, representation: Representation, note: str, idempotency_key: str) -> Representation:
        del principal
        return self._update_representation(project=project, layer=layer, representation=representation, operation={"note": note}, idempotency_key=idempotency_key)

    def activate_representation(self, *, principal: Principal, project: Project, layer: Layer, representation: Representation, idempotency_key: str) -> Representation:
        del principal
        return self._update_representation(project=project, layer=layer, representation=representation, operation={"active": True}, idempotency_key=idempotency_key)

    def deactivate_representation(self, *, principal: Principal, project: Project, layer: Layer, representation: Representation, idempotency_key: str) -> Representation:
        del principal
        return self._update_representation(project=project, layer=layer, representation=representation, operation={"active": False}, idempotency_key=idempotency_key)

    def archive_representation(self, *, principal: Principal, project: Project, layer: Layer, representation: Representation, idempotency_key: str) -> Representation:
        del principal
        return self._update_representation(project=project, layer=layer, representation=representation, operation={"archive": True}, idempotency_key=idempotency_key)

    def project_permissions(self, project_id: object, principal: Principal) -> frozenset[Permission]:
        try:
            payload = self._call(
                "GET",
                f"/api/v1/projects/{project_id}/acl/{principal.id}",
            )
        except RemoteServerError:
            return frozenset()
        roles_raw = payload.get("roles", ()) if isinstance(payload, Mapping) else ()
        roles: set[ProjectRole] = set()
        for role_name in roles_raw:
            try:
                roles.add(ProjectRole(str(role_name)))
            except ValueError:
                continue
        return permissions_for_roles(roles)

    def project_roles(
        self,
        project_id: object,
        principal_id: object,
    ) -> frozenset[ProjectRole]:
        payload = self._call(
            "GET",
            f"/api/v1/projects/{project_id}/acl/{principal_id}",
        )
        raw_roles = payload.get("roles", ()) if isinstance(payload, Mapping) else ()
        return frozenset(ProjectRole(str(role)) for role in raw_roles)

    def project_role_revision(
        self,
        project_id: object,
        principal_id: object,
    ) -> int:
        payload = self._call(
            "GET",
            f"/api/v1/projects/{project_id}/acl/{principal_id}",
        )
        return int(payload.get("revision", 0)) if isinstance(payload, Mapping) else 0

    def _change_project_role(
        self,
        *,
        method: str,
        project: Project,
        target_principal_id: object,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        payload = self._call(
            method,
            f"/api/v1/projects/{project.id}/acl/{target_principal_id}/{role.value}",
            payload={} if method == "PUT" else None,
            idempotency_key=idempotency_key,
            if_match=expected_revision,
        )
        raw_roles = payload.get("roles", ()) if isinstance(payload, Mapping) else ()
        return frozenset(ProjectRole(str(value)) for value in raw_roles)

    def assign_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: object,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        del principal
        return self._change_project_role(
            method="PUT",
            project=project,
            target_principal_id=target_principal_id,
            role=role,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        )

    def revoke_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: object,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        del principal
        return self._change_project_role(
            method="DELETE",
            project=project,
            target_principal_id=target_principal_id,
            role=role,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        )

    def list_artifact_series(
        self,
        project_id: object,
        *,
        layer_id: object | None = None,
        representation_id: object | None = None,
        include_archived: bool = False,
    ) -> tuple[ArtifactSeries, ...]:
        query = urllib.parse.urlencode(
            {
                key: value
                for key, value in {
                    "layer_id": None if layer_id is None else str(layer_id),
                    "representation_id": None if representation_id is None else str(representation_id),
                    "include_archived": "true" if include_archived else None,
                }.items()
                if value is not None
            }
        )
        payload = self._call("GET", f"/api/v1/projects/{project_id}/artifacts?{query}")
        items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            decode_model(ArtifactSeries, item)
            for item in items
            if isinstance(item, Mapping)
        )

    def create_artifact_series(
        self,
        *,
        principal: Principal,
        project_id: object,
        scope: ArtifactScope,
        name: str,
        layer_id: object | None = None,
        representation_id: object | None = None,
        frame_id: object | None = None,
        idempotency_key: str,
    ) -> ArtifactSeries:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/artifacts",
            payload={
                "scope": scope.value,
                "name": name,
                "layer_id": None if layer_id is None else str(layer_id),
                "representation_id": None if representation_id is None else str(representation_id),
                "frame_id": None if frame_id is None else str(frame_id),
            },
            idempotency_key=idempotency_key,
        )
        return decode_model(ArtifactSeries, payload)

    def artifact_stream_revision(self, project_id: object, series_id: object) -> int:
        payload = self._call(
            "GET", f"/api/v1/projects/{project_id}/artifacts/{series_id}/revision"
        )
        return int(payload.get("revision", 0)) if isinstance(payload, Mapping) else 0

    def artifact_versions(
        self, project_id: object, series_id: object
    ) -> tuple[ArtifactVersion, ...]:
        payload = self._call(
            "GET", f"/api/v1/projects/{project_id}/artifacts/{series_id}/versions"
        )
        items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            decode_model(ArtifactVersion, item)
            for item in items
            if isinstance(item, Mapping)
        )

    def active_artifact_version(
        self, project_id: object, series_id: object
    ) -> ArtifactVersion | None:
        payload = self._call(
            "GET", f"/api/v1/projects/{project_id}/artifacts/{series_id}/active"
        )
        item = payload.get("item") if isinstance(payload, Mapping) else None
        return decode_model(ArtifactVersion, item) if isinstance(item, Mapping) else None

    def artifact_version(self, project_id: object, version_id: object) -> ArtifactVersion | None:
        try:
            payload = self._call(
                "GET", f"/api/v1/projects/{project_id}/artifacts/versions/{version_id}/metadata"
            )
        except RemoteServerError as exc:
            if exc.status == 404:
                return None
            raise
        return decode_model(ArtifactVersion, payload) if isinstance(payload, Mapping) else None

    def add_managed_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: object,
        series_id: object,
        source: Path | str,
        parent_version_id: object | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        del principal
        path = Path(source).resolve(strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        query = urllib.parse.urlencode(
            {
                "filename": path.name,
                "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "sha256": digest.hexdigest(),
                **(
                    {}
                    if parent_version_id is None
                    else {"parent_version_id": str(parent_version_id)}
                ),
            }
        )
        try:
            payload = self._http.upload(
                f"/api/v1/projects/{project_id}/artifacts/{series_id}/versions?{query}",
                token=self.auth.access_token,
                source=path,
                headers={
                    "Idempotency-Key": idempotency_key,
                    "If-Match": str(
                        self.artifact_stream_revision(project_id, series_id)
                    ),
                },
            )
        except RemoteServerError as exc:
            if exc.status == 409 and exc.code == "revision.conflict":
                wake = ProjectEventWake(
                    project_id=str(project_id),
                    event_type="revision.conflict",
                    event_id="",
                    revision=(
                        None
                        if exc.problem.get("actual_revision") is None
                        else int(exc.problem["actual_revision"])
                    ),
                    entity_kind=str(exc.problem.get("entity_kind", "artifact_series")),
                    entity_id=str(exc.problem.get("entity_id", series_id)),
                )
                for handler in list(self._wake_handlers):
                    handler(wake)
                raise RemoteServerError(
                    "Другой пользователь уже сохранил изменение. "
                    "Экран обновлён; повторите текущую операцию.",
                    status=exc.status,
                    body=exc.body,
                    problem=exc.problem,
                ) from exc
            raise
        return decode_model(ArtifactVersion, payload)

    def add_external_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: object,
        series_id: object,
        source: Path | str,
        parent_version_id: object | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        del principal
        path = Path(source).resolve(strict=True)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/artifacts/{series_id}/external",
            payload={
                "filename": path.name,
                "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "uri": path.as_uri(),
                "sha256": digest.hexdigest(),
                "size_bytes": size,
                "parent_version_id": None if parent_version_id is None else str(parent_version_id),
                "parameters": {"source_path": str(path)},
            },
            idempotency_key=idempotency_key,
            if_match=self.artifact_stream_revision(project_id, series_id),
        )
        return decode_model(ArtifactVersion, payload)

    def rename_artifact_series(
        self, *, principal: Principal, series: ArtifactSeries, name: str, idempotency_key: str
    ) -> ArtifactSeries:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{series.project_id}/artifacts/{series.id}",
            payload={"name": name},
            idempotency_key=idempotency_key,
            if_match=series.revision,
        )
        return decode_model(ArtifactSeries, payload)

    def archive_artifact_series(
        self, *, principal: Principal, series: ArtifactSeries, idempotency_key: str
    ) -> ArtifactSeries:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{series.project_id}/artifacts/{series.id}",
            payload={"archive": True},
            idempotency_key=idempotency_key,
            if_match=series.revision,
        )
        return decode_model(ArtifactSeries, payload)

    def activate_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: object,
        series_id: object,
        version_id: object,
        idempotency_key: str,
    ) -> ArtifactVersion:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{project_id}/artifacts/{series_id}",
            payload={"active_version_id": str(version_id)},
            idempotency_key=idempotency_key,
            if_match=self.artifact_stream_revision(project_id, series_id),
        )
        return decode_model(ArtifactVersion, payload)

    def list_notes(
        self,
        project_id: object,
        *,
        layer_id: object | None = None,
        frame_id: object | None = None,
    ) -> tuple[NoteRevision, ...]:
        query = urllib.parse.urlencode(
            {
                key: value
                for key, value in {
                    "layer_id": None if layer_id is None else str(layer_id),
                    "frame_id": None if frame_id is None else str(frame_id),
                }.items()
                if value is not None
            }
        )
        payload = self._call("GET", f"/api/v1/projects/{project_id}/notes?{query}")
        items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            decode_model(NoteRevision, item)
            for item in items
            if isinstance(item, Mapping)
        )

    def create_note(
        self,
        *,
        principal: Principal,
        project_id: object,
        body: str,
        layer_id: object | None = None,
        frame_id: object | None = None,
        idempotency_key: str,
    ) -> NoteRevision:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/notes",
            payload={
                "body": body,
                "layer_id": None if layer_id is None else str(layer_id),
                "frame_id": None if frame_id is None else str(frame_id),
            },
            idempotency_key=idempotency_key,
        )
        return decode_model(NoteRevision, payload)

    def revise_note(
        self,
        *,
        principal: Principal,
        note: NoteRevision,
        body: str,
        idempotency_key: str,
    ) -> NoteRevision:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{note.project_id}/notes/{note.note_id}",
            payload={"body": body},
            idempotency_key=idempotency_key,
            if_match=note.revision,
        )
        return decode_model(NoteRevision, payload)

    def export_managed_artifact(
        self, project_id: object, version: ArtifactVersion, destination: Path | str
    ) -> Path:
        if version.blob is None:
            raise ValueError("The selected artifact version is external")
        return self._http.download(
            f"/api/v1/projects/{project_id}/artifacts/versions/{version.id}",
            token=self.auth.access_token,
            destination=Path(destination).resolve(),
        )

    def managed_artifact_path(self, project_id: object, version_id: object) -> Path:
        version = self.artifact_version(project_id, version_id)
        if version is None or version.blob is None:
            raise ValueError("Для запуска нужны файлы, сохранённые в проекте")
        target = self.data_dir / "remote-cache" / str(project_id) / version.blob.sha256
        if target.is_file() and target.stat().st_size == version.blob.size_bytes:
            return target
        target.unlink(missing_ok=True)
        temporary = target.with_name(f"{target.name}.part-{threading.get_ident()}")
        temporary.unlink(missing_ok=True)
        self._http.download(
            f"/api/v1/projects/{project_id}/artifacts/versions/{version.id}",
            token=self.auth.access_token,
            destination=temporary,
        )
        digest = hashlib.sha256()
        size = 0
        with temporary.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != version.blob.sha256 or size != version.blob.size_bytes:
            temporary.unlink(missing_ok=True)
            raise ValueError("Сервер вернул blob с неверной контрольной суммой или размером")
        temporary.replace(target)
        return target

    def read_project_blob(self, project_id: object, source_key: str) -> bytes:
        path = self.managed_artifact_path(project_id, source_key)
        return path.read_bytes()

    def external_artifact_changed(self, version: ArtifactVersion) -> bool:
        if version.external is None:
            return False
        parsed = urlparse(version.external.uri)
        if parsed.scheme != "file":
            raise ValueError(
                f"Внешний файл недоступен на этой рабочей станции: {version.external.uri}"
            )
        path = Path(unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path))
        if os.name == "nt" and parsed.netloc:
            path = Path(f"//{parsed.netloc}/{unquote(parsed.path.lstrip('/'))}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise FileNotFoundError(
                f"Внешний файл недоступен на этой рабочей станции: {path}"
            ) from exc
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return (
            size != version.external.observed_size_bytes
            or digest.hexdigest() != version.external.fingerprint_sha256
        )

    def export_artifact_version(
        self, project_id: object, version: ArtifactVersion, destination: Path | str
    ) -> Path:
        if version.external is None:
            return self.export_managed_artifact(project_id, version, destination)
        if self.external_artifact_changed(version):
            raise ValueError(
                f"Внешний файл изменён после регистрации: {version.external.uri}"
            )
        parsed = urlparse(version.external.uri)
        source = Path(unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path))
        if os.name == "nt" and parsed.netloc:
            source = Path(f"//{parsed.netloc}/{unquote(parsed.path.lstrip('/'))}")
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(source.resolve(strict=True), target)
        return target

    def review_batches(self) -> tuple[ReviewBatch, ...]:
        batches: list[ReviewBatch] = []
        for project in self.list_projects(include_archived=True):
            payload = self._call("GET", f"/api/v1/projects/{project.id}/reviews")
            items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
            batches.extend(
                decode_model(ReviewBatch, item)
                for item in items
                if isinstance(item, Mapping)
            )
        return tuple(sorted(batches, key=lambda item: (item.updated_at, str(item.id)), reverse=True))

    def active_review_batches(self) -> tuple[ReviewBatch, ...]:
        batches: list[ReviewBatch] = []
        for project in self.list_projects(include_archived=True):
            payload = self._call(
                "GET", f"/api/v1/projects/{project.id}/reviews?active_only=true"
            )
            items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
            batches.extend(
                decode_model(ReviewBatch, item)
                for item in items
                if isinstance(item, Mapping)
            )
        return tuple(sorted(batches, key=lambda item: (item.updated_at, str(item.id)), reverse=True))

    def create_review_batch(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        image_representation_id: object,
        vector_representation_id: object,
        coordinates: object,
        assignee_id: object,
        instructions: str = "",
        due_at: datetime | None = None,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        layer = next(
            (item for item in self.list_layers(project_id) if str(item.id) == str(layer_id)),
            None,
        )
        if layer is None:
            raise ValueError("Layer was not found")
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/reviews",
            payload={
                "layer_id": str(layer_id),
                "image_representation_id": str(image_representation_id),
                "vector_representation_id": str(vector_representation_id),
                "coordinates": [[int(x), int(y)] for x, y in coordinates],
                "assignee_id": str(assignee_id),
                "instructions": instructions,
                "due_at": None if due_at is None else due_at.isoformat(),
            },
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        return decode_model(ReviewBatch, payload)

    def accept_review(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        candidate_version_ids: object,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}",
            payload={
                "action": "accept",
                "candidate_version_ids": [str(item) for item in candidate_version_ids],
            },
            idempotency_key=idempotency_key,
            if_match=batch.revision,
        )
        return decode_model(ReviewBatch, payload)

    def request_review_changes(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        reason: str,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}",
            payload={"action": "request_changes", "reason": reason},
            idempotency_key=idempotency_key,
            if_match=batch.revision,
        )
        return decode_model(ReviewBatch, payload)

    def cancel_review_batch(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}",
            payload={"action": "cancel"},
            idempotency_key=idempotency_key,
            if_match=batch.revision,
        )
        return decode_model(ReviewBatch, payload)

    def review_candidate_version_ids(self, batch: ReviewBatch) -> tuple[object, ...]:
        latest_by_series: dict[str, object] = {}
        for event in self.history(batch.project_id):
            if (
                event.event_type != "ReviewReturnCommitted"
                or str(event.payload.get("review_batch_id", "")) != str(batch.id)
            ):
                continue
            for value in event.payload.get("candidate_version_ids", ()):
                version = self.artifact_version(batch.project_id, value)
                if version is not None:
                    latest_by_series[str(version.series_id)] = version.id
        return tuple(latest_by_series[key] for key in sorted(latest_by_series))

    def export_review_batch(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        destination: Path | str,
        idempotency_key: str,
    ) -> ReviewBatch:
        del principal
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}-{threading.get_ident()}.zip")
        temporary.unlink(missing_ok=True)
        try:
            self._http.download(
                f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}/package",
                token=self.auth.access_token,
                destination=temporary,
                method="POST",
                headers={
                    "Idempotency-Key": idempotency_key,
                    "If-Match": str(batch.revision),
                },
            )
            _extract_review_archive(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return next(
            (
                item
                for item in self.review_batches()
                if str(item.id) == str(batch.id)
            ),
            batch,
        )

    def _review_source(self, source: Path | str) -> tuple[Path, ReviewBatch]:
        folder = Path(source).resolve(strict=True)
        raw = (folder / "kraken-review.json").read_bytes()
        if len(raw) > 16 * 1024**2:
            raise ValueError("Review manifest exceeds the size limit")
        manifest = manifest_from_json(raw.decode("utf-8"))
        batch_id = manifest.batch_id or manifest.package_id
        batch = next(
            (
                item
                for item in self.review_batches()
                if str(item.project_id) == str(manifest.project_id)
                and str(item.id) == str(batch_id)
            ),
            None,
        )
        if batch is None:
            raise ValueError("Manifest does not match a known server review batch")
        return folder, batch

    def _upload_review_return(
        self,
        source: Path | str,
        batch: ReviewBatch,
        *,
        operation: str,
        idempotency_key: str,
    ) -> Mapping[str, object]:
        folder = Path(source).resolve(strict=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="kraken-review-return-", suffix=".zip"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink(missing_ok=True)
        try:
            _review_archive(folder, temporary)
            payload = self._http.upload(
                f"/api/v1/projects/{batch.project_id}/reviews/{batch.id}/return/{operation}",
                token=self.auth.access_token,
                source=temporary,
                headers={
                    "Idempotency-Key": idempotency_key,
                    "If-Match": str(batch.revision),
                },
            )
        finally:
            temporary.unlink(missing_ok=True)
        if not isinstance(payload, Mapping):
            raise RemoteServerError("Kraken Server returned an invalid review response")
        return payload

    def review_return_preflight(
        self,
        *,
        principal: Principal,
        source: Path | str,
        idempotency_key: str,
    ) -> tuple[ReviewBatch, ReviewReturnPlan]:
        del principal
        folder, batch = self._review_source(source)
        payload = self._upload_review_return(
            folder,
            batch,
            operation="preflight",
            idempotency_key=idempotency_key,
        )
        return batch, decode_model(ReviewReturnPlan, payload)

    def commit_review_return(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        source: Path | str,
        idempotency_key: str,
    ) -> ReviewReturnCommitResult:
        del principal
        payload = self._upload_review_return(
            source,
            batch,
            operation="commit",
            idempotency_key=idempotency_key,
        )
        return decode_model(ReviewReturnCommitResult, payload)

    def plugin_jobs(self) -> tuple[PluginJob, ...]:
        jobs: list[PluginJob] = []
        for project in self.list_projects(include_archived=True):
            payload = self._call("GET", f"/api/v1/projects/{project.id}/jobs")
            items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
            jobs.extend(
                decode_model(PluginJob, item)
                for item in items
                if isinstance(item, Mapping)
            )
        return tuple(sorted(jobs, key=lambda item: (item.updated_at, str(item.id)), reverse=True))

    def submit_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        project_id: object,
        layer_id: object,
        source_representation_id: object,
        target_representation_id: object,
        coordinates: object,
        capability: str,
        parameters: Mapping[str, object],
        idempotency_key: str,
    ) -> PluginJob:
        del principal, gateway
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/jobs",
            payload={
                "layer_id": str(layer_id),
                "source_representation_id": str(source_representation_id),
                "target_representation_id": str(target_representation_id),
                "coordinates": [[int(x), int(y)] for x, y in coordinates],
                "capability": capability,
                "parameters": dict(parameters),
            },
            idempotency_key=idempotency_key,
        )
        return decode_model(PluginJob, payload)

    def cancel_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        job: PluginJob,
        idempotency_key: str,
    ) -> PluginJob:
        del principal, gateway
        payload = self._call(
            "POST",
            f"/api/v1/projects/{job.project_id}/jobs/{job.id}/cancel",
            payload={},
            idempotency_key=idempotency_key,
            if_match=job.revision,
        )
        return decode_model(PluginJob, payload)

    def synchronize_plugin_jobs(self, *, principal: Principal, gateway: object) -> tuple[PluginJob, ...]:
        del principal, gateway
        return self.plugin_jobs()

    def project_workspace(self, project_id: object) -> None:
        return None

    def project_storage_label(self, project_id: object) -> str:
        return "Сервер PostgreSQL"

    def is_remote_project(self, project_id: object) -> bool:
        return str(project_id) in self._remote_ids or True

    def history(self, project_id: object, *, as_of: datetime | None = None) -> tuple[object, ...]:
        del as_of
        events = []
        cursor: str | None = None
        while True:
            query = urllib.parse.urlencode(
                {"limit": 500, **({} if cursor is None else {"cursor": cursor})}
            )
            payload = self._call(
                "GET", f"/api/v1/projects/{project_id}/history?{query}"
            )
            items = payload.get("items", []) if isinstance(payload, Mapping) else []
            events.extend(
                _RemoteHistoryEvent(item)
                for item in items
                if isinstance(item, Mapping)
            )
            cursor = (
                None
                if not isinstance(payload, Mapping) or payload.get("next_cursor") is None
                else str(payload["next_cursor"])
            )
            if cursor is None:
                break
        return tuple(events)

    def activity_records(self) -> tuple[object, ...]:
        from kraken_manager.infrastructure.reports import ActivityRecord

        records = [
            ActivityRecord.from_event(event)
            for project in self.list_projects(include_archived=True)
            for event in self.history(project.id)
        ]
        return tuple(sorted(records, key=lambda item: (item.recorded_at, item.event_id)))

    def statistics(
        self,
        project_id: object,
        *,
        start: datetime,
        end: datetime,
        timezone: object,
    ) -> tuple[ReportMetrics, dict[str, ReportSeries]]:
        query = urllib.parse.urlencode(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "timezone": str(getattr(timezone, "key", "UTC")),
            }
        )
        payload = self._call(
            "GET", f"/api/v1/projects/{project_id}/statistics?{query}"
        )
        metrics = decode_model(ReportMetrics, payload["metrics"])
        series = {
            str(key): decode_model(ReportSeries, value)
            for key, value in dict(payload.get("series", {})).items()
        }
        return metrics, series

    def _stream_revision(self, project_id: object, stream_id: str) -> int:
        return max(
            (
                event.revision
                for event in self.history(project_id)
                if event.stream_id == stream_id
            ),
            default=0,
        )

    def publish_karakal_analysis(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        frame_confidence: dict[str, float],
        report: dict[str, object],
        parameters: dict[str, object],
        plugin_version: str,
        idempotency_key: str,
    ) -> object:
        del principal
        stream_id = f"karakal:{layer_id}"
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/analyses/karakal",
            payload={
                "layer_id": str(layer_id),
                "frame_confidence": frame_confidence,
                "report": report,
                "parameters": parameters,
                "plugin_version": plugin_version,
            },
            idempotency_key=idempotency_key,
            if_match=self._stream_revision(project_id, stream_id),
        )
        return self._karakal_run(project_id, payload)

    def latest_karakal_analysis(
        self, project_id: object, layer_id: object, *, as_of: datetime | None = None
    ) -> object | None:
        events = self.history(project_id, as_of=as_of)
        event = next(
            (
                item
                for item in reversed(events)
                if item.event_type == "KarakalAnalysisPublished"
                and str(item.payload.get("layer_id", "")) == str(layer_id)
            ),
            None,
        )
        return None if event is None else self._karakal_run(project_id, event._payload)

    @staticmethod
    def _karakal_run(project_id: object, event: Mapping[str, object]) -> object:
        from .composition import KarakalAnalysisRun

        raw = event.get("payload", {})
        payload = raw if isinstance(raw, Mapping) else {}
        return KarakalAnalysisRun(
            run_id=str(payload.get("run_id", "")),
            project_id=str(project_id),
            layer_id=str(payload.get("layer_id", "")),
            publication_sequence=int(payload.get("publication_sequence", 0)),
            created_at=str(event.get("recorded_at", "")),
            frame_confidence={
                str(key): float(value)
                for key, value in dict(payload.get("frame_confidence", {})).items()
            },
            report=dict(payload.get("report", {})),
            parameters=dict(payload.get("parameters", {})),
            plugin_version=str(payload.get("plugin_version", "")),
        )

    def record_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        action: str,
        node_id: str,
        plugin_id: str,
        capability: str,
        mode: str,
        parameters: dict[str, object] | None = None,
    ) -> object:
        del principal
        stream_id = f"layer-pipeline:{layer_id}"
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/pipeline-actions",
            payload={
                "event_type": "LayerPipelineActionRequested",
                "layer_id": str(layer_id),
                "action": action,
                "node_id": node_id,
                "plugin_id": plugin_id,
                "capability": capability,
                "mode": mode,
                "parameters": dict(parameters or {}),
                "state": "launched",
            },
            idempotency_key=f"pipeline:{layer_id}:{uuid4()}",
            if_match=self._stream_revision(project_id, stream_id),
        )
        return _RemoteHistoryEvent(payload)

    def remove_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        action_event_id: str,
    ) -> object:
        del principal
        target = next(
            (
                item
                for item in self.history(project_id)
                if item.event_id == str(action_event_id)
                and item.event_type == "LayerPipelineActionRequested"
                and str(item.payload.get("layer_id", "")) == str(layer_id)
            ),
            None,
        )
        if target is None:
            raise ValueError("Pipeline step does not exist in this layer")
        stream_id = f"layer-pipeline:{layer_id}"
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/pipeline-actions",
            payload={
                "event_type": "LayerPipelineActionRemoved",
                "layer_id": str(layer_id),
                "action_event_id": target.event_id,
                "action": str(target.payload.get("action", "")),
            },
            idempotency_key=f"pipeline-remove:{target.event_id}",
            if_match=self._stream_revision(project_id, stream_id),
        )
        return _RemoteHistoryEvent(payload)

    def matrix_viewport(
        self,
        project_id: object,
        *,
        layer_id: object,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int = 0,
        representation_ids: object = (),
        include_missing: bool = True,
    ) -> Mapping[str, object]:
        query = urllib.parse.urlencode(
            [
                ("layer_id", str(layer_id)),
                ("x1", str(x1)),
                ("y1", str(y1)),
                ("x2", str(x2)),
                ("y2", str(y2)),
                ("lod", str(lod)),
                ("include_missing", "true" if include_missing else "false"),
                *[("representation_id", str(value)) for value in representation_ids],
            ]
        )
        return self._call(
            "GET",
            f"/api/v1/projects/{project_id}/viewport?{query}",
        )

    def frame_cells(
        self,
        project_id: object,
        layer_id: object,
        representation_id: object,
        *,
        as_of: datetime | None = None,
    ) -> tuple[object, ...]:
        if as_of is not None:
            raise ValueError("Temporal server matrix views are not supported by API v1")
        query = urllib.parse.urlencode(
            {
                "layer_id": str(layer_id),
                "representation_id": str(representation_id),
            }
        )
        payload = self._call("GET", f"/api/v1/projects/{project_id}/frames?{query}")
        return tuple(
            SimpleNamespace(
                x=int(item["x"]),
                y=int(item["y"]),
                status=str(item.get("status", "empty")),
                frame_id=str(item.get("frame_id", "")),
                artifact_version_id=str(item.get("artifact_version_id", "")),
                sha256=str(item.get("sha256", "")),
                modified_at=str(item.get("modified_at", "")),
                performer_color=str(item.get("performer_color", "")),
                performer_initials=str(item.get("performer_initials", "")),
                review_status=str(item.get("review_status", "not_checked")),
                quality=item.get("quality"),
            )
            for item in payload.get("items", ())
            if not item.get("missing")
        )

    def frame_management_states(
        self,
        project_id: object,
        layer_id: object,
        representation_id: object,
        *,
        as_of: datetime | None = None,
    ) -> tuple[object, ...]:
        return tuple(
            SimpleNamespace(
                frame_id=item.frame_id,
                artifact_version_id=item.artifact_version_id,
                modified_at=item.modified_at,
                performer_color=item.performer_color,
                performer_initials=item.performer_initials,
                review_status=item.review_status,
            )
            for item in self.frame_cells(
                project_id, layer_id, representation_id, as_of=as_of
            )
        )


@dataclass(frozen=True, slots=True)
class _RemoteHistoryEvent:
    _payload: Mapping[str, object]

    @property
    def event_id(self) -> str:
        return str(self._payload.get("event_id", ""))

    @property
    def event_type(self) -> str:
        return str(self._payload.get("event_type", ""))

    @property
    def stream_id(self) -> str:
        return str(self._payload.get("stream_id", ""))

    @property
    def revision(self) -> int:
        return int(self._payload.get("revision", 0))

    @property
    def recorded_at(self) -> datetime:
        value = self._payload.get("recorded_at")
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))

    @property
    def payload(self) -> Mapping[str, object]:
        raw = self._payload.get("payload", {})
        return raw if isinstance(raw, Mapping) else {}

    @property
    def actor(self) -> object:
        raw = self._payload.get("actor", {})
        if not isinstance(raw, Mapping):
            return None

        @dataclass(frozen=True, slots=True)
        class _Actor:
            display_name: str
            principal_id: str

        return _Actor(
            display_name=str(raw.get("display_name", "")),
            principal_id=str(raw.get("principal_id", "")),
        )


def load_remote_auth_from_env() -> RemoteAuth | None:
    token = os.environ.get("KRAKEN_GITLAB_TOKEN") or os.environ.get("KRAKEN_SERVER_TOKEN") or ""
    token = token.strip()
    if not token:
        return None
    subject = os.environ.get("KRAKEN_GITLAB_SUBJECT", "remote-user").strip() or "remote-user"
    display = os.environ.get("KRAKEN_GITLAB_DISPLAY_NAME", "Remote User").strip() or "Remote User"
    issuer = os.environ.get("KRAKEN_GITLAB_ISSUER", "https://gitlab.local").strip()
    principal = Principal.gitlab(
        issuer=issuer,
        subject=subject,
        display_name=display,
        principal_id=os.environ.get("KRAKEN_GITLAB_PRINCIPAL_ID") or None,
    )
    return RemoteAuth(access_token=token, principal=principal)


def build_remote_service(
    *,
    base_url: str | None = None,
    auth: RemoteAuth | None = None,
    data_dir: Path | str | None = None,
) -> RemoteServerProjectService | None:
    url = (base_url or os.environ.get("KRAKEN_SERVER_URL") or "").strip()
    if not url:
        return None
    credentials = auth or load_remote_auth_from_env()
    if credentials is None:
        return None
    return RemoteServerProjectService(url, auth=credentials, data_dir=data_dir)


__all__ = [
    "RemoteAuth",
    "RemoteHttpClient",
    "RemoteServerError",
    "RemoteServerProjectService",
    "RemoteSyncClient",
    "build_remote_service",
    "load_remote_auth_from_env",
]
