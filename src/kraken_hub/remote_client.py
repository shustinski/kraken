"""HTTP/WebSocket client for shared projects hosted by kraken-server."""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kraken_manager.application.ports import StorageProfile
from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.identity import (
    Permission,
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


class RemoteServerError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


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
            try:
                parsed = json.loads(detail)
                message = str(parsed.get("detail") or parsed.get("title") or detail)
            except (json.JSONDecodeError, TypeError, AttributeError):
                message = detail
            raise RemoteServerError(message, status=exc.code, body=detail) from exc
        except urllib.error.URLError as exc:
            raise RemoteServerError(f"Cannot reach Kraken Server at {self.base_url}: {exc.reason}") from exc


class RemoteSyncClient:
    """Background WebSocket subscriber that emits ProjectEventWake callbacks."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        on_event: Callable[[ProjectEventWake], None],
        on_catalog: Callable[[], None] | None = None,
    ) -> None:
        self._ws_url = self._to_ws_url(base_url)
        self._token = token
        self._on_event = on_event
        self._on_catalog = on_catalog
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
            # Fallback: poll is not implemented here; live sync requires websocket-client.
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
                )
                if wake.project_id:
                    self._on_event(wake)
                if self._on_catalog is not None:
                    self._on_catalog()
            elif kind == "catalog_changed" and self._on_catalog is not None:
                self._on_catalog()

        def on_open(ws: Any) -> None:
            self._ws = ws
            if self._catalog:
                self._send({"type": "subscribe", "catalog": True})
            for project_id in list(self._subscribed):
                self._send({"type": "subscribe", "project_id": project_id})

        def on_close(_ws: Any, *_args: object) -> None:
            self._ws = None

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

    def add_wake_handler(self, handler: Callable[[ProjectEventWake], None]) -> None:
        self._wake_handlers.append(handler)

    def add_catalog_handler(self, handler: Callable[[], None]) -> None:
        self._catalog_handlers.append(handler)

    def start_sync(self) -> None:
        if self._sync is not None:
            return

        def on_event(wake: ProjectEventWake) -> None:
            for handler in list(self._wake_handlers):
                handler(wake)

        def on_catalog() -> None:
            for handler in list(self._catalog_handlers):
                handler()

        self._sync = RemoteSyncClient(
            self.base_url,
            token=self.auth.access_token,
            on_event=on_event,
            on_catalog=on_catalog,
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
        return self._http.request(
            method,
            path,
            token=self.auth.access_token,
            payload=payload,
            headers=headers,
        )

    def list_projects(self, *, include_archived: bool = False) -> tuple[Project, ...]:
        payload = self._call("GET", "/api/v1/projects")
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
        payload = self._call("GET", f"/api/v1/projects/{project_id}/layers")
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
            f"/api/v1/projects/{project_id}/layers/{layer_id}/representations",
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

    def project_workspace(self, project_id: object) -> None:
        return None

    def project_storage_label(self, project_id: object) -> str:
        return "Сервер PostgreSQL"

    def is_remote_project(self, project_id: object) -> bool:
        return str(project_id) in self._remote_ids or True

    def history(self, project_id: object, *, as_of: datetime | None = None) -> tuple[object, ...]:
        del as_of
        payload = self._call("GET", f"/api/v1/projects/{project_id}/history?limit=500")
        items = payload.get("items", []) if isinstance(payload, Mapping) else []
        events = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            events.append(_RemoteHistoryEvent(item))
        return tuple(events)

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
    ) -> Mapping[str, object]:
        del representation_ids
        return self._call(
            "GET",
            f"/api/v1/projects/{project_id}/viewport"
            f"?layer_id={layer_id}&x1={x1}&y1={y1}&x2={x2}&y2={y2}&lod={lod}",
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
