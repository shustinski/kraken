"""remote_client: list layers."""
from __future__ import annotations
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    StructureState,
)
from kraken_hub.application.workspace import (
    project_from_server_dict,
)

from kraken_hub.infrastructure.remote.parts.extract_review_archive import RemoteServerError


class ListLayersMixin:
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
        del (
            as_of
        )  # server temporal get is history-based; catalog uses current projection
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
        project = project_from_server_dict(
            payload if isinstance(payload, Mapping) else {}
        )
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

    def list_layers(
        self, project_id: object, *, include_archived: bool = False
    ) -> tuple[Layer, ...]:
        suffix = "?include_archived=true" if include_archived else ""
        payload = self._call("GET", f"/api/v1/projects/{project_id}/layers{suffix}")
        items = (
            payload
            if isinstance(payload, list)
            else payload.get("items", [])
            if isinstance(payload, Mapping)
            else []
        )
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
