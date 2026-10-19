"""remote_client: issue and track performer work."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from kraken_manager.application.work_package import (
    WorkIssueRequest,
    latest_dataset_version,
    place_work_order,
)
from kraken_manager.domain.common import PerformerId
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model


class IssueWorkMixin:
    def work_orders(self) -> tuple[WorkOrder, ...]:
        orders: list[WorkOrder] = []
        for project in self.list_projects(include_archived=True):
            payload = self._call("GET", f"/api/v1/projects/{project.id}/work-orders")
            items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
            orders.extend(decode_model(WorkOrder, item) for item in items if isinstance(item, Mapping))
        return tuple(sorted(orders, key=lambda item: (item.updated_at, item.id), reverse=True))

    def issue_work(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        kind: str,
        placement: str,
        assignee_id: object,
        coordinates: tuple[tuple[int, int], ...] = (),
        image_representation_id: str = "",
        vector_representation_id: str = "",
        instructions: str = "",
        due_at: datetime | None = None,
        destination_parent: Path | str | None = None,
        idempotency_key: str,
        transport: str | None = None,
    ) -> WorkOrder:
        work_placement = WorkPlacement(placement)
        body = {
            "layer_id": str(layer_id),
            "kind": WorkKind(kind).value,
            "placement": work_placement.value,
            "assignee_id": str(assignee_id),
            "coordinates": [[int(x), int(y)] for x, y in coordinates],
            "image_representation_id": image_representation_id,
            "vector_representation_id": vector_representation_id,
            "instructions": instructions,
            "due_at": None if due_at is None else due_at.isoformat(),
            "transport": transport,
        }
        if work_placement is WorkPlacement.SERVER:
            payload = self._call(
                "POST",
                f"/api/v1/projects/{project_id}/work-orders",
                payload=body,
                idempotency_key=idempotency_key,
            )
            return decode_model(WorkOrder, payload)
        project = self.get_project(project_id)
        layer = next((item for item in self.list_layers(project_id) if str(item.id) == str(layer_id)), None)
        if project is None or layer is None:
            raise ValueError("Проект или слой не найден")
        performer = next((item for item in self.list_performers() if str(item.id) == str(assignee_id)), None)
        if performer is None or not performer.active:
            raise ValueError("Исполнитель не найден или архивирован")

        def active_version(series_id):
            return self.active_artifact_version(project.id, series_id)

        def read_bytes(version_id: str) -> bytes:
            version = self.artifact_version(project.id, version_id)
            if version is None or version.blob is None:
                raise ValueError("Файл не сохранён в проекте")
            return self.read_project_blob(project.id, version.sha256)

        order = place_work_order(
            WorkIssueRequest(
                project=project,
                layer=layer,
                kind=WorkKind(kind),
                placement=WorkPlacement.EXTERNAL,
                assignee_id=str(PerformerId(str(assignee_id))),
                created_by=str(principal.id),
                coordinates=tuple(coordinates),
                image_representation_id=image_representation_id,
                vector_representation_id=vector_representation_id,
                instructions=instructions,
                due_at=due_at,
                destination_parent=None if destination_parent is None else Path(destination_parent),
                server_directory_for=lambda _order_id: Path(),
                idempotency_key=idempotency_key,
                transport=transport or "files",
                remote=True,
                now=datetime.now(UTC),
            ),
            active_version=active_version,
            read_bytes=read_bytes,
            dataset_version=lambda: latest_dataset_version(
                self.list_artifact_series(project.id, layer_id=layer.id),
                lambda series_id: self.artifact_versions(project.id, series_id),
            ),
        )
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/work-orders",
            payload={"record": encode_model(order)},
            idempotency_key=idempotency_key,
        )
        return decode_model(WorkOrder, payload)

    def mark_work_running(self, order: WorkOrder, *, job_id: str = "", idempotency_key: str) -> WorkOrder:
        running = order.mark_running(job_id=job_id)
        payload = self._call(
            "POST",
            f"/api/v1/projects/{order.project_id}/work-orders",
            payload={"record": encode_model(running)},
            idempotency_key=idempotency_key,
        )
        return decode_model(WorkOrder, payload)

    def cancel_work_order(self, order: WorkOrder, *, idempotency_key: str) -> WorkOrder:
        payload = self._call(
            "POST",
            f"/api/v1/projects/{order.project_id}/work-orders/{order.id}/cancel",
            idempotency_key=idempotency_key,
        )
        return decode_model(WorkOrder, payload)

    def complete_work_order(self, order: WorkOrder, *, idempotency_key: str) -> WorkOrder:
        payload = self._call(
            "POST",
            f"/api/v1/projects/{order.project_id}/work-orders/{order.id}/complete",
            idempotency_key=idempotency_key,
        )
        return decode_model(WorkOrder, payload)
