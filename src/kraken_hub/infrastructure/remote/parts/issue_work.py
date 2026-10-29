"""remote_client: issue and track performer work."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from kraken_manager.infrastructure.work_package import (
    WorkIssueRequest,
    external_package_directory,
    latest_dataset_version,
    place_work_order,
)
from kraken_manager.domain.common import PerformerId
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model


class IssueWorkMixin:
    def plan_legacy_work_order_migration(self, *, principal, project_id):
        return self._call("GET", f"/api/v1/projects/{project_id}/work-orders/legacy-migration")

    def migrate_legacy_work_orders(self, *, principal, project_id, expected_plan_digest, idempotency_key):
        return self._call("POST", f"/api/v1/projects/{project_id}/work-orders/legacy-migration",
            payload={"expected_plan_digest": expected_plan_digest}, idempotency_key=idempotency_key,
            timeout=max(180, self._http.timeout))

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
        companion_representations: tuple[tuple[str, str], ...] | None = None,
    ) -> WorkOrder:
        work_placement = WorkPlacement(placement)
        companions = (
            self._review_companion_plan(project_id, vector_representation_id, image_representation_id)
            if companion_representations is None
            else tuple(companion_representations)
        )
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
            "companion_representations": [
                {"folder": folder, "representation_id": representation_id}
                for folder, representation_id in companions
            ],
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
        if WorkKind(kind) is WorkKind.VECTOR_REVIEW:
            if destination_parent is None:
                raise ValueError("Выберите каталог внешнего накопителя")
            batch = self.create_review_batch(
                principal=principal, project_id=project.id, layer_id=layer.id,
                image_representation_id=image_representation_id,
                vector_representation_id=vector_representation_id, coordinates=coordinates,
                assignee_id=assignee_id, instructions=instructions, due_at=due_at,
                idempotency_key=f"{idempotency_key}:review",
            )
            now = datetime.now(UTC)
            destination = external_package_directory(
                Path(destination_parent), project_name=project.name, kind=WorkKind.VECTOR_REVIEW, now=now,
            )
            destination.rmdir()  # The signed-package exporter owns destination creation.
            issued = self.export_review_batch(
                principal=principal, batch=batch, destination=destination,
                idempotency_key=f"{idempotency_key}:export",
            )
            if companions:
                from kraken_manager.infrastructure.work_package import write_companion_images

                def read_bytes(version_id: str) -> bytes:
                    version = self.artifact_version(project.id, version_id)
                    if version is None or version.blob is None:
                        raise ValueError("Файл не сохранён в проекте")
                    return self.read_project_blob(project.id, version.sha256)

                write_companion_images(
                    destination,
                    project=project,
                    companions=companions,
                    selection=issued.selection,
                    active_version=lambda series_id: self.active_artifact_version(project.id, series_id),
                    read_bytes=read_bytes,
                )
            order = WorkOrder.create(
                order_id=str(issued.id), project_id=project.id, layer_id=layer.id,
                kind=WorkKind.VECTOR_REVIEW, placement=WorkPlacement.EXTERNAL,
                assignee_id=str(assignee_id), created_by=principal.id, destination=str(destination),
                frame_count=len(issued.items), selection=issued.selection, instructions=instructions,
                due_at=due_at, transport="files", image_representation_id=image_representation_id,
                vector_representation_id=vector_representation_id, capability="frames.vectorize.v1",
                launch_mode="review-package", related_batch_id=str(issued.id),
                idempotency_key=idempotency_key, created_at=now,
            )
            payload = self._call(
                "POST", f"/api/v1/projects/{project_id}/work-orders",
                payload={"record": encode_model(order)}, idempotency_key=idempotency_key,
            )
            return decode_model(WorkOrder, payload)

        def active_version(series_id):
            return self.active_artifact_version(project.id, series_id)

        def read_bytes(version_id: str) -> bytes:
            version = self.artifact_version(project.id, version_id)
            if version is None or version.blob is None:
                raise ValueError("Файл не сохранён в проекте")
            return self.read_project_blob(project.id, str(version.id))

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
                companion_representations=companions,
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

    def _review_companion_plan(self, project_id, vector_representation_id: str, primary_image_id: str):
        from kraken_manager.domain.layer_board import companion_export_plan, review_image_ids
        from kraken_manager.domain.project import RepresentationKind

        if not vector_representation_id or not hasattr(self, "history"):
            return ()
        ordered = review_image_ids(self.history(project_id), (str(vector_representation_id),))
        names = {}
        for layer in self.list_layers(project_id):
            for representation in self.list_representations(project_id, layer.id):
                if representation.kind is RepresentationKind.IMAGE:
                    names[str(representation.id)] = (layer.name, representation.name)
        return companion_export_plan(ordered, primary_image_id=str(primary_image_id or ""), names_by_id=names)

    def mark_work_running(self, order: WorkOrder, *, job_id: str = "", idempotency_key: str, principal=None) -> WorkOrder:
        running = order.mark_running(job_id=job_id)
        payload = self._call(
            "POST",
            f"/api/v1/projects/{order.project_id}/work-orders",
            payload={"record": encode_model(running)},
            idempotency_key=idempotency_key,
            if_match=order.revision,
        )
        return decode_model(WorkOrder, payload)

    def cancel_work_order(self, order: WorkOrder, *, idempotency_key: str, principal=None) -> WorkOrder:
        payload = self._call(
            "POST",
            f"/api/v1/projects/{order.project_id}/work-orders/{order.id}/cancel",
            idempotency_key=idempotency_key,
            if_match=order.revision,
        )
        return decode_model(WorkOrder, payload)

    def complete_work_order(self, order: WorkOrder, *, idempotency_key: str, principal=None) -> WorkOrder:
        payload = self._call(
            "POST",
            f"/api/v1/projects/{order.project_id}/work-orders/{order.id}/complete",
            idempotency_key=idempotency_key,
            if_match=order.revision,
        )
        return decode_model(WorkOrder, payload)
