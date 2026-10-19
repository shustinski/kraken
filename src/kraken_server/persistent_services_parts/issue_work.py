"""persistent_services: issue performer work into a server folder."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import AuthorizationError
from kraken_manager.application.work_package import (
    WorkIssueRequest,
    latest_dataset_version,
    package_has_result,
    place_work_order,
    selection_from_coordinates,
    write_work_manifest,
)
from kraken_manager.domain.common import ArtifactVersionId, PerformerId, PrincipalId, validate_uuid
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model
from kraken_server.services import CommandContext, ConflictError, ForbiddenError, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import SystemClock, _layer_id, _project_id


class IssueWorkMixin:
    def list_work_orders(self, project_id: str) -> list[dict[str, Any]]:
        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError("Project was not found")
        return [encode_model(order) for order in self.projections.list_work_orders(project.id)]

    def issue_work_order(self, project_id: str, payload: Mapping[str, Any], context: CommandContext) -> dict[str, Any]:
        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError("Project was not found")
        self._authorize_issued_work(project, context)
        record = payload.get("record")
        if isinstance(record, Mapping):
            order = decode_model(WorkOrder, record)
            if str(order.project_id) != str(project.id):
                raise ValidationError("Задание относится к другому проекту")
            self.projections.save_work_order(order)
            return encode_model(order)
        if str(payload.get("placement", "")) != WorkPlacement.SERVER.value:
            raise ValidationError("Сервер размещает работу только в серверной папке")
        layer = self.projections.get_layer(_layer_id(str(payload.get("layer_id", ""))))
        if layer is None or str(layer.project_id) != str(project.id):
            raise NotFoundError("Layer was not found")
        if self.performer_store is None:
            raise ConflictError("Server performer store is not configured")
        performer = self.performer_store.get(PerformerId(str(payload.get("assignee_id", ""))))
        if performer is None or not performer.active:
            raise ValidationError("Исполнитель не найден или архивирован")
        existing = next(
            (
                order
                for order in self.projections.list_work_orders(project.id)
                if order.idempotency_key and order.idempotency_key == context.idempotency_key
            ),
            None,
        )
        if existing is not None:
            return encode_model(existing)
        if WorkKind(str(payload.get("kind", ""))) is WorkKind.VECTOR_REVIEW:
            return self._issue_server_review(project, layer, payload, context)
        coordinates = tuple((int(item[0]), int(item[1])) for item in payload.get("coordinates", ()))
        now = SystemClock().now()

        def active_version(series_id):
            return self.projections.get_active_artifact_version(series_id)

        def read_bytes(version_id: str) -> bytes:
            version = self.projections.get_artifact_version(ArtifactVersionId(validate_uuid(version_id, field="version_id")))
            if version is None or version.blob is None:
                raise ValidationError("Файл не сохранён в проекте")
            return self.uow_factory.blobs.read(version.blob)

        def dataset():
            return latest_dataset_version(
                self.projections.list_artifact_series(project.id, layer_id=layer.id),
                self.projections.list_artifact_versions,
            )

        try:
            order = place_work_order(
                WorkIssueRequest(
                    project=project,
                    layer=layer,
                    kind=WorkKind(str(payload.get("kind", ""))),
                    placement=WorkPlacement.SERVER,
                    assignee_id=str(performer.id),
                    created_by=context.actor_id,
                    coordinates=coordinates,
                    image_representation_id=str(payload.get("image_representation_id", "")),
                    vector_representation_id=str(payload.get("vector_representation_id", "")),
                    instructions=str(payload.get("instructions", "")),
                    due_at=None,
                    destination_parent=None,
                    server_directory_for=lambda order_id: self._server_work_directory(project.id, order_id),
                    idempotency_key=context.idempotency_key,
                    transport=payload.get("transport"),
                    remote=True,
                    now=now,
                ),
                active_version=active_version,
                read_bytes=read_bytes,
                dataset_version=dataset,
            )
        except (ValueError, OSError) as exc:
            raise ValidationError(str(exc)) from exc
        self.projections.save_work_order(order)
        return encode_model(order)

    def cancel_work_order(self, project_id: str, order_id: str, context: CommandContext) -> dict[str, Any]:
        del context
        order = self._stored_work_order(project_id, order_id)
        cancelled = order.cancel(at=SystemClock().now())
        self.projections.save_work_order(cancelled)
        return encode_model(cancelled)

    def complete_work_order(self, project_id: str, order_id: str, context: CommandContext) -> dict[str, Any]:
        del context
        order = self._stored_work_order(project_id, order_id)
        if order.kind is WorkKind.VECTOR_REVIEW:
            raise ValidationError("Исправление векторов завершается принятием пакета проверки")
        if order.transport == "http":
            raise ValidationError("Результат HTTP-задания возвращается из Contour в проект")
        if not package_has_result(Path(order.destination), order.kind):
            raise ValidationError("В папке работы ещё нет файла результата")
        completed = order.complete(at=SystemClock().now())
        self.projections.save_work_order(completed)
        return encode_model(completed)

    def _issue_server_review(self, project, layer, payload: Mapping[str, Any], context: CommandContext) -> dict[str, Any]:
        coordinates = tuple((int(item[0]), int(item[1])) for item in payload.get("coordinates", ()))
        selection = selection_from_coordinates(coordinates)
        selection.validate_bounds(width=project.width, height=project.height)
        due_raw = payload.get("due_at")
        due_at = None if not due_raw else datetime.fromisoformat(str(due_raw))
        created = self.create_review_batch(
            str(project.id),
            {
                "layer_id": str(layer.id),
                "image_representation_id": str(payload.get("image_representation_id", "")),
                "vector_representation_id": str(payload.get("vector_representation_id", "")),
                "coordinates": [[x, y] for x, y in coordinates],
                "assignee_id": str(payload.get("assignee_id", "")),
                "instructions": str(payload.get("instructions", "")),
                "due_at": None if due_at is None else due_at.isoformat(),
            },
            CommandContext(context.actor_id, f"{context.idempotency_key}:review", layer.revision),
        )
        destination = self._server_work_directory(project.id, str(created["id"]))
        self.export_review_package(
            str(project.id),
            str(created["id"]),
            str(destination),
            CommandContext(context.actor_id, f"{context.idempotency_key}:export", int(created["revision"])),
        )
        now = SystemClock().now()
        order = WorkOrder.create(
            project_id=project.id,
            layer_id=layer.id,
            kind=WorkKind.VECTOR_REVIEW,
            placement=WorkPlacement.SERVER,
            assignee_id=str(payload.get("assignee_id", "")),
            created_by=context.actor_id,
            destination=str(destination),
            frame_count=len(coordinates),
            selection=selection,
            instructions=str(payload.get("instructions", "")),
            due_at=due_at,
            transport="files",
            image_representation_id=str(payload.get("image_representation_id", "")),
            vector_representation_id=str(payload.get("vector_representation_id", "")),
            capability="frames.vectorize.v1",
            launch_mode="review-package",
            related_batch_id=str(created["id"]),
            idempotency_key=context.idempotency_key,
            created_at=now,
        )
        write_work_manifest(destination, order, ())
        self.projections.save_work_order(order)
        return encode_model(order)

    def _stored_work_order(self, project_id: str, order_id: str) -> WorkOrder:
        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError("Project was not found")
        order = self.projections.get_work_order(validate_uuid(order_id, field="work_order_id"))
        if order is None or str(order.project_id) != str(project.id):
            raise NotFoundError("Задание не найдено")
        return order

    def _authorize_issued_work(self, project, context: CommandContext) -> None:
        principal = self.identities.get(PrincipalId(validate_uuid(context.actor_id, field="actor_id")))
        if principal is None:
            raise ForbiddenError("Учётная запись не найдена")
        storage = self.profiles.get(project.storage_profile)
        roles = self.identities.roles_for(project.id, principal.id)
        try:
            AuthorizationPolicy().require(
                principal=principal,
                storage=storage,
                permission=Permission.ASSIGN_WORK,
                roles=roles,
            )
        except AuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc

    def _server_work_directory(self, project_id: object, order_id: str) -> Path:
        files = getattr(self, "workspace_files", None)
        binding = None if files is None else files.registry.get_project(str(project_id))
        derived = "" if binding is None else binding.derived_project_dir
        if derived and Path(derived).is_dir():
            return Path(derived) / "work-orders" / order_id
        root = getattr(self, "derived_root", None)
        if root:
            return Path(root) / str(project_id) / "work-orders" / order_id
        raise ValidationError("Серверная папка проекта не настроена")
