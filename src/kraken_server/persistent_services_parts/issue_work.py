"""persistent_services: issue performer work into a server folder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import AuthorizationError
from kraken_manager.application.work_order_lifecycle import record_work_order
from kraken_manager.infrastructure.work_package import (
    WorkIssueRequest,
    latest_dataset_version,
    package_has_result,
    place_work_order,
    selection_from_coordinates,
)
from kraken_manager.domain.common import ArtifactVersionId, PerformerId, PrincipalId, validate_uuid
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkOrderState, WorkPlacement
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
        try:
            return self._issue_work_order(project_id, payload, context)
        except NotFoundError:
            raise
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ValidationError(str(exc)) from exc

    def _issue_work_order(self, project_id: str, payload: Mapping[str, Any], context: CommandContext) -> dict[str, Any]:
        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError("Project was not found")
        self._authorize_issued_work(project, context)
        record = payload.get("record")
        if isinstance(record, Mapping):
            try:
                order = decode_model(WorkOrder, record)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValidationError("Invalid work order record") from exc

            def register(projections):
                if str(order.project_id) != str(project.id):
                    raise ValidationError("Задание относится к другому проекту")
                current = projections.get_work_order(order.id)
                if current is not None:
                    self._check_work_revision(current, context)
                    if order.state is not WorkOrderState.RUNNING or current.state is not WorkOrderState.ISSUED:
                        raise ConflictError("Only an issued work order can be started through record registration")
                    expected = current.mark_running(job_id=order.related_job_id, at=order.updated_at)
                    if order != expected:
                        raise ValidationError("Work order metadata is immutable")
                    if order.related_job_id:
                        job = projections.get_plugin_job(order.related_job_id)
                        if job is None or job.project_id != project.id or job.layer_id != order.layer_id:
                            raise ValidationError("Work order job reference is outside its project or layer")
                    return expected
                if order.placement is not WorkPlacement.EXTERNAL or order.transport != "files":
                    raise ValidationError("Clients may register only external file packages")
                if order.created_by != context.actor_id or order.state is not WorkOrderState.ISSUED or order.revision != 0:
                    raise ValidationError("New work order identity or state is invalid")
                layer = projections.get_layer(order.layer_id)
                if layer is None or layer.project_id != project.id:
                    raise ValidationError("Layer belongs to another project")
                self._validate_work_references(
                    projections, project, layer,
                    image_representation_id=order.image_representation_id,
                    vector_representation_id=order.vector_representation_id,
                    dataset_version_id=order.dataset_version_id,
                )
                order.selection.validate_bounds(width=project.width, height=project.height)
                performer = None if self.performer_store is None else self.performer_store.get(order.assignee_id)
                if performer is None or not performer.active:
                    raise ValidationError("Исполнитель не найден или архивирован")
                if order.related_job_id:
                    raise ValidationError("Client cannot register unverified job references")
                if order.kind is WorkKind.VECTOR_REVIEW:
                    batch = projections.get_review_batch(order.related_batch_id)
                    if (
                        batch is None or batch.project_id != project.id or batch.layer_id != order.layer_id
                        or batch.assignee_id != order.assignee_id or batch.selection != order.selection
                        or len(batch.items) != order.frame_count or batch.state.value != "issued"
                    ):
                        raise ValidationError("Work order review reference is outside its project, scope or assignee")
                elif order.related_batch_id:
                    raise ValidationError("Only vector review work may link a review batch")
                return replace(order, idempotency_key=context.idempotency_key)

            return encode_model(self._record_work(project, context, f"register:{order.id}", register, command=payload))
        if str(payload.get("placement", "")) != WorkPlacement.SERVER.value:
            raise ValidationError("Сервер размещает работу только в серверной папке")
        layer = self.projections.get_layer(_layer_id(str(payload.get("layer_id", ""))))
        if layer is None or str(layer.project_id) != str(project.id):
            raise NotFoundError("Layer was not found")
        self._validate_work_references(
            self.projections, project, layer,
            image_representation_id=str(payload.get("image_representation_id", "")),
            vector_representation_id=str(payload.get("vector_representation_id", "")),
            require_image=WorkKind(str(payload.get("kind", ""))) is not WorkKind.TRAIN,
        )
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
            if self.events.find_by_idempotency_key(project.id, context.idempotency_key):
                return encode_model(self._record_work(project, context, "issue", lambda _projections: existing, command=payload))
            return encode_model(existing)
        if WorkKind(str(payload.get("kind", ""))) is WorkKind.VECTOR_REVIEW:
            return self._issue_server_review(project, layer, payload, context)
        coordinates = tuple((int(item[0]), int(item[1])) for item in payload.get("coordinates", ()))
        now = SystemClock().now()

        def read_bytes(version_id: str) -> bytes:
            version = self.projections.get_artifact_version(ArtifactVersionId(validate_uuid(version_id, field="version_id")))
            if version is None or version.blob is None:
                raise ValidationError("Файл не сохранён в проекте")
            return self.uow_factory.blobs.read(version.blob)

        def prepare_order(projections):
            # Allocate files only after transactional authorization and replay.
            # Read source versions through the transaction-bound projections.
            def current_version(series_id):
                return projections.get_active_artifact_version(series_id)

            def current_dataset():
                return latest_dataset_version(
                    projections.list_artifact_series(project.id, layer_id=layer.id),
                    projections.list_artifact_versions,
                )
            return place_work_order(
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
                    due_at=None if not payload.get("due_at") else datetime.fromisoformat(str(payload["due_at"])),
                    destination_parent=None,
                    server_directory_for=lambda order_id: self._server_work_directory(project.id, order_id),
                    idempotency_key=context.idempotency_key,
                    transport=payload.get("transport"),
                    remote=True,
                    now=now,
                ),
                active_version=current_version,
                read_bytes=read_bytes,
                dataset_version=current_dataset,
            )
        try:
            return encode_model(self._record_work(project, context, "issue", prepare_order, command=payload))
        except (ValueError, OSError) as exc:
            raise ValidationError(str(exc)) from exc

    def cancel_work_order(self, project_id: str, order_id: str, context: CommandContext) -> dict[str, Any]:
        order = self._stored_work_order(project_id, order_id)
        project = self.projections.get_project(order.project_id)
        def change(projections):
            current = projections.get_work_order(order.id)
            self._check_work_revision(current, context)
            return current.cancel(at=SystemClock().now())
        return encode_model(self._record_work(project, context, f"cancel:{order.id}", change))

    def complete_work_order(self, project_id: str, order_id: str, context: CommandContext) -> dict[str, Any]:
        order = self._stored_work_order(project_id, order_id)
        project = self.projections.get_project(order.project_id)
        def change(projections):
            current = projections.get_work_order(order.id)
            self._check_work_revision(current, context)
            if current.kind is WorkKind.VECTOR_REVIEW:
                raise ValidationError("Исправление векторов завершается принятием пакета проверки")
            if current.transport == "http":
                raise ValidationError("Результат HTTP-задания возвращается из Contour в проект")
            if current.placement is WorkPlacement.EXTERNAL:
                raise ValidationError("Import external results before completing work; server cannot inspect client paths")
            expected = self._server_work_directory(current.project_id, current.id).resolve()
            if Path(current.destination).resolve() != expected:
                raise ValidationError("Work order destination is outside its server directory")
            if not package_has_result(expected, current.kind):
                raise ValidationError("В папке работы ещё нет файла результата")
            return current.complete(at=SystemClock().now())
        return encode_model(self._record_work(project, context, f"complete:{order.id}", change))

    @staticmethod
    def _check_work_revision(order, context):
        if order is None:
            raise NotFoundError("Задание не найдено")
        if context.expected_revision is not None and context.expected_revision != order.revision:
            raise ConflictError("Work order revision changed")

    def _record_work(self, project, context, operation, change, *, command=None):
        principal = self.identities.get(PrincipalId(validate_uuid(context.actor_id, field="actor_id")))
        if principal is None:
            raise ForbiddenError("Учётная запись не найдена")
        try:
            return record_work_order(
                uow_factory=self.uow_factory, profiles=self.profiles, principal=principal,
                project_id=project.id, idempotency_key=context.idempotency_key,
                operation=operation, change=change, encode=encode_model, decode=decode_model,
                command={"project_id": str(project.id), "operation": operation,
                         "request": command, "expected_revision": context.expected_revision},
                gitlab_identity_verified=True,
            )
        except Exception as exc:
            from kraken_manager.domain.common import InvalidStateTransition
            from kraken_manager.infrastructure.filesystem.event_store import EventStreamConflict
            if isinstance(exc, (InvalidStateTransition, EventStreamConflict)):
                raise ConflictError(str(exc)) from exc
            self._translate_lifecycle_error(exc)
            raise

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
        return encode_model(self._record_work(project, context, "issue", lambda _projections: order, command=payload))

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
                gitlab_identity_verified=True,
            )
        except AuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc

    @staticmethod
    def _validate_work_references(
        projections, project, layer, *, image_representation_id="",
        vector_representation_id="", dataset_version_id="", require_image=False,
    ):
        """Reject foreign, archived or mistyped data before package creation."""
        from kraken_manager.domain.project import RepresentationKind, StructureState

        if layer.state is StructureState.ARCHIVED:
            raise ValidationError("Cannot issue work for an archived layer")
        if require_image and not image_representation_id:
            raise ValidationError("Image representation is required")
        references = {}
        for name, identifier, expected_kind in (
            ("image", image_representation_id, RepresentationKind.IMAGE),
            ("vector", vector_representation_id, RepresentationKind.VECTOR),
        ):
            if not identifier:
                continue
            representation = projections.get_representation(validate_uuid(identifier, field=f"{name}_representation_id"))
            if (
                representation is None or representation.project_id != project.id
                or representation.layer_id != layer.id or representation.kind is not expected_kind
                or representation.state is StructureState.ARCHIVED
            ):
                raise ValidationError(f"Work {name} representation is outside its project/layer, archived or mistyped")
            references[name] = representation
        vector = references.get("vector")
        if vector is not None and image_representation_id and vector.source_image_representation_id not in {
            None, image_representation_id,
        }:
            raise ValidationError("Vector representation belongs to another source image representation")
        if dataset_version_id:
            version = projections.get_artifact_version(validate_uuid(dataset_version_id, field="dataset_version_id"))
            series = None if version is None else projections.get_artifact_series(version.series_id)
            if (
                series is None or series.project_id != project.id or series.layer_id != layer.id
                or series.archived or version.media_type != "application/zip"
            ):
                raise ValidationError("Work dataset version is outside its project/layer or is not a dataset archive")

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
