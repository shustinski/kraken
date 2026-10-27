"""composition: issue performer work to removable media or a server folder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import AuthorizationError
from kraken_manager.application.work_order_lifecycle import record_work_order
from kraken_manager.infrastructure.work_package import (
    WorkIssueRequest,
    latest_dataset_version,
    package_has_result,
    place_work_order,
    representation_of,
    selection_from_coordinates,
)
from kraken_manager.domain.common import PerformerId, new_uuid
from kraken_manager.domain.identity import Permission, Principal
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkOrderState, WorkPlacement
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model


class IssueWorkMixin:
    def work_orders(self) -> tuple[WorkOrder, ...]:
        orders: dict[str, WorkOrder] = {}
        for project in self.list_projects(include_archived=True):
            for order in self._projection(project.id).list_work_orders(project.id):
                orders[str(order.id)] = order
        return tuple(sorted(orders.values(), key=lambda item: (item.updated_at, item.id), reverse=True))

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
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Проект не найден")
        layer = next((item for item in self.list_layers(project.id) if str(item.id) == str(layer_id)), None)
        if layer is None:
            raise ValueError("Слой не найден")
        self._authorize_work(principal, project)
        existing = self._work_order_for_key(project.id, idempotency_key)
        if existing is not None:
            return existing
        performer = self.performers.get(PerformerId(str(assignee_id)))
        if performer is None or not performer.active:
            raise ValueError("Исполнитель не найден или архивирован")
        work_kind = WorkKind(kind)
        work_placement = WorkPlacement(placement)
        if work_kind is WorkKind.VECTOR_REVIEW:
            return self._issue_vector_review(
                principal=principal,
                project=project,
                layer=layer,
                placement=work_placement,
                assignee_id=str(assignee_id),
                coordinates=tuple(coordinates),
                image_representation_id=image_representation_id,
                vector_representation_id=vector_representation_id,
                instructions=instructions,
                due_at=due_at,
                destination_parent=None if destination_parent is None else Path(destination_parent),
                idempotency_key=idempotency_key,
            )
        representations = self.list_representations(project.id, layer.id)
        if work_kind is not WorkKind.TRAIN:
            image = representation_of(representations, image_representation_id, RepresentationKind.IMAGE)
            if image is None:
                raise ValueError("Выберите репрезентацию изображений этого слоя")
        now = self.clock.now()

        def active_version(series_id):
            return self._projection(project.id).get_active_artifact_version(series_id)

        def read_bytes(version_id: str) -> bytes:
            version = self.artifact_version(project.id, version_id)
            if version is None or version.blob is None:
                raise ValueError("Файл не сохранён в проекте")
            return self.read_project_blob(project.id, version.sha256)

        def dataset():
            return latest_dataset_version(
                self.list_artifact_series(project.id, layer_id=layer.id),
                lambda series_id: self.artifact_versions(project.id, series_id),
            )

        order = place_work_order(
            WorkIssueRequest(
                project=project,
                layer=layer,
                kind=work_kind,
                placement=work_placement,
                assignee_id=str(assignee_id),
                created_by=str(principal.id),
                coordinates=tuple(coordinates),
                image_representation_id=image_representation_id,
                vector_representation_id=vector_representation_id,
                instructions=instructions,
                due_at=due_at,
                destination_parent=None if destination_parent is None else Path(destination_parent),
                server_directory_for=lambda order_id: self._server_work_directory(project.id, order_id),
                idempotency_key=idempotency_key,
                transport=transport,
                remote=bool(getattr(self, "is_remote_project", lambda _value: False)(project.id)),
                now=now,
            ),
            active_version=active_version,
            read_bytes=read_bytes,
            dataset_version=dataset,
        )
        return self._record_work_order(principal, order, idempotency_key, "issue")

    def mark_work_running(self, order: WorkOrder, *, job_id: str = "", idempotency_key: str, principal=None) -> WorkOrder:
        return self._mutate_work_order(
            order, principal, idempotency_key, "start",
            lambda current: current.mark_running(job_id=job_id, at=self.clock.now()),
        )

    def cancel_work_order(self, order: WorkOrder, *, idempotency_key: str, principal=None) -> WorkOrder:
        return self._mutate_work_order(
            order, principal, idempotency_key, "cancel", lambda current: current.cancel(at=self.clock.now()),
        )

    def complete_work_order(self, order: WorkOrder, *, idempotency_key: str, principal=None) -> WorkOrder:
        def change(current):
            if current.kind is WorkKind.VECTOR_REVIEW:
                raise ValueError("Исправление векторов завершается принятием пакета проверки")
            if current.transport == "http":
                raise ValueError("Результат HTTP-задания возвращается из Contour в проект")
            if not package_has_result(Path(current.destination), current.kind):
                raise ValueError("В папке работы ещё нет файла результата")
            return current.complete(at=self.clock.now())
        return self._mutate_work_order(order, principal, idempotency_key, "complete", change)

    def _record_work_order(self, principal, order, key, operation):
        return record_work_order(
            uow_factory=self._uow(str(order.project_id)), profiles=self.profiles, principal=principal,
            project_id=order.project_id, idempotency_key=key, operation=f"{operation}:{order.id}",
            command={"order_id": order.id},
            change=lambda _projections: order, encode=encode_model, decode=decode_model,
        )

    def _mutate_work_order(self, order, principal, key, operation, change):
        from kraken_manager.application.errors import ConcurrencyError
        # Preserve the local Python API used before principals were explicit.
        # Hub always passes the actual signed-in principal.
        principal = principal or self.identities.get(order.created_by)
        if principal is None:
            raise ValueError("Учётная запись не найдена")
        def checked(projections):
            current = projections.get_work_order(order.id)
            if current is None:
                raise ValueError("Задание не найдено")
            if current.revision != order.revision:
                raise ConcurrencyError("Work order revision changed")
            if current.state in {WorkOrderState.COMPLETED, WorkOrderState.CANCELLED}:
                raise ValueError("Задание уже закрыто")
            return change(current)
        return record_work_order(
            uow_factory=self._uow(str(order.project_id)), profiles=self.profiles, principal=principal,
            project_id=order.project_id, idempotency_key=key, operation=f"{operation}:{order.id}",
            command={"order_id": order.id, "revision": order.revision},
            change=checked, encode=encode_model, decode=decode_model,
        )

    def _issue_vector_review(self, **kwargs) -> WorkOrder:
        from kraken_manager.infrastructure.work_package import external_package_directory

        project = kwargs["project"]
        layer = kwargs["layer"]
        placement: WorkPlacement = kwargs["placement"]
        coordinates = kwargs["coordinates"]
        selection = selection_from_coordinates(tuple(coordinates))
        selection.validate_bounds(width=project.width, height=project.height)
        now = self.clock.now()
        if placement is WorkPlacement.EXTERNAL:
            parent = kwargs["destination_parent"]
            if parent is None:
                raise ValueError("Выберите каталог внешнего накопителя")
            destination = external_package_directory(
                parent, project_name=project.name, kind=WorkKind.VECTOR_REVIEW, now=now
            )
            destination.rmdir()
        else:
            destination = self._server_work_directory(project.id, new_uuid())
        batch = self.create_review_batch(
            principal=kwargs["principal"],
            project_id=project.id,
            layer_id=layer.id,
            image_representation_id=kwargs["image_representation_id"],
            vector_representation_id=kwargs["vector_representation_id"],
            coordinates=coordinates,
            assignee_id=kwargs["assignee_id"],
            instructions=kwargs["instructions"],
            due_at=kwargs["due_at"],
            idempotency_key=f"{kwargs['idempotency_key']}:review",
        )
        issued = self.export_review_batch(
            principal=kwargs["principal"],
            batch=batch,
            destination=destination,
            idempotency_key=f"{kwargs['idempotency_key']}:export",
        )
        order = WorkOrder.create(
            project_id=project.id,
            layer_id=layer.id,
            kind=WorkKind.VECTOR_REVIEW,
            placement=placement,
            assignee_id=kwargs["assignee_id"],
            created_by=kwargs["principal"].id,
            destination=str(destination),
            frame_count=len(tuple(selection.iter_coordinates())),
            selection=selection,
            instructions=kwargs["instructions"],
            due_at=kwargs["due_at"],
            transport="files",
            image_representation_id=kwargs["image_representation_id"],
            vector_representation_id=kwargs["vector_representation_id"],
            capability="frames.vectorize.v1",
            launch_mode="review-package",
            related_batch_id=str(issued.id),
            idempotency_key=kwargs["idempotency_key"],
            created_at=now,
        )
        return self._record_work_order(kwargs["principal"], order, kwargs["idempotency_key"], "issue")

    def _authorize_work(self, principal: Principal, project) -> None:
        with self._uow(str(project.id))() as unit:
            roles = unit.acl.roles_for(project.id, principal.id)
        try:
            AuthorizationPolicy().require(
                principal=principal,
                storage=self.profiles.get(project.storage_profile),
                permission=Permission.ASSIGN_WORK,
                roles=roles,
            )
        except AuthorizationError as exc:
            raise ValueError(str(exc)) from exc

    def _server_work_directory(self, project_id: object, order_id: str) -> Path:
        workspace = self.project_workspace(project_id)
        derived = getattr(workspace, "derived_project_dir", "") if workspace is not None else ""
        if derived and Path(derived).is_dir():
            return Path(derived) / "work-orders" / order_id
        return self.data_dir / "server-work" / str(project_id) / order_id

    def _work_order_for_key(self, project_id: object, idempotency_key: str) -> WorkOrder | None:
        if not idempotency_key:
            return None
        for order in self._projection(project_id).list_work_orders(project_id):
            if order.idempotency_key == idempotency_key:
                return order
        return None

    def _require_work_order(self, order: WorkOrder) -> WorkOrder:
        current = self._projection(order.project_id).get_work_order(order.id)
        if current is None:
            raise ValueError("Задание не найдено")
        if current.state in {WorkOrderState.COMPLETED, WorkOrderState.CANCELLED}:
            raise ValueError("Задание уже закрыто")
        return current
