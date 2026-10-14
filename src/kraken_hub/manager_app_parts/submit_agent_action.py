"""manager_app: submit agent action."""
from __future__ import annotations
from collections.abc import Mapping
from uuid import uuid4
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose


class SubmitAgentActionMixin:
    def _submit_agent_action(
        self,
        *,
        layer_id: str,
        source_representation_id: str,
        capability: str,
        parameters: Mapping[str, object],
    ):
        from kraken_hub.manager_app import AgentPluginGateway
        workspace = self._workspace
        project = self.service.get_project(self._project_id)
        if workspace is None or project is None or not source_representation_id:
            raise ValueError("Не выбраны исходные изображения")
        is_remote = bool(
            getattr(self.service, "is_remote_project", lambda _value: False)(project.id)
        )
        interactive_registry = None
        if is_remote and capability in {"frames.vectorize.v1", "frames.dataset.prepare.v1", "dataset.model.train.v1"}:
            interactive_registry = self.agent_runtime.interactive_registry(capability)
            parameters = {**parameters, "execution_host": "desktop", "input_transport": "server"}
        elif is_remote and capability == "frames.binary-segment.v2":
            interactive_registry = self.agent_runtime.execution_registry(capability)
            parameters = {**parameters, "execution_host": "desktop", "input_transport": "server"}
        capabilities = frozenset()
        if not is_remote:
            capabilities = self.agent_runtime.ensure_started()
            if capability not in capabilities:
                raise ValueError(f"Установленные плагины не поддерживают {capability}")
        coordinates = workspace.matrix_view.selected_coordinates(
            maximum=self.service.profile.capabilities.max_frames
        )
        if not coordinates:
            coordinates = tuple(
                (cell.x, cell.y)
                for cell in self.service.frame_cells(
                    project.id,
                    layer_id,
                    source_representation_id,
                )
            )
        if not coordinates:
            raise ValueError("В выбранном слое нет изображений для запуска задания")
        coordinate_by_frame = {
            str(project.frame_id_at(x, y)): (x, y)
            for x, y in coordinates
        }
        if capability in {"frames.vectorize.v1", "frames.binary-segment.v2"}:
            vector_output = capability == "frames.vectorize.v1"
            representations = self.service.list_representations(project.id, layer_id)
            targets = [
                item
                for item in representations
                if item.kind is (RepresentationKind.VECTOR if vector_output else RepresentationKind.IMAGE)
                and (vector_output or item.purpose is RepresentationPurpose.BINARY)
                and str(item.source_image_representation_id or "") == source_representation_id
            ]
            selected_id = str(workspace.vector_representation_combo.currentData() or "") if vector_output else ""
            target = next(
                (item for item in targets if str(item.id) == selected_id),
                targets[0] if targets else None,
            )
            if target is None:
                layer = next(
                    (item for item in self.service.list_layers(project.id) if str(item.id) == layer_id),
                    None,
                )
                source = next(
                    (item for item in representations if str(item.id) == source_representation_id),
                    None,
                )
                if layer is None or source is None or source.kind is not RepresentationKind.IMAGE:
                    raise ValueError("Исходный слой изображений больше не доступен")
                target = self.service.create_representation(
                    principal=self.session.principal,
                    project=project,
                    layer=layer,
                    name=f"CIF — {source.name}" if vector_output else f"NeuralImage — {source.name}",
                    kind=RepresentationKind.VECTOR if vector_output else RepresentationKind.IMAGE,
                    purpose=RepresentationPurpose.VECTOR if vector_output else RepresentationPurpose.BINARY,
                    source_image_representation_id=source.id,
                    idempotency_key=str(uuid4()),
                    active=True,
                )
            target_id = str(target.id)
        else:
            target_id = source_representation_id

        def version_path(version_id):
            return self.service.managed_artifact_path(project.id, version_id)

        def frame_coordinate(frame_id):
            coordinate = coordinate_by_frame.get(str(frame_id))
            if coordinate is None:
                raise ValueError(f"Кадр {frame_id} не входит в выбранный набор")
            return coordinate

        def media_type(version_id):
            version = self.service.artifact_version(project.id, version_id)
            if version is None:
                raise ValueError(f"Версия {version_id} больше не доступна")
            return version.media_type

        gateway = None
        if not is_remote:
            gateway = AgentPluginGateway(
                base_url=self.agent_runtime.base_url,
                token=self.agent_runtime.token,
                staging_root=self.service.data_dir / "agent" / "staging",
                source_for_version=version_path,
                coordinate_for_frame=frame_coordinate,
                media_type_for_version=media_type,
                capabilities=capabilities,
                v2_capabilities=frozenset(
                    operation
                    for operation, protocol in self.agent_runtime.protocol_by_capability.items()
                    if protocol == "2.0"
                ),
            )
            self._agent_gateway = gateway
        job = self.service.submit_plugin_job(
            principal=self.session.principal,
            gateway=gateway,
            project_id=project.id,
            layer_id=layer_id,
            source_representation_id=source_representation_id,
            target_representation_id=target_id,
            coordinates=coordinates,
            capability=capability,
            parameters=parameters,
            idempotency_key=str(uuid4()),
        )
        if interactive_registry is not None:
            self._run_remote_interactive_job(job, registry=interactive_registry)
        return job
