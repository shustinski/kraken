"""composition: create layer disk."""
from __future__ import annotations
from kraken_manager.workspace import (
    ImageConversionSettings,
    LayerFileBinding,
    LayerSourceScan,
    WorkspaceValidationError,
    layer_binding_to_dict,
    validate_workspace_name,
)
from kraken_manager.domain.common import (
    LayerId,
    new_uuid,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
)


class CreateLayerDiskMixin:
    def create_layer_from_disk(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        scan: LayerSourceScan,
        conversion: ImageConversionSettings,
        idempotency_key: str,
        progress=None,
        cancelled=None,
    ) -> tuple[Layer, LayerFileBinding, Representation]:
        workspace = self.project_workspace(project.id)
        if workspace is None:
            raise WorkspaceValidationError("project has no two-root workspace")
        safe_name = validate_workspace_name(name, field_name="Название слоя")
        reserved_layer_id = LayerId(new_uuid())
        binding = self.workspace_files.import_layer(
            project=workspace,
            layer_id=str(reserved_layer_id),
            layer_name=safe_name,
            scan=scan,
            conversion=conversion,
            progress=progress,
            cancelled=cancelled,
        )
        try:
            current_project = self.get_project(project.id) or project
            layer = self.create_layer(
                principal=principal,
                project=current_project,
                name=safe_name,
                layer_type=layer_type,
                order=order,
                idempotency_key=idempotency_key,
                layer_id=reserved_layer_id,
            )
            latest_project = self.get_project(project.id) or current_project
            representation = self.create_representation(
                principal=principal,
                project=latest_project,
                layer=layer,
                name="Исходные изображения",
                kind=RepresentationKind.IMAGE,
                idempotency_key=f"{idempotency_key}:representation",
                source=binding.image_directory,
                active=True,
                purpose=RepresentationPurpose.SOURCE,
            )
            self.materialize_representation_inputs(
                principal=principal,
                project=latest_project,
                layer=layer,
                representation=representation,
            )
            self._record_workspace_event(
                principal=principal,
                project_id=project.id,
                stream_id=f"layer-files:{layer.id}",
                event_type="LayerFileBoundV1",
                payload=layer_binding_to_dict(binding),
                idempotency_key=f"{idempotency_key}:file-binding",
            )
            return layer, binding, representation
        except Exception:
            stored = self.layer_file_binding(project.id, reserved_layer_id)
            if stored is not None:
                self.workspace_files.remove_managed_layer_layout(stored, workspace)
                self.workspace_registry.remove_layer(str(project.id), str(reserved_layer_id))
            current_project = self.get_project(project.id) or project
            current_layer = next(
                (item for item in self.list_layers(project.id) if item.id == reserved_layer_id),
                None,
            )
            if current_layer is not None:
                try:
                    self.archive_layer(
                        principal=principal,
                        project=current_project,
                        layer=current_layer,
                        idempotency_key=f"{idempotency_key}:compensate",
                    )
                except Exception:
                    pass
            raise
