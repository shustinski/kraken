"""composition: create external layer."""
from __future__ import annotations
from pathlib import Path
from kraken_manager.workspace import (
    LayerFileBinding,
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


class CreateExternalLayerMixin:
    def create_external_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        image_directory: Path | str,
        ssc_directory: Path | str | None,
        prv_directory: Path | str | None,
        idempotency_key: str,
    ) -> tuple[Layer, LayerFileBinding, Representation]:
        safe_name = validate_workspace_name(name, field_name="Название слоя")
        reserved_layer_id = LayerId(new_uuid())
        binding = self.workspace_files.bind_external_layer(
            project_id=str(project.id),
            layer_id=str(reserved_layer_id),
            layer_name=safe_name,
            image_directory=image_directory,
            ssc_directory=ssc_directory,
            prv_directory=prv_directory,
            maximum_frames=project.frame_count,
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
