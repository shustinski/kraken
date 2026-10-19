"""composition: materialize representation inputs."""
from __future__ import annotations
from typing import TYPE_CHECKING
from kraken_manager.application.imports import ImportMappingMode
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
    Representation,
)


class MaterializeRepresentationInputsMixin:
    def materialize_representation_inputs(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
    ) -> tuple[FrameCellSnapshot, ...]:
        """Create stable project versions for directory-backed plugin inputs."""

        existing = self.frame_cells(project.id, layer.id, representation.id)
        if existing:
            return existing
        source = str(representation.source or "").strip()
        if not source:
            raise ValueError("У исходного слоя не указан каталог с изображениями")
        plan = self.plan_import_directory(
            project=project,
            directory=source,
            mode=ImportMappingMode.ROW_MAJOR_SUFFIX,
        )
        if not plan.ready or not plan.items:
            details = "; ".join(issue.message for issue in plan.issues)
            message = "Не удалось сопоставить изображения с кадрами проекта"
            raise ValueError(f"{message}: {details}" if details else message)
        self.commit_managed_import(
            principal=principal,
            project=project,
            layer=layer,
            representation=representation,
            plan=plan,
            idempotency_key=f"agent-inputs:{representation.id}",
        )
        return self.frame_cells(project.id, layer.id, representation.id)

if TYPE_CHECKING:
    from kraken_hub.composition_parts.frame_cell_snapshot import FrameCellSnapshot
