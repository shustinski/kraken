"""composition: frame management states."""
from __future__ import annotations
from datetime import datetime
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
    RepresentationId,
)

from kraken_hub.composition.parts.frame_cell_snapshot import FrameManagementState


class FrameManagementStatesMixin:
    def frame_management_states(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        representation_id: RepresentationId | str,
        *,
        as_of: datetime | None = None,
    ) -> tuple[FrameManagementState, ...]:
        return tuple(
            FrameManagementState(
                frame_id=item.frame_id,
                artifact_version_id=item.artifact_version_id,
                modified_at=item.modified_at,
                performer_color=item.performer_color,
                performer_initials=item.performer_initials,
                review_status=item.review_status,
            )
            for item in self.frame_cells(
                project_id,
                layer_id,
                representation_id,
                as_of=as_of,
            )
        )
