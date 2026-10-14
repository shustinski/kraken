"""composition: frame cells."""
from __future__ import annotations
from datetime import datetime
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
    RepresentationId,
)
from kraken_manager.domain.project import (
    RepresentationKind,
)

from kraken_hub.composition_parts.frame_cell_snapshot import FrameCellSnapshot


class FrameCellsMixin:
    def frame_cells(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        representation_id: RepresentationId | str,
        *,
        as_of: datetime | None = None,
    ) -> tuple[FrameCellSnapshot, ...]:
        project = self.get_project(project_id, as_of=as_of)
        if project is None:
            return ()
        projection = self._projection(project.id)
        representation = projection.get_representation(
            RepresentationId(str(representation_id)), as_of=as_of
        )
        if representation is None or str(representation.layer_id) != str(layer_id):
            return ()
        cells: dict[str, FrameCellSnapshot] = {}
        performers = {
            str(item.principal_id): item
            for item in self.list_performers()
            if item.principal_id is not None
        }
        accepted_versions = {
            str(identifier)
            for event in self.history(project.id, as_of=as_of)
            if event.event_type == "ReviewBatchAccepted"
            for identifier in event.payload.get("candidate_version_ids", ())
        }
        analysis = self.latest_karakal_analysis(project.id, layer_id, as_of=as_of)
        confidence = {} if analysis is None else analysis.frame_confidence
        for series in projection.list_artifact_series(
            project.id,
            layer_id=LayerId(str(layer_id)),
            representation_id=representation.id,
            as_of=as_of,
        ):
            version = projection.get_active_artifact_version(series.id, as_of=as_of)
            cursor = version
            coordinate: tuple[int, int] | None = None
            visited: set[str] = set()
            while cursor is not None and str(cursor.id) not in visited:
                visited.add(str(cursor.id))
                x, y = cursor.parameters.get("x"), cursor.parameters.get("y")
                if (
                    isinstance(x, int)
                    and not isinstance(x, bool)
                    and isinstance(y, int)
                    and not isinstance(y, bool)
                    and 1 <= x <= project.width
                    and 1 <= y <= project.height
                ):
                    coordinate = (x, y)
                    break
                cursor = (
                    None
                    if cursor.parent_version_id is None
                    else projection.get_artifact_version(cursor.parent_version_id, as_of=as_of)
                )
            if version is None or coordinate is None or series.frame_id is None:
                continue
            status = "image_ready" if representation.kind is RepresentationKind.IMAGE else "vectorized"
            performer = performers.get(str(version.author_principal_id))
            cells[str(series.frame_id)] = FrameCellSnapshot(
                coordinate[0],
                coordinate[1],
                status,
                str(series.frame_id),
                str(version.id),
                version.sha256,
                version.created_at.isoformat(),
                "" if performer is None else performer.color,
                "" if performer is None else "".join(part[:1] for part in performer.name.split()[:2]).upper(),
                "checked" if str(version.id) in accepted_versions else "not_checked",
                confidence.get(str(series.frame_id)),
            )
        review_status = {
            "issued": "in_review",
            "partially_returned": "in_review",
            "awaiting_acceptance": "returned_changed",
            "changes_requested": "changes_requested",
        }
        for batch in projection.list_active_review_batches(
            project.id, LayerId(str(layer_id)), as_of=as_of
        ):
            status = review_status.get(batch.state.value)
            if status is None:
                continue
            for item in batch.items:
                previous = cells.get(str(item.frame_id))
                if previous is not None:
                    cells[str(item.frame_id)] = FrameCellSnapshot(
                        previous.x,
                        previous.y,
                        status,
                        previous.frame_id,
                        previous.artifact_version_id,
                        previous.sha256,
                        previous.modified_at,
                        previous.performer_color,
                        previous.performer_initials,
                        "in_review",
                        previous.quality,
                    )
        return tuple(sorted(cells.values(), key=lambda item: (item.y, item.x)))
