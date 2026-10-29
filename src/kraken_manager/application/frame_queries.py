"""Frame read-model policies shared by all storage implementations."""

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

from kraken_manager.application.results import FrameCellSnapshot, KarakalAnalysisRun
from collections.abc import Callable, Iterable
from typing import Protocol
from kraken_manager.application.ports import ProjectionStore
from kraken_manager.domain.identity import Performer
from kraken_manager.domain.events import EventEnvelope


class FrameIndex(Protocol):
    def rows(self, project_id, layer_id, representation_id, bounds) -> tuple: ...


class FrameQueryService:
    def __init__(
        self,
        projection: ProjectionStore,
        *,
        history: Callable,
        performers: Callable[[], Iterable[Performer]],
        frame_index: FrameIndex | None = None,
    ):
        self._projections = projection
        self._history = history
        self._performers = performers
        self._frame_index = frame_index

    def frame_cells(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        representation_id: RepresentationId | str,
        *,
        as_of: datetime | None = None,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[FrameCellSnapshot, ...]:
        project = self._projections.get_project(ProjectId(str(project_id)), as_of=as_of)
        if project is None:
            return ()
        if bounds is not None and not (
            1 <= bounds[0] <= bounds[2] <= project.width and 1 <= bounds[1] <= bounds[3] <= project.height
        ):
            raise ValueError("Viewport is outside the project grid")
        projection = self._projections
        representation = projection.get_representation(RepresentationId(str(representation_id)), as_of=as_of)
        if representation is None or str(representation.layer_id) != str(layer_id):
            return ()
        cells: dict[str, FrameCellSnapshot] = {}
        performers = {str(item.principal_id): item for item in self._performers() if item.principal_id is not None}
        accepted_versions = {
            str(identifier)
            for event in self._history(project.id, as_of=as_of, event_types=("ReviewBatchAccepted",))
            if event.event_type == "ReviewBatchAccepted"
            for identifier in event.payload.get("candidate_version_ids", ())
        }
        analysis = self.latest_analysis(project.id, layer_id, as_of=as_of)
        confidence = {} if analysis is None else analysis.frame_confidence
        coordinates = {}
        if self._frame_index is not None and bounds is not None and as_of is None:
            rows = self._frame_index.rows(project.id, layer_id, representation.id, bounds)
            series_list = tuple(row[0] for row in rows)
            active_versions = {str(row[0].id): row[1] for row in rows}
            coordinates = {str(row[0].id): row[2] for row in rows}
        else:
            series_list = projection.list_artifact_series(
                project.id,
                layer_id=LayerId(str(layer_id)),
                representation_id=representation.id,
                as_of=as_of,
            )
            active_versions = (
                projection.list_active_artifact_versions(series.id for series in series_list) if as_of is None else None
            )
        for series in series_list:
            version = (
                active_versions.get(str(series.id))
                if active_versions is not None
                else projection.get_active_artifact_version(series.id, as_of=as_of)
            )
            cursor = version
            coordinate: tuple[int, int] | None = coordinates.get(str(series.id))
            visited: set[str] = set()
            while coordinate is None and cursor is not None and str(cursor.id) not in visited:
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
        for batch in projection.list_active_review_batches(project.id, LayerId(str(layer_id)), as_of=as_of):
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
        return tuple(
            sorted(
                (
                    item
                    for item in cells.values()
                    if bounds is None or bounds[0] <= item.x <= bounds[2] and bounds[1] <= item.y <= bounds[3]
                ),
                key=lambda item: (item.y, item.x),
            )
        )

    def latest_analysis(self, project_id, layer_id, *, as_of=None):
        return latest_karakal_analysis(self._history, project_id, layer_id, as_of=as_of)


def latest_karakal_analysis(history, project_id, layer_id, *, as_of=None) -> KarakalAnalysisRun | None:
    event = next(
        (
            item
            for item in reversed(history(project_id, as_of=as_of, event_types=("KarakalAnalysisPublished",)))
            if item.event_type == "KarakalAnalysisPublished" and str(item.payload.get("layer_id", "")) == str(layer_id)
        ),
        None,
    )
    return None if event is None else karakal_analysis_from_event(event)


def karakal_analysis_from_event(event: EventEnvelope) -> KarakalAnalysisRun:
    payload = event.payload
    return KarakalAnalysisRun(
        run_id=str(payload["run_id"]),
        project_id=str(event.project_id),
        layer_id=str(payload["layer_id"]),
        publication_sequence=int(payload["publication_sequence"]),
        created_at=event.recorded_at.isoformat(),
        frame_confidence={
            str(identifier): float(value) for identifier, value in dict(payload.get("frame_confidence", {})).items()
        },
        report=dict(payload.get("report", {})),
        parameters=dict(payload.get("parameters", {})),
        plugin_version=str(payload.get("plugin_version", "")),
    )
