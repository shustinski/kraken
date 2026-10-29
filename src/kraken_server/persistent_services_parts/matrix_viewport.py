"""persistent_services: matrix viewport."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from kraken_manager.domain.common import (
    RepresentationId,
    validate_uuid,
)
from kraken_manager.domain.project import (
    RepresentationKind,
)
from kraken_server.services import NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_id,
    _project_id,
)


class MatrixViewportMixin:
    def frame_cells(self, project_id, layer_id, representation_id, *, as_of=None, bounds=None):
        from kraken_manager.application.frame_queries import FrameQueryService
        from kraken_manager.infrastructure.filesystem._codec import encode_model

        _project_id(project_id)
        _layer_id(layer_id)
        validate_uuid(str(representation_id), field="representation_id")

        def history(identifier, *, as_of=None, event_types=None):
            return tuple(
                event for event in self._pipeline_events(str(identifier))
                if (as_of is None or event.recorded_at <= as_of)
                and (event_types is None or event.event_type in event_types)
            )

        store = getattr(self, "performer_store", None)
        queries = FrameQueryService(
            self.projections, history=history,
            performers=lambda: () if store is None else store.list(),
        )
        return [encode_model(item) for item in queries.frame_cells(
            project_id, layer_id, representation_id, as_of=as_of, bounds=bounds,
        )]

    def matrix_viewport(
        self,
        project_id: str,
        *,
        layer_id: str,
        representation_ids: Iterable[str] = (),
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int,
        include_missing: bool = True,
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        if not (1 <= x1 <= x2 <= project["width"] and 1 <= y1 <= y2 <= project["height"]):
            raise ValidationError("Viewport is outside the project grid")
        if lod < 0 or lod > 24:
            raise ValidationError("LOD must be between 0 and 24")
        project_model = self.projections.get_project(_project_id(project_id))
        assert project_model is not None
        layer_identifier = _layer_id(layer_id)
        layer = self.projections.get_layer(layer_identifier)
        if layer is None or str(layer.project_id) != project_id:
            raise NotFoundError(layer_id)
        identifiers = tuple(
            RepresentationId(validate_uuid(str(value), field="representation_id"))
            for value in representation_ids
        )
        if not identifiers:
            identifiers = tuple(
                representation.id
                for representation in self.projections.list_representations(layer.id)
            )
        representations = {
            representation.id: representation
            for identifier in identifiers
            if (representation := self.projections.get_representation(identifier))
            is not None
            and representation.layer_id == layer_identifier
        }
        priority = {
            "image_ready": 1,
            "vectorized": 3,
            "in_review": 4,
            "returned_changed": 6,
            "approved": 7,
            "changes_requested": 8,
            "error": 10,
        }
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        coverage: dict[str, set[tuple[int, int]]] = {}
        newest = ""
        series_groups: list[tuple[Any, tuple[Any, ...]]] = []
        viewport_series: list[Any] = []
        for representation in representations.values():
            rows = tuple(
                series
                for series in self.projections.list_artifact_series(
                    project_model.id,
                    layer_id=layer_identifier,
                    representation_id=representation.id,
                )
                if str(series.project_id) == project_id
            )
            series_groups.append((representation, rows))
            viewport_series.extend(rows)
        active_versions = self._active_artifact_versions(viewport_series)
        for representation, series_rows in series_groups:
            current_coverage = coverage.setdefault(str(representation.id), set())
            for series in series_rows:
                version = active_versions.get(str(series.id))
                cursor = version
                coordinate: tuple[int, int] | None = None
                visited: set[str] = set()
                while cursor is not None and str(cursor.id) not in visited:
                    visited.add(str(cursor.id))
                    raw_x = cursor.parameters.get("x")
                    raw_y = cursor.parameters.get("y")
                    if (
                        isinstance(raw_x, int)
                        and not isinstance(raw_x, bool)
                        and isinstance(raw_y, int)
                        and not isinstance(raw_y, bool)
                    ):
                        coordinate = (raw_x, raw_y)
                        break
                    cursor = (
                        None
                        if cursor.parent_version_id is None
                        else self.projections.get_artifact_version(
                            cursor.parent_version_id
                        )
                    )
                if version is None or coordinate is None or series.frame_id is None:
                    continue
                x, y = coordinate
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    continue
                current_coverage.add(coordinate)
                newest = max(newest, version.created_at.isoformat())
                status = (
                    "image_ready"
                    if representation.kind is RepresentationKind.IMAGE
                    else "vectorized"
                )
                cell = {
                    "artifact_version_id": str(version.id),
                    "frame_id": str(series.frame_id),
                    "sha256": version.sha256,
                    "status": status,
                    "x": x,
                    "y": y,
                    "modified_at": version.created_at.isoformat(),
                    "review_status": "not_checked",
                }
                if representation.kind is RepresentationKind.IMAGE and version.blob is not None:
                    cell.update(
                        {
                            "asset_sha256": version.blob.sha256,
                            "asset_source_key": str(version.id),
                            "asset_revision": str(version.id),
                            "asset_media_type": version.media_type,
                        }
                    )
                previous = merged.get(coordinate)
                if previous is None or priority[status] >= priority.get(
                    str(previous.get("status")), 0
                ):
                    if previous is not None:
                        for field in (
                            "asset_sha256",
                            "asset_source_key",
                            "asset_revision",
                            "asset_media_type",
                        ):
                            if field in previous and field not in cell:
                                cell[field] = previous[field]
                    merged[coordinate] = cell
                elif "asset_source_key" in cell:
                    previous.update(
                        {
                            key: value
                            for key, value in cell.items()
                            if key.startswith("asset_")
                        }
                    )
        review_status = {
            "issued": "in_review",
            "partially_returned": "in_review",
            "awaiting_acceptance": "returned_changed",
            "changes_requested": "changes_requested",
        }
        cells_by_frame = {
            str(cell["frame_id"]): cell for cell in merged.values()
        }
        for batch in self.projections.list_active_review_batches(
            project_model.id, layer_identifier
        ):
            status = review_status.get(batch.state.value)
            if status is None:
                continue
            for item in batch.items:
                cell = cells_by_frame.get(str(item.frame_id))
                if cell is not None:
                    cell["status"] = status
                    cell["review_status"] = "in_review"
        managed = {
            str(identifier)
            for identifier, representation in representations.items()
            if representation.source == "managed-import"
        }
        if lod == 0 and include_missing and managed:
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    missing = tuple(
                        identifier
                        for identifier in managed
                        if (x, y) not in coverage.get(identifier, set())
                    )
                    if missing:
                        cell = merged.setdefault(
                            (x, y),
                            {
                                "artifact_version_id": "",
                                "frame_id": str(project_model.frame_id_at(x, y)),
                                "sha256": "",
                                "x": x,
                                "y": y,
                            },
                        )
                        cell.update(
                            {
                                "status": "error",
                                "missing": True,
                                "missing_representation_ids": missing,
                            }
                        )
        cells = tuple(
            sorted(merged.values(), key=lambda value: (value["y"], value["x"]))
        )
        if lod > 0:
            span = 1 << lod
            buckets: dict[tuple[int, int], dict[str, Any]] = {}
            for cell in cells:
                bucket_x = (int(cell["x"]) - 1) // span
                bucket_y = (int(cell["y"]) - 1) // span
                bucket = buckets.setdefault(
                    (bucket_x, bucket_y),
                    {
                        "bounds": {
                            "x1": bucket_x * span + 1,
                            "y1": bucket_y * span + 1,
                            "x2": min(project_model.width, (bucket_x + 1) * span),
                            "y2": min(project_model.height, (bucket_y + 1) * span),
                        },
                        "materialized_count": 0,
                        "status_counts": {},
                    },
                )
                bucket["materialized_count"] += 1
                counts = bucket["status_counts"]
                status = str(cell.get("status", "empty"))
                counts[status] = counts.get(status, 0) + 1
        return {
            "project_id": project_id,
            "layer_id": layer_id,
            "bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "lod": lod,
            "revision": newest or str(project["revision"]),
            "cells": cells if lod == 0 else (),
            "aggregates": (
                tuple(buckets[key] for key in sorted(buckets)) if lod > 0 else ()
            ),
        }
