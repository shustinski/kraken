"""composition: matrix viewport."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
    RepresentationId,
)
from kraken_manager.domain.project import (
    RepresentationKind,
)


class MatrixViewportMixin:
    def matrix_viewport(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str,
        representation_ids: Iterable[RepresentationId | str],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int = 0,
    ) -> dict[str, Any]:
        """Return sparse cells or aggregate buckets for a visible matrix region."""

        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project does not exist")
        if not (1 <= int(x1) <= int(x2) <= project.width and 1 <= int(y1) <= int(y2) <= project.height):
            raise ValueError("Viewport is outside the project grid")
        if int(lod) < 0 or int(lod) > 24:
            raise ValueError("LOD must be between 0 and 24")

        identifiers = tuple(str(value) for value in representation_ids if str(value))
        representations = {
            str(value.id): value
            for value in self.list_representations(project.id, LayerId(str(layer_id)))
            if str(value.id) in identifiers
        }
        priority = {
            "empty": 0,
            "image_ready": 1,
            "processing": 2,
            "vectorized": 3,
            "in_review": 4,
            "returned_unchanged": 5,
            "returned_changed": 6,
            "approved": 7,
            "changes_requested": 8,
            "conflict": 9,
            "error": 10,
        }
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        coverage: dict[str, set[tuple[int, int]]] = {}
        for identifier in identifiers:
            current_coverage = coverage.setdefault(identifier, set())
            representation = representations.get(identifier)
            frame_cells = self.frame_cells(project.id, layer_id, identifier)
            for item in frame_cells:
                if not (x1 <= item.x <= x2 and y1 <= item.y <= y2):
                    continue
                coordinate = (item.x, item.y)
                current_coverage.add(coordinate)
                current = merged.get(coordinate)
                image_asset = (
                    {
                        "asset_sha256": item.sha256,
                        "asset_revision": item.artifact_version_id,
                    }
                    if representation is not None
                    and representation.kind is RepresentationKind.IMAGE
                    and item.sha256
                    else {}
                )
                if current is not None and priority.get(str(current["status"]), 0) > priority.get(item.status, 0):
                    current.update(image_asset)
                    continue
                replacement = {
                    "artifact_version_id": item.artifact_version_id,
                    "frame_id": item.frame_id,
                    "sha256": item.sha256,
                    "status": item.status,
                    "x": item.x,
                    "y": item.y,
                    "modified_at": item.modified_at,
                    "performer_color": item.performer_color,
                    "performer_initials": item.performer_initials,
                    "review_status": item.review_status,
                    "quality": item.quality,
                }
                if current is not None:
                    for field in (
                        "asset_sha256",
                        "asset_source_key",
                        "asset_revision",
                        "asset_path",
                        "asset_media_type",
                    ):
                        if field in current:
                            replacement[field] = current[field]
                replacement.update(image_asset)
                merged[coordinate] = replacement
            if (
                not frame_cells
                and representation is not None
                and representation.kind is RepresentationKind.IMAGE
                and representation.source
                and representation.source != "managed-import"
            ):
                for cell in self._external_directory_viewport(
                    project,
                    representation,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                ):
                    coordinate = (int(cell["x"]), int(cell["y"]))
                    current_coverage.add(coordinate)
                    current = merged.get(coordinate)
                    if current is None:
                        merged[coordinate] = cell
                    else:
                        current.update(
                            {
                                field: cell[field]
                                for field in (
                                    "asset_source_key",
                                    "asset_revision",
                                    "asset_path",
                                    "asset_media_type",
                                )
                            }
                        )

        managed = {
            identifier
            for identifier, representation in representations.items()
            if representation.source == "managed-import"
        }
        if int(lod) == 0 and managed:
            for y in range(int(y1), int(y2) + 1):
                for x in range(int(x1), int(x2) + 1):
                    coordinate = (x, y)
                    missing_identifiers = tuple(
                        identifier
                        for identifier in managed
                        if coordinate not in coverage.get(identifier, set())
                    )
                    if not missing_identifiers:
                        continue
                    current = merged.get(coordinate)
                    if current is None:
                        current = {
                            "artifact_version_id": "",
                            "frame_id": str(project.frame_id_at(x, y)),
                            "sha256": "",
                            "x": x,
                            "y": y,
                        }
                        merged[coordinate] = current
                    # A sparse managed vector representation must not erase the
                    # raster asset already merged into this coordinate.  The
                    # missing state is orthogonal to the image used for the
                    # thumbnail.
                    current.update(
                        {
                            "status": "error",
                            "missing": True,
                            "missing_representation_ids": missing_identifiers,
                        }
                    )

        revision = str(self._projection(project.id).checkpoint)
        if int(lod) == 0:
            return {
                "project_id": str(project.id),
                "layer_id": str(layer_id),
                "bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "lod": 0,
                "revision": revision,
                "cells": tuple(sorted(merged.values(), key=lambda value: (value["y"], value["x"]))),
                "aggregates": (),
            }

        span = 1 << int(lod)
        buckets: dict[tuple[int, int], dict[str, Any]] = {}
        for cell in merged.values():
            bucket_x = (int(cell["x"]) - 1) // span
            bucket_y = (int(cell["y"]) - 1) // span
            bucket = buckets.setdefault(
                (bucket_x, bucket_y),
                {
                    "bounds": {
                        "x1": bucket_x * span + 1,
                        "y1": bucket_y * span + 1,
                        "x2": min(project.width, (bucket_x + 1) * span),
                        "y2": min(project.height, (bucket_y + 1) * span),
                    },
                    "materialized_count": 0,
                    "status_counts": {},
                },
            )
            bucket["materialized_count"] += 1
            status = str(cell["status"])
            bucket["status_counts"][status] = int(bucket["status_counts"].get(status, 0)) + 1
        return {
            "project_id": str(project.id),
            "layer_id": str(layer_id),
            "bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "lod": int(lod),
            "revision": revision,
            "cells": (),
            "aggregates": tuple(buckets[key] for key in sorted(buckets, key=lambda value: (value[1], value[0]))),
        }
