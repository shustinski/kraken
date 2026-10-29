"""composition: external directory viewport."""
from __future__ import annotations
import hashlib
import mimetypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from kraken_core.safe_files import ensure_regular_directory
from kraken_manager.workspace import (
    WorkspaceValidationError,
    map_frame_positions,
)
from kraken_manager.domain.project import (
    GridOrientation,
    Project,
    Representation,
)


class ExternalDirectoryViewportMixin:
    def _external_directory_viewport(
        self,
        project: Project,
        representation: Representation,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[dict[str, Any], ...]:
        """Read legacy explicit-directory representations without importing them."""

        try:
            root = ensure_regular_directory(representation.source)
            directory_revision = root.stat().st_mtime_ns
        except (OSError, ValueError):
            return ()
        cache_key = (str(representation.id), str(root), directory_revision)
        paths = self._external_source_indexes.get(cache_key)
        if paths is None:
            discovered = tuple(
                sorted(
                    (
                        path
                        for path in root.iterdir()
                        if path.is_file()
                        and path.suffix.casefold() in {".jpg", ".jpeg", ".bmp", ".png"}
                    ),
                    key=self._natural_path_key,
                )
            )
            binding = self.layer_file_binding(project.id, representation.layer_id)
            if binding is not None and binding.frame_positions:
                slots: list[Path | None] = [None] * project.frame_count
                matched = 0
                for path in discovered:
                    position = binding.frame_positions.get(path.name)
                    if position is not None and 1 <= position <= project.frame_count:
                        slots[position - 1] = path
                        matched += 1
                if matched:
                    paths = tuple(slots)
                else:
                    try:
                        derived_positions = map_frame_positions(
                            discovered,
                            maximum=project.frame_count,
                        )
                    except WorkspaceValidationError:
                        paths = discovered
                    else:
                        for path in discovered:
                            slots[derived_positions[path.name] - 1] = path
                        paths = tuple(slots)
            else:
                paths = discovered
            self._external_source_indexes = {
                key: value
                for key, value in self._external_source_indexes.items()
                if key[0] != str(representation.id)
            }
            self._external_source_indexes[cache_key] = paths

        cells: list[dict[str, Any]] = []
        analysis = self.latest_karakal_analysis(project.id, representation.layer_id)
        confidence = {} if analysis is None else analysis.frame_confidence
        for index, path in enumerate(paths[: project.frame_count]):
            x = index % project.width + 1
            visual_row = index // project.width
            y = (
                visual_row + 1
                if project.orientation is GridOrientation.Y_DOWN
                else project.height - visual_row
            )
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            if path is None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            normalized_path = str(path.resolve())
            frame_id = str(project.frame_id_at(x, y))
            cells.append(
                {
                    "artifact_version_id": "",
                    "frame_id": frame_id,
                    "sha256": "",
                    "status": "image_ready",
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                    "performer_color": "",
                    "performer_initials": "",
                    "review_status": "not_checked",
                    "quality": confidence.get(frame_id),
                    "x": x,
                    "y": y,
                    "asset_source_key": hashlib.sha256(
                        normalized_path.encode("utf-8")
                    ).hexdigest(),
                    "asset_revision": f"{stat.st_mtime_ns}:{stat.st_size}",
                    "asset_path": normalized_path,
                    "asset_media_type": mimetypes.guess_type(path.name)[0] or "image/*",
                }
            )
        return tuple(cells)
