"""persistent_services: active artifact versions."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any


class ActiveArtifactVersionsMixin:
    def _active_artifact_versions(self, series: Iterable[Any]) -> dict[str, Any]:
        rows = tuple(series)
        if not rows:
            return {}
        batch = getattr(self.projections, "list_active_artifact_versions", None)
        if batch is None:
            loaded = {
                str(item.id): self.projections.get_active_artifact_version(item.id)
                for item in rows
            }
            return {key: value for key, value in loaded.items() if value is not None}
        return {str(key): value for key, value in batch(item.id for item in rows).items()}
