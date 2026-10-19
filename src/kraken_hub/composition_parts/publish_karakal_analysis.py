"""composition: publish karakal analysis."""
from __future__ import annotations
from typing import TYPE_CHECKING
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
    new_uuid,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.infrastructure.filesystem import (
    FilesystemEventStore,
)


class PublishKarakalAnalysisMixin:
    def publish_karakal_analysis(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        frame_confidence: dict[str, float],
        report: dict[str, object],
        parameters: dict[str, object],
        plugin_version: str,
        idempotency_key: str,
    ) -> KarakalAnalysisRun:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project does not exist")
        layer = next(
            (value for value in self.list_layers(project.id) if str(value.id) == str(layer_id)),
            None,
        )
        if layer is None:
            raise ValueError("Layer does not exist")
        normalized: dict[str, float] = {}
        for frame_id, raw_value in frame_confidence.items():
            value = float(raw_value)
            if not 0.0 <= value <= 1.0:
                raise ValueError("Karakal confidence must be in range 0..1")
            normalized[str(frame_id)] = value
        prior = next(
            (
                event
                for event in reversed(self.history(project.id))
                if event.event_type == "KarakalAnalysisPublished"
                and event.idempotency_key == idempotency_key
            ),
            None,
        )
        if prior is None:
            stream_id = f"karakal:{layer.id}"
            store = FilesystemEventStore(self.catalog_root, str(project.id))
            revision = store.current_revision(stream_id)
            run_id = new_uuid()
            event = EventEnvelope.create(
                stream_id=stream_id,
                project_id=project.id,
                revision=revision + 1,
                event_type="KarakalAnalysisPublished",
                payload={
                    "run_id": run_id,
                    "layer_id": str(layer.id),
                    "publication_sequence": revision + 1,
                    "frame_confidence": normalized,
                    "report": dict(report),
                    "parameters": dict(parameters),
                    "plugin_version": str(plugin_version),
                },
                actor=ActorSnapshot.from_principal(principal),
                program=ProgramSnapshot("Karakal", str(plugin_version)),
                idempotency_key=idempotency_key,
            )
            store.append(stream_id, expected_revision=revision, events=(event,))
            prior = event
        return self._karakal_run_from_event(prior)

if TYPE_CHECKING:
    from kraken_hub.composition_parts.frame_cell_snapshot import KarakalAnalysisRun
