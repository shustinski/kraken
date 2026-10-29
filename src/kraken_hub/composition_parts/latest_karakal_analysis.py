"""composition: latest karakal analysis."""
from __future__ import annotations
from datetime import datetime
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
)
from kraken_manager.domain.events import EventEnvelope

from kraken_hub.composition_parts.frame_cell_snapshot import KarakalAnalysisRun


class LatestKarakalAnalysisMixin:
    def latest_karakal_analysis(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        *,
        as_of: datetime | None = None,
    ) -> KarakalAnalysisRun | None:
        event = next(
            (
                item
                for item in reversed(self.history(
                    project_id, as_of=as_of, event_types=("KarakalAnalysisPublished",)
                ))
                if item.event_type == "KarakalAnalysisPublished"
                and str(item.payload.get("layer_id", "")) == str(layer_id)
            ),
            None,
        )
        return None if event is None else self._karakal_run_from_event(event)

    @staticmethod
    def _karakal_run_from_event(event: EventEnvelope) -> KarakalAnalysisRun:
        payload = event.payload
        return KarakalAnalysisRun(
            run_id=str(payload["run_id"]),
            project_id=str(event.project_id),
            layer_id=str(payload["layer_id"]),
            publication_sequence=int(payload["publication_sequence"]),
            created_at=event.recorded_at.isoformat(),
            frame_confidence={
                str(identifier): float(value)
                for identifier, value in dict(payload.get("frame_confidence", {})).items()
            },
            report=dict(payload.get("report", {})),
            parameters=dict(payload.get("parameters", {})),
            plugin_version=str(payload.get("plugin_version", "")),
        )
