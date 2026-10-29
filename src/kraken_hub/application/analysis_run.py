"""One published Karakal analysis, shared by local and server catalogs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KarakalAnalysisRun:
    run_id: str
    project_id: str
    layer_id: str
    publication_sequence: int
    created_at: str
    frame_confidence: dict[str, float]
    report: dict[str, object]
    parameters: dict[str, object]
    plugin_version: str
