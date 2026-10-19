"""remote_client: remove layer pipeline action."""
from __future__ import annotations
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import datetime
from uuid import uuid4
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
    Representation,
)

from kraken_hub.remote_client_parts.remote_history_event import _RemoteHistoryEvent


class RemoveLayerPipelineActionMixin:
    def publish_karakal_analysis(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        frame_confidence: dict[str, float],
        report: dict[str, object],
        parameters: dict[str, object],
        plugin_version: str,
        idempotency_key: str,
    ) -> object:
        del principal
        stream_id = f"karakal:{layer_id}"
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/analyses/karakal",
            payload={
                "layer_id": str(layer_id),
                "frame_confidence": frame_confidence,
                "report": report,
                "parameters": parameters,
                "plugin_version": plugin_version,
            },
            idempotency_key=idempotency_key,
            if_match=self._stream_revision(project_id, stream_id),
        )
        return self._karakal_run(project_id, payload)

    def latest_karakal_analysis(
        self, project_id: object, layer_id: object, *, as_of: datetime | None = None
    ) -> object | None:
        events = self.history(project_id, as_of=as_of)
        event = next(
            (
                item
                for item in reversed(events)
                if item.event_type == "KarakalAnalysisPublished"
                and str(item.payload.get("layer_id", "")) == str(layer_id)
            ),
            None,
        )
        return None if event is None else self._karakal_run(project_id, event._payload)

    @staticmethod
    def _karakal_run(project_id: object, event: Mapping[str, object]) -> object:
        from kraken_hub.composition import KarakalAnalysisRun

        raw = event.get("payload", {})
        payload = raw if isinstance(raw, Mapping) else {}
        return KarakalAnalysisRun(
            run_id=str(payload.get("run_id", "")),
            project_id=str(project_id),
            layer_id=str(payload.get("layer_id", "")),
            publication_sequence=int(payload.get("publication_sequence", 0)),
            created_at=str(event.get("recorded_at", "")),
            frame_confidence={
                str(key): float(value)
                for key, value in dict(payload.get("frame_confidence", {})).items()
            },
            report=dict(payload.get("report", {})),
            parameters=dict(payload.get("parameters", {})),
            plugin_version=str(payload.get("plugin_version", "")),
        )

    def record_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        action: str,
        node_id: str,
        plugin_id: str,
        capability: str,
        mode: str,
        parameters: dict[str, object] | None = None,
    ) -> object:
        del principal
        stream_id = f"layer-pipeline:{layer_id}"
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/pipeline-actions",
            payload={
                "event_type": "LayerPipelineActionRequested",
                "layer_id": str(layer_id),
                "action": action,
                "node_id": node_id,
                "plugin_id": plugin_id,
                "capability": capability,
                "mode": mode,
                "parameters": dict(parameters or {}),
                "state": "launched",
            },
            idempotency_key=f"pipeline:{layer_id}:{uuid4()}",
            if_match=self._stream_revision(project_id, stream_id),
        )
        return _RemoteHistoryEvent(payload, project_id=str(project_id))

    def set_layer_stitching_status(
        self, *, principal, project_id, layer_id, kind: str, completed: bool, version_id: str = "",
    ):
        del principal
        payload = self._call(
            "POST", f"/api/v1/projects/{project_id}/pipeline-actions",
            payload={"event_type": "LayerStitchingStatusChangedV1", "layer_id": str(layer_id),
                "kind": kind, "completed": completed, "version_id": version_id},
            idempotency_key=f"stitching:{uuid4()}",
            if_match=self._stream_revision(project_id, f"layer-pipeline:{layer_id}"),
        )
        return _RemoteHistoryEvent(payload, project_id=str(project_id))

    def remove_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: object,
        layer_id: object,
        action_event_id: str,
        remove_task: bool = False,
    ) -> object:
        del principal
        target = next(
            (
                item
                for item in self.history(project_id)
                if item.event_id == str(action_event_id)
                and item.event_type in {"LayerPipelineActionRequested", "PluginJobCreated"}
                and str(item.payload.get("layer_id", item.payload.get("manifest", {}).get("layer_id", "")))
                == str(layer_id)
            ),
            None,
        )
        if target is None:
            raise ValueError("Pipeline step does not exist in this layer")
        stream_id = f"layer-pipeline:{layer_id}"
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/pipeline-actions",
            payload={
                "event_type": "LayerPipelineActionRemoved",
                "layer_id": str(layer_id),
                "action_event_id": target.event_id,
                "remove_task": remove_task,
                "plugin_job_id": str(target.payload.get("plugin_job_id", "")),
                "action": str(target.payload.get("action", "")),
            },
            idempotency_key=f"pipeline-remove:{target.event_id}:{'task' if remove_task else 'step'}",
            if_match=self._stream_revision(project_id, stream_id),
        )
        return _RemoteHistoryEvent(payload, project_id=str(project_id))

    def matrix_viewport(
        self,
        project_id: object,
        *,
        layer_id: object,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int = 0,
        representation_ids: object = (),
        include_missing: bool = True,
    ) -> Mapping[str, object]:
        query = urllib.parse.urlencode(
            [
                ("layer_id", str(layer_id)),
                ("x1", str(x1)),
                ("y1", str(y1)),
                ("x2", str(x2)),
                ("y2", str(y2)),
                ("lod", str(lod)),
                ("include_missing", "true" if include_missing else "false"),
                *[("representation_id", str(value)) for value in representation_ids],
            ]
        )
        return self._call(
            "GET",
            f"/api/v1/projects/{project_id}/viewport?{query}",
        )

    def materialize_representation_inputs(
        self, *, principal: Principal, project: Project, layer: Layer, representation: Representation
    ) -> tuple[object, ...]:
        del principal
        self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/{layer.id}/representations/{representation.id}/prepare-inputs",
            payload={},
            idempotency_key=f"workspace-inputs:{representation.id}",
            timeout=600,
        )
        return self.frame_cells(project.id, layer.id, representation.id)
