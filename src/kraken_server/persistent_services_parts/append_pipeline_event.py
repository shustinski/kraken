"""persistent_services: append pipeline event."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from kraken_manager.domain.common import (
    PluginJobId,
)
from kraken_manager.domain.events import ProgramSnapshot
from kraken_server.services import CommandContext, ConflictError, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import _layer_id


class AppendPipelineEventMixin:
    def publish_karakal_analysis(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        layer_id = str(payload.get("layer_id", ""))
        try:
            layer = self.projections.get_layer(_layer_id(layer_id))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid layer id") from exc
        if layer is None or str(layer.project_id) != project_id:
            raise NotFoundError("Layer was not found")
        plugin_version = str(payload.get("plugin_version", "")).strip()
        if not plugin_version:
            raise ValidationError("Karakal plugin version is required")
        confidence = {
            str(frame_id): float(value)
            for frame_id, value in dict(payload.get("frame_confidence", {})).items()
        }
        if any(value < 0.0 or value > 1.0 for value in confidence.values()):
            raise ValidationError("Karakal confidence must be in range 0..1")
        stream_id = f"karakal:{layer_id}"
        revision = self.events.current_revision(stream_id)
        return self._append_auxiliary_event(
            project_id,
            stream_id=stream_id,
            event_type="KarakalAnalysisPublished",
            payload={
                "run_id": str(payload.get("run_id", "")),
                "layer_id": layer_id,
                "publication_sequence": revision + 1,
                "frame_confidence": confidence,
                "report": dict(payload.get("report", {})),
                "parameters": dict(payload.get("parameters", {})),
                "plugin_version": plugin_version,
            },
            program=ProgramSnapshot("Karakal", plugin_version),
            context=context,
        )

    def append_pipeline_event(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        layer_id = str(payload.get("layer_id", ""))
        try:
            layer = self.projections.get_layer(_layer_id(layer_id))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid layer id") from exc
        if layer is None or str(layer.project_id) != project_id:
            raise NotFoundError("Layer was not found")
        event_type = str(payload.get("event_type", ""))
        if event_type not in {"LayerPipelineActionRequested", "LayerPipelineActionRemoved", "LayerStitchingStatusChangedV1"}:
            raise ValidationError("Unsupported pipeline event type")
        if event_type == "LayerPipelineActionRemoved":
            if not isinstance(payload.get("remove_task", False), bool):
                raise ValidationError("remove_task must be a boolean")
            target = next(
                (
                    event
                    for event in self._pipeline_events(project_id)
                    if event.event_id == str(payload.get("action_event_id", ""))
                ),
                None,
            )
            if target is None or target.event_type not in {"LayerPipelineActionRequested", "PluginJobCreated"}:
                raise NotFoundError("Pipeline step was not found")
            target_layer = target.payload.get("layer_id", target.payload.get("manifest", {}).get("layer_id", ""))
            if str(target_layer) != layer_id:
                raise ValidationError("Pipeline step belongs to another layer")
            if payload.get("remove_task"):
                job_id = str(target.payload.get("plugin_job_id", ""))
                job = self.projections.get_plugin_job(PluginJobId(job_id))
                if job is None or job.state.value not in {"succeeded", "failed", "cancelled"}:
                    raise ConflictError("Cancel the task before deleting it")
                payload = {**payload, "plugin_job_id": job_id}
        return self._append_auxiliary_event(
            project_id,
            stream_id=f"layer-pipeline:{layer_id}",
            event_type=event_type,
            payload={key: value for key, value in payload.items() if key != "event_type"},
            program=ProgramSnapshot(str(payload.get("plugin_id") or "Kraken")),
            context=context,
        )
