"""Resolve pipeline ports through authoritative artifact/job projections."""

from __future__ import annotations

from pathlib import PurePath

from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.pipeline_execution import deepcopy
from kraken_manager.domain.artifacts import deterministic_frame_series_id, is_training_dataset_artifact, is_published_model_artifact
from kraken_manager.domain.pipeline import PipelineDataKind
from kraken_manager.domain.pipeline_layer import PipelineLayerOperation
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose, StructureState
from kraken_manager.domain.workflows import PluginInputV1


class PipelineBindings:
    def __init__(self, projections, events, project, layer_id):
        self.projections, self.events, self.project, self.layer_id = projections, events, project, str(layer_id)
        from .pipeline_stitch import PipelineStitchBindings
        self.stitch = PipelineStitchBindings(self)

    def validate(self, definition):
        if definition.project_id != str(self.project.id) or definition.layer_id != self.layer_id:
            raise ConflictError("Pipeline definition belongs to another project/layer")
        layer = self.projections.get_layer(self.layer_id)
        if layer is None or str(layer.project_id) != str(self.project.id) or layer.state is StructureState.ARCHIVED:
            raise NotFoundError("Active pipeline layer was not found")
        for op in definition.operations:
            if isinstance(op, PipelineLayerOperation):
                self.stitch.validate(op)
                continue
            for identifier, kind in (
                (op.source_representation_id, op.input_kind),
                (op.target_representation_id, op.output_kind),
            ):
                representation = self.projections.get_representation(identifier)
                if (
                    representation is None
                    or str(representation.project_id) != str(self.project.id)
                    or str(representation.layer_id) != self.layer_id
                    or representation.state is StructureState.ARCHIVED
                ):
                    raise NotFoundError("Pipeline representation was not found in its layer")
                expected = RepresentationKind.VECTOR if kind is PipelineDataKind.VECTOR else RepresentationKind.IMAGE
                if (
                    kind not in {PipelineDataKind.DATASET, PipelineDataKind.MODEL}
                    and representation.kind is not expected
                ):
                    raise ConflictError("Pipeline representation kind does not match its port")
                if kind is PipelineDataKind.MASK and representation.purpose is not RepresentationPurpose.BINARY:
                    raise ConflictError("Binary-mask port requires a binary representation")
                if (
                    kind is PipelineDataKind.CONFIDENCE
                    and representation.purpose is not RepresentationPurpose.CONFIDENCE
                ):
                    raise ConflictError("Confidence port requires a confidence representation")
            for x, y in op.coordinates:
                self.project.frame_id_at(x, y)

    def bind(self, op, producer_outputs, model_outputs=()):
        if isinstance(op, PipelineLayerOperation):
            return self.stitch.bind(op, producer_outputs)
        producer_outputs = set(producer_outputs)
        inputs, heads, media = [], {}, {}
        parameters = deepcopy(op.parameters)
        model_version = None
        if op.model_input is not None:
            from kraken_manager.application.model_inputs import resolve_model_input
            identifier = op.model_input.version_id
            if op.model_input.producer_id is not None:
                candidates = [self.projections.get_artifact_version(value) for value in set(model_outputs)]
                candidates = [version for version in candidates if version and is_published_model_artifact(version)]
                if len(candidates) != 1:
                    raise ConflictError("Model producer must publish exactly one managed model")
                identifier = str(candidates[0].id)
            model_version, model_series, metadata = resolve_model_input(
                self.projections, self.project.id, self.layer_id, {"model_version_id": identifier},
            )
            parameters.update(metadata)
            heads[str(model_series.id)] = str(model_version.id)
        # Layer attachments (datasets/models) have no frame series. The existing
        # training capability binds an explicit published dataset version.
        if op.input_kind is PipelineDataKind.DATASET:
            identifier = parameters.get("dataset_version_id")
            if producer_outputs:
                candidates = [self.projections.get_artifact_version(value) for value in producer_outputs]
                candidates = [v for v in candidates if v and is_training_dataset_artifact(v)]
                if len(candidates) != 1:
                    raise ConflictError("Producer must publish exactly one dataset")
                identifier = str(candidates[0].id)
            version = self.projections.get_artifact_version(identifier) if identifier else None
            series = self.projections.get_artifact_series(version.series_id) if version else None
            if (
                version is None
                or series is None
                or series.archived
                or str(series.project_id) != str(self.project.id)
                or str(series.layer_id) != self.layer_id
                or not is_training_dataset_artifact(version)
            ):
                raise ConflictError("Pipeline needs a published dataset in this layer")
            x, y = op.coordinates[0]
            parameters["dataset_version_id"] = str(version.id)
            inputs.append(
                PluginInputV1(self.project.frame_id_at(x, y), version.id, version.sha256, "inputs/dataset.zip")
            )
            heads[str(series.id)] = str(version.id)
            media[str(version.id)] = version.media_type
            coordinates = [[x, y]]
        else:
            coordinates = [list(xy) for xy in op.coordinates]
            for x, y in op.coordinates:
                frame_id = self.project.frame_id_at(x, y)
                series_id = deterministic_frame_series_id(op.source_representation_id, frame_id)
                version = self.projections.get_active_artifact_version(series_id)
                if version is None:
                    raise ConflictError(f"Pipeline input is missing at ({x}, {y})")
                if producer_outputs and str(version.id) not in producer_outputs:
                    raise ConflictError("Pipeline source head was replaced outside its producer run")
                suffix = PurePath(version.filename).suffix or ".bin"
                inputs.append(PluginInputV1(frame_id, version.id, version.sha256, f"inputs/{x}_{y}{suffix}"))
                heads[str(series_id)] = str(version.id)
                media[str(version.id)] = version.media_type
        parameters.update(input_media_types=media, project_width=self.project.width)
        if op.capability in {"frames.vectorize.v1", "dataset.model.train.v1"}:
            parameters["execution_mode"] = "headless"
        if op.quality_gate:
            from kraken_manager.application.quality import QualityService

            QualityService.require_clear_inputs(
                self.events,
                self.project.id,
                tuple(item.artifact_version_id for item in inputs) + (() if model_version is None else (model_version.id,)),
                projections=self.projections,
            )
        return dict(
            layer_id=self.layer_id,
            capability=op.capability,
            target_representation_id=op.target_representation_id,
            coordinates=coordinates,
            parameters=parameters,
            input_heads=heads,
            inputs=[
                dict(
                    frame_id=str(item.frame_id),
                    artifact_version_id=str(item.artifact_version_id),
                    sha256=item.sha256,
                    relative_path=item.relative_path,
                )
                for item in inputs
            ],
        )

    def current(self, request):
        for series_id, version_id in request["input_heads"].items():
            current = self.projections.get_active_artifact_version(series_id)
            if current is None or str(current.id) != version_id:
                return False
        return request.get("kind") != "layer" or self.stitch.current(request)

    def observe(self, job_id):
        job = self.projections.get_plugin_job(job_id)
        if job is None:
            return self.stitch.observe(job_id)
        if str(job.project_id) != str(self.project.id) or str(job.layer_id) != self.layer_id:
            raise ConflictError("Pipeline job belongs to another scope")
        outputs = []
        history = self.events.load_stream(f"plugin-job:{job_id}")
        expected_version = next(
            (
                event.payload.get("manifest", {}).get("parameters", {}).get("pipeline_algorithm_version")
                for event in history
                if event.event_type == "PluginJobCreated"
            ),
            None,
        )
        incompatible = False
        for event in history:
            if event.event_type in {"PluginResultImported", "PluginResultV2Imported"}:
                for identifier in event.payload.get("artifact_version_ids", ()):
                    version = self.projections.get_artifact_version(identifier)
                    series = self.projections.get_artifact_series(version.series_id) if version else None
                    if (
                        series
                        and str(series.project_id) == str(self.project.id)
                        and str(series.layer_id) == self.layer_id
                    ):
                        outputs.append(str(identifier))
                        incompatible |= expected_version is not None and version.tool_version != expected_version
        if incompatible:
            return dict(
                state="failed",
                error="Published plugin version does not match the pinned pipeline algorithm",
                output_version_ids=[],
            )
        return dict(state=job.state.value, error=job.error, output_version_ids=outputs)
