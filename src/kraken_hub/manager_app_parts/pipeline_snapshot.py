"""manager_app: pipeline snapshot."""
from __future__ import annotations
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose
from kraken_manager.presentation.qt import (
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
)
from kraken_manager.workspace import DerivedRunKind
from kraken_manager.domain.stitching import StitchingKind, stitching_series_id

_OPERATION_TITLES = {
    "prepare_dataset": "Создание выборки",
    "frames.dataset.prepare.v1": "Создание выборки",
    "vectorize": "Векторизация изображений",
    "frames.vectorize.v1": "Векторизация изображений",
    "train": "Обучение модели",
    "dataset.model.train.v1": "Обучение модели",
    "recognize": "Бинаризация моделью",
    "recognize_external": "Бинаризация моделью",
    "frames.binary-segment.v1": "Бинаризация моделью",
    "frames.binary-segment.v2": "Бинаризация моделью",
    "karakal": "Карта уверенности",
    "layer.confidence.analyze.v1": "Карта уверенности",
    "frames.confidence.v1": "Получение уверенности",
}

_STATE_TITLES = {
    "queued": "В очереди", "running": "Выполняется", "launched": "Запущено",
    "succeeded": "Завершено", "failed": "Ошибка", "cancelled": "Отменено",
    "partial": "Частичный результат", "draft": "Подготовка",
    "staging": "Подготовка файлов", "waiting_for_user": "Ожидает действий пользователя",
    "importing": "Сохранение результата", "recovery_required": "Требуется восстановление",
    "awaiting_authorization": "Ожидает разрешения",
}


class PipelineSnapshotMixin:
    def _pipeline_snapshot(self, project_id, layer_id) -> LayerPipelineSnapshot:
        representations = self.service.list_representations(project_id, layer_id)
        project_history = self.service.history(project_id)
        histories = [
            event
            for event in project_history
            if event.event_type == "PluginJobCreated"
            and str(event.payload.get("manifest", {}).get("layer_id", "")) == str(layer_id)
        ]
        latest_job_events = {}
        applied_parameters = {}
        for event in project_history:
            job = event.payload.get("job", {})
            if not isinstance(job, dict) and not hasattr(job, "get"):
                continue
            job_id = str(event.payload.get("plugin_job_id", job.get("id", "")))
            if job_id and str(job.get("layer_id", "")) == str(layer_id):
                latest_job_events[job_id] = event
                result = event.payload.get("result", {})
                if result.get("parameters_applied"):
                    applied_parameters[job_id] = dict(result["parameters_applied"])
            if event.event_type == "ArtifactVersionCreated":
                version_job = str(event.payload.get("plugin_job_id", ""))
                if version_job and event.payload.get("parameters"):
                    applied_parameters[version_job] = dict(event.payload["parameters"])
        removed_action_ids = {
            str(event.payload.get("action_event_id", ""))
            for event in project_history
            if event.event_type == "LayerPipelineActionRemoved"
            and str(event.payload.get("layer_id", "")) == str(layer_id)
        }
        action_events = [
            event
            for event in project_history
            if event.event_type == "LayerPipelineActionRequested"
            and str(event.payload.get("layer_id", "")) == str(layer_id)
            and event.event_id not in removed_action_ids
        ]
        histories = [event for event in histories if event.event_id not in removed_action_ids]
        recorded_job_ids = {
            str(event.payload.get("plugin_job_id", ""))
            for event in project_history
            if event.event_type == "PluginJobCreated"
        }
        action_events = [
            event
            for event in action_events
            if str(event.payload.get("parameters", {}).get("plugin_job_id", "")) not in recorded_job_ids
        ]
        sources = [
            value for value in representations
            if value.kind is RepresentationKind.IMAGE and value.purpose is RepresentationPurpose.SOURCE
        ]
        binaries = [
            value for value in representations
            if value.kind is RepresentationKind.IMAGE and value.purpose is RepresentationPurpose.BINARY
        ]
        confidence_images = [
            value for value in representations
            if value.kind is RepresentationKind.IMAGE and value.purpose is RepresentationPurpose.CONFIDENCE
        ]
        vectors = [value for value in representations if value.kind is RepresentationKind.VECTOR]
        karakal_run = self.service.latest_karakal_analysis(project_id, layer_id)
        workspace_runs = self.service.list_derived_runs(project_id, layer_id)
        stitching_nodes = []
        series_reader = getattr(self.service, "list_artifact_series", None)
        series_items = () if not callable(series_reader) else series_reader(project_id, layer_id=layer_id)
        if series_items:
            by_id = {str(item.id): item for item in series_items if not item.archived}
            statuses = {
                str(event.payload.get("kind")): bool(event.payload.get("completed"))
                for event in project_history if event.event_type == "LayerStitchingStatusChangedV1"
                and str(event.payload.get("layer_id")) == str(layer_id)
            }
            for kind in StitchingKind:
                series_id = stitching_series_id(layer_id, kind)
                if str(series_id) not in by_id:
                    continue
                primary = self.service.active_artifact_version(project_id, series_id)
                if primary is None:
                    continue
                stitching_nodes.append(PipelineNode(
                    f"stitching:{kind.value}", kind.title, "stitching",
                    "Слой сшит" if statuses.get(kind.value, False) else "Файл загружен",
                    details={"файл": primary.filename, "сшит": statuses.get(kind.value, False)},
                ))
        if not sources and binaries:
            sources = binaries[:1]
        lanes: list[PipelineLane] = []
        for source in sources:
            lane_nodes: list[PipelineNode] = [
                PipelineNode(
                    str(source.id),
                    source.name,
                    "source",
                    "Исходные изображения",
                    str(source.id),
                    source.active,
                    details={
                        "источник": (
                            "Файлы сохранены в проекте"
                            if source.source == "managed-import" or not source.source
                            else source.source
                        ),
                        "создан": source.created_at.isoformat(),
                    },
                )
            ]
            lane_edges: list[tuple[str, str]] = []
            if source is sources[0]:
                lane_nodes.extend(stitching_nodes)
                lane_edges.extend((str(source.id), node.node_id) for node in stitching_nodes)
            lane_binaries = [
                value
                for value in binaries
                if str(value.source_image_representation_id or "") == str(source.id)
                or (
                    value.source_image_representation_id is None
                    and source is sources[0]
                )
            ]
            lane_confidence = [
                value for value in confidence_images
                if str(value.source_image_representation_id or "") == str(source.id)
            ]
            lane_vectors = [
                value for value in vectors
                if str(value.source_image_representation_id or "") in {str(source.id), *(str(item.id) for item in lane_binaries)}
            ]
            target_ids = {str(item.id) for item in (*lane_binaries, *lane_vectors, *lane_confidence)}
            for action_event in action_events:
                action_payload = action_event.payload
                action_parameters = dict(action_payload.get("parameters", {}))
                action_source = str(
                    action_parameters.get("source_representation_id")
                    or action_payload.get("node_id", "")
                )
                if action_source != str(source.id):
                    # Backward-compatible fallback for old layer-level actions
                    # that were recorded before source association existed.
                    if action_source or source is not sources[0]:
                        continue
                action_id = f"action:{action_event.event_id}"
                lane_nodes.append(
                    PipelineNode(
                        action_id,
                        _OPERATION_TITLES.get(str(action_payload.get("action", "")), "Обработка слоя"),
                        "job",
                        _STATE_TITLES.get(str(action_payload.get("state", "launched")), "Запущено"),
                        state=str(action_payload.get("state", "")),
                        details={
                            "время": action_event.recorded_at.isoformat(),
                            "автор": action_event.actor.display_name,
                            "программа": action_payload.get("plugin_id", ""),
                            "capability": action_payload.get("capability", ""),
                            "режим": action_payload.get("mode", ""),
                            "параметры": action_parameters,
                            "pipeline_event_id": action_event.event_id,
                            "deletable": True,
                        },
                    )
                )
                lane_edges.append((str(source.id), action_id))
            if source is sources[0]:
                for run in workspace_runs:
                    job_id = f"workspace-job:{run.run_id}"
                    lane_nodes.append(
                        PipelineNode(
                            job_id,
                            _OPERATION_TITLES.get(run.operation, "Обработка слоя"),
                            "job",
                            _STATE_TITLES.get(run.state.value, run.state.value),
                            state=run.state.value,
                            details={
                                "run_id": run.run_id,
                                "path": run.path,
                                "operation": run.operation,
                                "plugin": run.plugin_id,
                                "создан": run.created_at,
                                "параметры": run.provenance.get("parameters", {}),
                            },
                        )
                    )
                    lane_edges.append((str(source.id), job_id))
                    if run.state.value != "succeeded":
                        continue
                    output_id = f"workspace-output:{run.run_id}"
                    output_kind = {
                        DerivedRunKind.DATASET: "dataset",
                        DerivedRunKind.RESULT: "model",
                        DerivedRunKind.VECTOR: "vector",
                    }[run.kind]
                    output_title = {
                        DerivedRunKind.DATASET: "Выборка Contour",
                        DerivedRunKind.RESULT: "Результат NeuralImage",
                        DerivedRunKind.VECTOR: "Векторы Contour",
                    }[run.kind]
                    lane_nodes.append(
                        PipelineNode(
                            output_id,
                            output_title,
                            output_kind,
                            run.state.value,
                            state=run.state.value,
                            details={
                                "run_id": run.run_id,
                                "path": run.path,
                                "operation": run.operation,
                                "plugin": run.plugin_id,
                            },
                        )
                    )
                    lane_edges.append((job_id, output_id))
            for event in histories:
                manifest = dict(event.payload.get("manifest", {}))
                target_id = str(manifest.get("target_representation_id", ""))
                parameters = dict(manifest.get("parameters", {}))
                declared_source = str(parameters.get("source_representation_id", ""))
                if declared_source and declared_source != str(source.id):
                    continue
                if not declared_source and target_id not in target_ids and source is not sources[0]:
                    continue
                job_id = str(event.payload.get("plugin_job_id", event.event_id))
                current_event = latest_job_events.get(job_id, event)
                job = dict(current_event.payload.get("job", event.payload.get("job", {})))
                capability = str(manifest.get("capability", "Задание"))
                parameters = applied_parameters.get(job_id, parameters)
                program = getattr(current_event, "program", None)
                output_id = target_id
                output_kind = ""
                output_title = ""
                if capability == "frames.dataset.prepare.v1":
                    output_id = f"{job_id}:dataset"
                    output_kind = "dataset"
                    output_title = "Выборка Contour"
                elif capability == "dataset.model.train.v1":
                    output_id = f"{job_id}:model"
                    output_kind = "model"
                    output_title = "Модель NeuralImage"
                lane_nodes.append(
                    PipelineNode(
                        job_id,
                        _OPERATION_TITLES.get(capability, "Обработка слоя"),
                        "job",
                        _STATE_TITLES.get(str(job.get("state", "queued")), str(job.get("state", "queued"))),
                        state=str(job.get("state", "")),
                        details={
                            "время": current_event.recorded_at.isoformat(),
                            "автор": current_event.actor.display_name,
                            "программа": ("" if program is None else f"{program.name} {program.version or ''}".strip()),
                            "состояние": str(job.get("state", "queued")),
                            "прогресс": job.get("progress", 0),
                            "ошибка": job.get("error") or "",
                            "параметры": parameters,
                            "pipeline_event_id": event.event_id,
                            "job_id": job_id,
                            "deletable": True,
                        },
                    )
                )
                lane_edges.append((str(source.id), job_id))
                if output_kind and str(job.get("state", "")) == "succeeded":
                    output_versions = [
                        str(item.payload.get("version_id", item.payload.get("artifact_version_id", "")))
                        for item in project_history
                        if item.event_type == "ArtifactVersionCreated"
                        and str(item.payload.get("plugin_job_id", "")) == job_id
                    ]
                    lane_nodes.append(
                        PipelineNode(
                            output_id,
                            output_title,
                            output_kind,
                            str(job.get("state", "")),
                            state=str(job.get("state", "")),
                            details={
                                "job_id": job_id,
                                "capability": capability,
                                "artifact_version_ids": tuple(output_versions),
                                "pipeline_event_id": event.event_id,
                                "deletable": True,
                            },
                        )
                    )
                if output_id and (not output_kind or str(job.get("state", "")) == "succeeded"):
                    lane_edges.append((job_id, output_id))
            for value in lane_binaries:
                lane_nodes.append(
                    PipelineNode(
                        str(value.id), value.name, "binary", "Бинарные изображения",
                        str(value.id), value.active, details={"создан": value.created_at.isoformat()},
                    )
                )
                if not any(target == str(value.id) for _, target in lane_edges):
                    lane_edges.append((str(source.id), str(value.id)))
            for value in lane_vectors:
                lane_nodes.append(
                    PipelineNode(
                        str(value.id), value.name, "vector", "CIF",
                        str(value.id), value.active, details={"создан": value.created_at.isoformat()},
                    )
                )
                if not any(target == str(value.id) for _, target in lane_edges):
                    lane_edges.append((str(value.source_image_representation_id or source.id), str(value.id)))
            for value in lane_confidence:
                lane_nodes.append(PipelineNode(
                    str(value.id), "Уверенность модели", "confidence", value.name,
                    str(value.id), value.active, details={"создан": value.created_at.isoformat()},
                ))
                if not any(target == str(value.id) for _, target in lane_edges):
                    lane_edges.append((str(source.id), str(value.id)))
            if karakal_run is not None and source is sources[0]:
                karakal_id = f"karakal:{karakal_run.run_id}"
                lane_nodes.append(
                    PipelineNode(
                        karakal_id,
                        "Карта уверенности",
                        "karakal",
                        f"{len(karakal_run.frame_confidence)} кадров",
                        details={
                            "время": karakal_run.created_at,
                            "версия": karakal_run.plugin_version,
                            "параметры": karakal_run.parameters,
                            "отчёт": karakal_run.report,
                        },
                    )
                )
                for value in lane_binaries:
                    lane_edges.append((str(value.id), karakal_id))
            if not lane_vectors:
                missing_id = f"{source.id}:missing-cif"
                lane_nodes.append(PipelineNode(missing_id, "CIF не получен", "missing", "Результат отсутствует"))
                lane_edges.append((str(source.id), missing_id))
            lanes.append(PipelineLane(str(source.id), source.name, tuple(lane_nodes), tuple(lane_edges)))
        workflow_event = next((event for event in reversed(project_history)
                               if event.event_type == "LayerPipelineWorkflowV1"
                               and str(event.payload.get("workflow", {}).get("layer_id", "")) == str(layer_id)), None)
        if workflow_event is not None:
            workflow = workflow_event.payload["workflow"]
            nodes, edges = [], []
            for operation in workflow["definition"]["operations"]:
                execution = workflow.get("nodes", {}).get(operation["id"], {})
                attempts = execution.get("attempts", ())
                latest = attempts[-1] if attempts else {}
                status = execution.get("status", "draft")
                nodes.append(PipelineNode(
                    operation["id"], operation["title"], "operation", _STATE_TITLES.get(status, status),
                    state=status, details={"workflow_operation": True, "capability": operation["capability"],
                        "algorithm_version": operation["algorithm_version"], "attempts": len(attempts),
                        "job_id": latest.get("job_id"), "error": latest.get("error") or execution.get("error"),
                        "artifact_version_ids": latest.get("output_version_ids", ()),
                        "workflow_revision": workflow_event.revision,
                        "approval_required": operation["approval_required"] and not execution.get("approved", False)},
                ))
                edges.extend((dependency, operation["id"]) for dependency in operation["dependencies"])
            lanes.append(PipelineLane(f"workflow:{layer_id}", "Технологический процесс", tuple(nodes), tuple(edges)))
        return LayerPipelineSnapshot(str(project_id), str(layer_id), tuple(lanes))
