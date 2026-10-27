"""Authorized server AnalysisRun commands and durable existing-queue handoff."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version, PackageNotFoundError
from uuid import UUID, uuid5
import platform

from kraken_core.analysis_protocol import AnalysisParameter
from kraken_core.analysis_run_protocol import (
    AnalysisRecipe,
    AnalysisRuntimeIdentity,
    AnalysisRunManifest,
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
)
from kraken_manager.application.analysis_runs import AnalysisRunCoordinator
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.server_analysis import ServerAnalysisWorkflow, resolve_analysis_inputs
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.pipeline import PipelineResources
from kraken_manager.infrastructure.postgres.analysis_workflow import PostgresAnalysisWorkflowStore
from kraken_manager.infrastructure.postgres.analysis_store import PostgresAnalysisProjectionStore


def runtime_identity():
    def installed(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return "not-installed"

    return AnalysisRuntimeIdentity(
        installed("karakal"),
        "server-declared-runtime",
        platform.python_version(),
        installed("numpy"),
        installed("opencv-python"),
        platform.system(),
    )


class AnalysisRunsMixin:
    def _analysis_authorize(self, unit, project_id, actor):
        project = self._planning_project(unit.projections, project_id, mutate=True)
        AuthorizationPolicy().require(
            principal=actor,
            storage=self.profiles.get(project.storage_profile),
            roles=unit.acl.roles_for(project.id, actor.id),
            permission=Permission.RUN_PLUGIN,
            gitlab_identity_verified=True,
        )
        return project

    def _analysis_view(self, state):
        manifest = state["manifest"]
        imported = [part for part in state["partitions"].values() if part["state"] == "imported"]
        measured = bool(imported) and all(part.get("result", {}).get("worker_runtime") for part in imported)
        return {
            "run_id": manifest["run_id"],
            "project_id": manifest["project_id"],
            "layer_id": state["layer_id"],
            "state": state["state"],
            "revision": state["revision"],
            "manifest": manifest,
            "partitions": list(state["partitions"].values()),
            "actor_id": state["actor_id"],
            "resources": state["resources"],
            "maps": state.get("maps", {}),
            "worker_runtime": state.get("worker_runtime") if measured else None,
            "pinned_runtime": state.get("worker_runtime"),
            "runtime_verification": "worker_measured" if measured else "unverified",
            "total_frames": len(manifest["frame_ids"]),
            "completed_frames": sum(p.get("completed_frames", 0) for p in state["partitions"].values()),
            "failed_frames": sum(p.get("failed_frames", 0) for p in state["partitions"].values()),
        }

    def get_analysis_run(self, project_id, run_id):
        try:
            state = ServerAnalysisWorkflow(self.events).load(run_id)
            if str(state["manifest"]["project_id"]) != str(project_id):
                raise NotFoundError("Analysis run was not found in the project")
            return self._analysis_view(state)
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def list_analysis_runs(self, project_id):
        store = PostgresAnalysisProjectionStore(self.engine)
        return [self.get_analysis_run(project_id, run.run_id) for run in store.list_runs_for_project(project_id)]

    def analysis_frame_results(self, project_id, run_id, metric_key):
        self.get_analysis_run(project_id, run_id)
        return list(PostgresAnalysisProjectionStore(self.engine).frame_results(run_id, metric_key))

    def compare_analysis_runs(self, project_id, left_id, right_id, metric_key):
        from kraken_manager.application.quality_analysis import QualityAnalysisService

        left, right = self.get_analysis_run(project_id, left_id), self.get_analysis_run(project_id, right_id)
        if left["state"] != "completed" or right["state"] != "completed":
            raise ConflictError("Comparison requires completed analysis runs")
        a, b = AnalysisRunManifest.from_payload(left["manifest"]), AnalysisRunManifest.from_payload(right["manifest"])
        left_runtime, right_runtime = left.get("worker_runtime"), right.get("worker_runtime")
        if not left_runtime or not right_runtime:
            raise ConflictError("Comparison requires measured worker runtime receipts; repeat legacy analyses")
        if QualityAnalysisService.signature(a, AnalysisRuntimeIdentity.from_payload(left_runtime)) != QualityAnalysisService.signature(
                b, AnalysisRuntimeIdentity.from_payload(right_runtime)):
            raise ConflictError("Comparison requires matching recipe/runtime/parameters")
        store = PostgresAnalysisProjectionStore(self.engine)
        if metric_key not in store.metric_keys(left_id) or metric_key not in store.metric_keys(right_id):
            raise ConflictError("Metric was not recorded in both runs")
        first = {row["frame_id"]: row for row in store.frame_results(left_id, metric_key)}
        second = {row["frame_id"]: row for row in store.frame_results(right_id, metric_key)}
        return {
            "left_run_id": left_id,
            "right_run_id": right_id,
            "metric_key": metric_key,
            "left_sources": left["manifest"]["source_bindings"],
            "right_sources": right["manifest"]["source_bindings"],
            "missing_from_left": sorted(second.keys() - first.keys()),
            "missing_from_right": sorted(first.keys() - second.keys()),
            "frames": [
                {
                    "frame_id": key,
                    "left": first[key]["raw_value"],
                    "right": second[key]["raw_value"],
                    "delta": None
                    if first[key]["raw_value"] is None or second[key]["raw_value"] is None
                    else second[key]["raw_value"] - first[key]["raw_value"],
                }
                for key in sorted(first.keys() & second.keys())
            ],
        }

    def transport_analysis_command(self, project_id, run_id, payload, context):
        try:
            if set(payload) != {"operation"} or payload["operation"] not in {"retry", "cancel", "repeat"}:
                raise ValueError("Analysis command must be retry, cancel or repeat")
            operation, actor = payload["operation"], self._actor(context.actor_id)
            command = {"operation": operation, "run_id": run_id}
            cancelled = []
            with self.uow_factory() as unit:
                self._analysis_authorize(unit, project_id, actor)
                workflow = ServerAnalysisWorkflow(unit.event_store)
                state = workflow.load(run_id)
                if state["manifest"]["project_id"] != str(project_id):
                    raise NotFoundError("Analysis run was not found in the project")
                prior = unit.event_store.find_by_idempotency_key(project_id, context.idempotency_key)
                target_id = (
                    workflow.command_id(project_id, context.idempotency_key) if operation == "repeat" else run_id
                )
                if prior:
                    workflow.append(
                        target_id,
                        project_id,
                        "analysis_run.command",
                        {},
                        principal=actor,
                        expected_revision=context.expected_revision,
                        key=context.idempotency_key,
                        command=command,
                        at=datetime.now(UTC),
                    )
                else:
                    if state["revision"] != context.expected_revision:
                        from kraken_manager.application.errors import ConcurrencyError

                        raise ConcurrencyError("Analysis revision changed")
                    if operation == "repeat":
                        original = AnalysisRunManifest.from_payload(state["manifest"])
                        frames = tuple(
                            frame
                            for part in state["partitions"].values()
                            for frame in AnalysisPartitionJobManifest.from_payload(part["manifest"]).frames
                        )
                        manifest, partitions = AnalysisRunCoordinator.build_contracts(
                            project_id=project_id,
                            frames=frames,
                            source_bindings=original.source_bindings,
                            recipe=original.recipe,
                            runtime=original.runtime,
                            parameters=original.parameters,
                            run_id=target_id,
                            expected_runtime=(AnalysisRuntimeIdentity.from_payload(state["worker_runtime"])
                                if state.get("worker_runtime") is not None else
                                AnalysisPartitionJobManifest.from_payload(next(iter(state["partitions"].values()))["manifest"]).expected_runtime),
                        )
                        workflow.append(
                            target_id,
                            project_id,
                            "analysis_run.created",
                            {
                                "manifest": manifest.to_payload(),
                                "partitions": [p.to_payload() for p in partitions],
                                "actor_id": str(actor.id),
                                "layer_id": state["layer_id"],
                                "resources": state["resources"],
                            },
                            principal=actor,
                            expected_revision=0,
                            key=context.idempotency_key,
                            command=command,
                            at=datetime.now(UTC),
                        )
                    else:
                        replacements = []
                        if operation == "cancel":
                            cancelled.extend(entry["manifest"]["job_id"] for entry in state["maps"].values()
                                             if entry["state"] in {"queued", "running"})
                        for partition in state["partitions"].values():
                            manifest = AnalysisPartitionJobManifest.from_payload(partition["manifest"])
                            if operation == "cancel" and partition["state"] in {"queued", "running"}:
                                cancelled.append(manifest.job_id)
                            if operation == "retry" and (
                                partition["state"] in {"failed", "cancelled"} or partition.get("failed_frames", 0) > 0
                            ):
                                cancelled.append(manifest.job_id)
                                replacements.append(
                                    replace(
                                        manifest,
                                        job_id=str(
                                            uuid5(
                                                UUID(run_id), f"retry:{context.idempotency_key}:{manifest.partition_id}"
                                            )
                                        ),
                                        require_worker_runtime=True,
                                        expected_runtime=(AnalysisRuntimeIdentity.from_payload(state["worker_runtime"])
                                            if state.get("worker_runtime") is not None else manifest.expected_runtime),
                                    ).to_payload()
                                )
                        if operation == "retry" and not replacements:
                            raise ConflictError("No failed or cancelled analysis partition to retry")
                        workflow.append(
                            run_id,
                            project_id,
                            "analysis_run.command",
                            {"operation": operation, "replacements": replacements, "cancel_jobs": cancelled},
                            principal=actor,
                            expected_revision=state["revision"],
                            key=context.idempotency_key,
                            command=command,
                            at=datetime.now(UTC),
                        )
                    PostgresAnalysisWorkflowStore(unit, self.engine).sync(workflow.load(target_id))
                unit.commit()
            for identifier in ServerAnalysisWorkflow(self.events).load(target_id).get("cancel_jobs", ()):
                self.agent_gateway.cancel(identifier)
            if operation != "cancel":
                self.dispatch_analysis_run(project_id, target_id)
            return self.get_analysis_run(project_id, target_id)
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def rebuild_analysis_run(self, project_id, run_id):
        """Operator rebuild over immutable events/blobs; it performs no compute."""
        with self.uow_factory() as unit:
            state = ServerAnalysisWorkflow(unit.event_store).load(run_id)
            if state["manifest"]["project_id"] != str(project_id):
                raise NotFoundError("Analysis run belongs to another project")
            PostgresAnalysisWorkflowStore(unit, self.engine).rebuild(state)
            unit.commit()
        return self.get_analysis_run(project_id, run_id)

    def transport_create_analysis_run(self, project_id, payload, context):
        try:
            if self.agent_gateway is None or not hasattr(self.agent_gateway, "submit_analysis"):
                raise ConflictError("Server Agent queue does not support analysis partitions")
            if set(payload) - {"layer_id", "coordinates", "bindings", "recipe", "parameters", "runtime", "resources"}:
                raise ValueError("Unknown AnalysisRun fields")
            actor = self._actor(context.actor_id)
            run_id = ServerAnalysisWorkflow.command_id(project_id, context.idempotency_key)
            command = {"operation": "create", "request": payload}
            with self.uow_factory() as unit:
                project = self._analysis_authorize(unit, project_id, actor)
                workflow = ServerAnalysisWorkflow(unit.event_store)
                prior = unit.event_store.find_by_idempotency_key(project.id, context.idempotency_key)
                if prior:
                    workflow.append(
                        run_id,
                        project.id,
                        "analysis_run.created",
                        {},
                        principal=actor,
                        expected_revision=0,
                        key=context.idempotency_key,
                        command=command,
                        at=datetime.now(UTC),
                    )
                else:
                    if context.expected_revision not in {None, 0}:
                        raise ConflictError("New analysis runs require revision zero")
                    coordinates = payload["coordinates"]
                    limit = self.profiles.get(project.storage_profile).capabilities.max_frames
                    if limit is not None and len(coordinates) > limit:
                        raise ConflictError("Analysis selection exceeds the storage profile limit")
                    frames, sources = resolve_analysis_inputs(
                        unit.projections, project, payload["layer_id"], coordinates, payload["bindings"]
                    )
                    recipe = AnalysisRecipe.from_payload(payload["recipe"])
                    runtime = (
                        AnalysisRuntimeIdentity.from_payload(payload["runtime"])
                        if "runtime" in payload
                        else runtime_identity()
                    )
                    raw_parameters = payload.get("parameters", {})
                    if not isinstance(raw_parameters, dict):
                        raise ValueError("Analysis parameters must be an object")
                    if any(str(key).startswith("_") for key in raw_parameters):
                        raise ValueError("Reserved analysis parameters cannot be supplied by clients")
                    parameters = tuple(
                        AnalysisParameter(str(key), value) for key, value in sorted(raw_parameters.items())
                    )
                    manifest, partitions = AnalysisRunCoordinator.build_contracts(
                        project_id=str(project.id),
                        frames=frames,
                        source_bindings=sources,
                        recipe=recipe,
                        runtime=runtime,
                        expected_runtime=runtime if "runtime" in payload else None,
                        parameters=parameters,
                        run_id=run_id,
                    )
                    resources = PipelineResources(**payload.get("resources", {})).to_dict()
                    workflow.append(
                        run_id,
                        project.id,
                        "analysis_run.created",
                        {
                            "manifest": manifest.to_payload(),
                            "partitions": [p.to_payload() for p in partitions],
                            "actor_id": str(actor.id),
                            "layer_id": str(payload["layer_id"]),
                            "resources": resources,
                        },
                        principal=actor,
                        expected_revision=0,
                        key=context.idempotency_key,
                        command=command,
                        at=datetime.now(UTC),
                    )
                    PostgresAnalysisWorkflowStore(unit, self.engine).sync(workflow.load(run_id))
                unit.commit()
            self.dispatch_analysis_run(project_id, run_id)
            return self.get_analysis_run(project_id, run_id)
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def dispatch_analysis_run(self, project_id, run_id):
        state = ServerAnalysisWorkflow(self.events).load(run_id)
        if state["manifest"]["project_id"] != str(project_id):
            raise NotFoundError("Analysis run belongs to another project")
        for identifier in state.get("cancel_jobs", ()):
            self.agent_gateway.cancel(identifier)
        if not any(partition["state"] == "queued" for partition in state["partitions"].values()):
            return
        actor = self._actor(state["actor_id"])
        for partition in state["partitions"].values():
            if partition["state"] != "queued":
                continue
            manifest = AnalysisPartitionJobManifest.from_payload(partition["manifest"])
            with self.uow_factory() as unit:
                self._analysis_authorize(unit, project_id, actor)
            # Canonical intent was committed first. Repeating the same queue job
            # repairs a crash between enqueue and acknowledgement.
            self.agent_gateway.submit_analysis(manifest, actor_id=str(actor.id), resources=state["resources"])
            with self.uow_factory() as unit:
                self._analysis_authorize(unit, project_id, actor)
                workflow = ServerAnalysisWorkflow(unit.event_store)
                current = workflow.load(run_id)
                active = current["partitions"][manifest.partition_id]
                if active["state"] != "queued" or active["manifest"]["job_id"] != manifest.job_id:
                    self.agent_gateway.cancel(manifest.job_id)
                    continue
                workflow.append(
                    run_id,
                    project_id,
                    "analysis_partition.dispatched",
                    {"partition_id": manifest.partition_id, "job_id": manifest.job_id},
                    principal=actor,
                    expected_revision=current["revision"],
                    key=None,
                    command={"operation": "dispatch"},
                    at=datetime.now(UTC),
                )
                PostgresAnalysisWorkflowStore(unit, self.engine).sync(workflow.load(run_id))
                unit.commit()

    def run_analysis_cycle(self):
        if self.agent_gateway is None or not hasattr(self.agent_gateway, "submit_analysis"):
            return 0
        count = 0
        for run in PostgresAnalysisProjectionStore(self.engine).list_runs():
            try:
                self.dispatch_analysis_maps(str(run.project_id), run.run_id)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Analysis map dispatch remains pending: %s", run.run_id)
            if run.state == "completed":
                continue
            try:
                self.dispatch_analysis_run(str(run.project_id), run.run_id)
                count += 1
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Analysis dispatch intent remains pending: %s", run.run_id)
        return count

    def import_analysis_result(self, job_id, agent):
        manifest = self.agent_gateway.manifest(job_id, agent)
        publications = self.agent_gateway.publications_for(job_id, agent)
        if not isinstance(manifest, AnalysisPartitionJobManifest) or len(publications) != 1:
            raise ConflictError("Exactly one native analysis partition publication is required")
        result = AnalysisPartitionResultManifest.from_payload(publications[0])
        if any(parameter.key == "_render_frame_id" for parameter in manifest.parameters):
            return self.import_analysis_map(manifest, result)
        if result.bundle is None:
            if (result.job_id, result.run_id, result.partition_id, result.project_id) != (
                manifest.job_id,
                manifest.run_id,
                manifest.partition_id,
                manifest.project_id,
            ):
                raise ConflictError("Analysis failure publication belongs to another partition")
            return self.fail_analysis_job(job_id, result.message or f"Analysis partition {result.outcome.value}", result=result)
        reference = self.agent_gateway.output_blob(job_id, "analysis-bundle")
        with self.uow_factory() as unit:
            workflow = ServerAnalysisWorkflow(unit.event_store)
            state = workflow.load(manifest.run_id)
            actor = self._actor(state["actor_id"])
            self._analysis_authorize(unit, manifest.project_id, actor)
            partition = state["partitions"].get(manifest.partition_id)
            if (
                partition is None
                or partition["manifest"] != manifest.to_payload()
                or partition["state"] not in {"queued", "running", "imported"}
            ):
                raise ConflictError("Analysis partition attempt was cancelled or superseded")
            store = PostgresAnalysisWorkflowStore(unit, self.engine)
            records = store.records(manifest, result, reference)
            from kraken_core.analysis_runtime import validate_worker_runtime
            validate_worker_runtime(manifest, result, pinned=None if state.get("worker_runtime") is None
                                    else AnalysisRuntimeIdentity.from_payload(state["worker_runtime"]))
            workflow.append(
                manifest.run_id,
                manifest.project_id,
                "analysis_partition.imported",
                {
                    "partition_id": manifest.partition_id,
                    "job_id": job_id,
                    "result": result.to_payload(),
                    "blob": {"sha256": reference.sha256, "size_bytes": reference.size_bytes},
                    "completed_frames": sum(frame.status == "ready" for frame in records),
                    "failed_frames": sum(frame.status != "ready" for frame in records),
                },
                principal=actor,
                expected_revision=state["revision"],
                key=f"analysis-import:{job_id}:{reference.sha256}",
                command={"operation": "import", "result": result.to_payload()},
                at=datetime.now(UTC),
            )
            store.projection.import_partition(result, records)
            store.sync(workflow.load(manifest.run_id))
            unit.commit()
        return self.get_analysis_run(manifest.project_id, manifest.run_id)

    def fail_analysis_job(self, job_id, error, *, result=None):
        import sqlalchemy as sa

        projection = PostgresAnalysisProjectionStore(self.engine)
        with self.engine.connect() as connection:
            row = connection.execute(
                sa.select(projection.partitions.c.run_id, projection.partitions.c.partition_id).where(
                    projection.partitions.c.job_id == job_id
                )
            ).one_or_none()
        if row is None:
            manifest = self.agent_gateway.analysis_job_manifest(job_id) if self.agent_gateway is not None else None
            if manifest is not None and any(parameter.key == "_render_frame_id" for parameter in manifest.parameters):
                return self.fail_analysis_map(manifest, error)
            raise NotFoundError("Analysis partition job was not found")
        with self.uow_factory() as unit:
            workflow = ServerAnalysisWorkflow(unit.event_store)
            state = workflow.load(row.run_id)
            actor = self._actor(state["actor_id"])
            self._analysis_authorize(unit, state["manifest"]["project_id"], actor)
            partition = state["partitions"][row.partition_id]
            if partition["state"] not in {"queued", "running"} or partition["manifest"]["job_id"] != job_id:
                raise ConflictError("Analysis attempt is no longer active")
            workflow.append(
                row.run_id,
                state["manifest"]["project_id"],
                "analysis_partition.failed",
                {"partition_id": row.partition_id, "job_id": job_id, "error": str(error)[:10000],
                 **({} if result is None else {"result": result.to_payload()})},
                principal=actor,
                expected_revision=state["revision"],
                key=None,
                command={"operation": "fail"},
                at=datetime.now(UTC),
            )
            PostgresAnalysisWorkflowStore(unit, self.engine).sync(workflow.load(row.run_id))
            unit.commit()
        return self.get_analysis_run(state["manifest"]["project_id"], row.run_id)
