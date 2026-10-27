"""On-demand maps through the existing analysis queue, event log and blob store."""
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid5

from kraken_core.analysis_protocol import AnalysisOutcome, AnalysisParameter
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest, AnalysisRuntimeIdentity
from kraken_manager.application.errors import ConflictError, ConcurrencyError, NotFoundError
from kraken_manager.application.server_analysis import ServerAnalysisWorkflow
from kraken_manager.domain.artifacts import BlobRef


class AnalysisMapsMixin:
    def transport_request_analysis_map(self, project_id, run_id, frame_id, context):
        try:
            actor = self._actor(context.actor_id)
            command = {"operation": "render_map", "frame_id": frame_id}
            with self.uow_factory() as unit:
                self._analysis_authorize(unit, project_id, actor)
                workflow = ServerAnalysisWorkflow(unit.event_store)
                state = workflow.load(run_id)
                if state["manifest"]["project_id"] != str(project_id):
                    raise NotFoundError("Analysis run was not found in the project")
                prior = unit.event_store.find_by_idempotency_key(project_id, context.idempotency_key)
                if prior:
                    workflow.append(run_id, project_id, "analysis_map.requested", {}, principal=actor,
                                    expected_revision=context.expected_revision, key=context.idempotency_key,
                                    command=command, at=datetime.now(UTC))
                else:
                    if state["revision"] != context.expected_revision:
                        raise ConcurrencyError("Analysis revision changed")
                    original = next((AnalysisPartitionJobManifest.from_payload(part["manifest"])
                                     for part in state["partitions"].values()
                                     if any(frame["frame_id"] == frame_id for frame in part["manifest"]["frames"])), None)
                    if original is None:
                        raise NotFoundError("Frame does not belong to this analysis run")
                    if not any(row["frame_id"] == frame_id and row["status"] == "ready"
                               for row in self.analysis_frame_results(project_id, run_id, (state["manifest"]["recipe"]["metric_keys"] or ["xor"])[0])):
                        raise ConflictError("A map requires a successfully analysed frame")
                    identifier = str(uuid5(UUID(run_id), "map:" + context.idempotency_key))
                    manifest = replace(original, job_id=identifier, partition_id=identifier,
                                       partition_index=0, partition_count=1,
                                       frames=tuple(frame for frame in original.frames if frame.frame_id == frame_id),
                                       parameters=tuple(p for p in original.parameters if not p.key.startswith("_"))
                                       + (AnalysisParameter("_render_frame_id", frame_id),), require_worker_runtime=True,
                                       expected_runtime=None if state.get("worker_runtime") is None
                                       else AnalysisRuntimeIdentity.from_payload(state["worker_runtime"]))
                    workflow.append(run_id, project_id, "analysis_map.requested",
                                    {"frame_id": frame_id, "manifest": manifest.to_payload(), "actor_id": str(actor.id), "error": "",
                                     "cancel_job_id": state["maps"].get(frame_id, {}).get("manifest", {}).get("job_id")},
                                    principal=actor, expected_revision=state["revision"], key=context.idempotency_key,
                                    command=command, at=datetime.now(UTC))
                unit.commit()
            self.dispatch_analysis_maps(project_id, run_id)
            return self.get_analysis_run(project_id, run_id)
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def dispatch_analysis_maps(self, project_id, run_id):
        if self.agent_gateway is None:
            return
        state = ServerAnalysisWorkflow(self.events).load(run_id)
        if state["manifest"]["project_id"] != str(project_id):
            raise NotFoundError("Analysis run was not found in the project")
        for frame_id, entry in state["maps"].items():
            if entry["state"] != "queued":
                continue
            manifest = AnalysisPartitionJobManifest.from_payload(entry["manifest"])
            actor = self._actor(entry["actor_id"])
            with self.uow_factory() as unit:
                self._analysis_authorize(unit, project_id, actor)
            if entry.get("cancel_job_id"):
                self.agent_gateway.cancel(entry["cancel_job_id"])
            self.agent_gateway.submit_analysis(manifest, actor_id=actor.id, resources=state["resources"])
            with self.uow_factory() as unit:
                workflow = ServerAnalysisWorkflow(unit.event_store)
                current = workflow.load(run_id)
                active = current["maps"].get(frame_id)
                if active is None or active["manifest"] != manifest.to_payload():
                    self.agent_gateway.cancel(manifest.job_id)
                    continue
                if active["state"] != "queued":
                    continue
                workflow.append(run_id, project_id, "analysis_map.dispatched",
                                {"frame_id": frame_id, "job_id": manifest.job_id}, principal=actor,
                                expected_revision=current["revision"], key=None, command={"operation": "dispatch_map"}, at=datetime.now(UTC))
                unit.commit()

    def _active_map(self, state, manifest):
        if state["manifest"]["project_id"] != manifest.project_id:
            raise ConflictError("Map belongs to another project")
        frame_id = next((p.value for p in manifest.parameters if p.key == "_render_frame_id"), None)
        entry = state["maps"].get(frame_id)
        if entry is None or entry["manifest"] != manifest.to_payload() or entry["state"] not in {"queued", "running", "ready"}:
            raise ConflictError("Map attempt was superseded or is no longer active")
        return frame_id, entry

    def import_analysis_map(self, manifest, result):
        if (result.job_id, result.run_id, result.partition_id, result.project_id) != (
                manifest.job_id, manifest.run_id, manifest.partition_id, manifest.project_id):
            raise ConflictError("Rendered map result belongs to another attempt")
        if result.outcome != AnalysisOutcome.SUCCEEDED:
            return self.fail_analysis_map(manifest, result.message or "Map rendering failed")
        artifact = result.rendered_map
        if artifact is None or len(manifest.frames) != 1 or artifact.frame_id != manifest.frames[0].frame_id:
            raise ConflictError("Worker did not publish the requested frame map")
        reference = self.agent_gateway.output_blob(manifest.job_id, "analysis-map")
        if (reference.sha256, reference.size_bytes) != (artifact.sha256, artifact.size_bytes):
            raise ConflictError("Rendered map hash/size differs from registered output")
        with self.uow_factory() as unit:
            workflow = ServerAnalysisWorkflow(unit.event_store)
            state = workflow.load(manifest.run_id)
            frame_id, entry = self._active_map(state, manifest)
            actor = self._actor(entry["actor_id"])
            self._analysis_authorize(unit, manifest.project_id, actor)
            from kraken_core.analysis_runtime import validate_worker_runtime
            validate_worker_runtime(manifest, result, pinned=None if state.get("worker_runtime") is None
                                    else AnalysisRuntimeIdentity.from_payload(state["worker_runtime"]))
            self._verified_map_bytes(unit.blobs, reference)
            workflow.append(manifest.run_id, manifest.project_id, "analysis_map.imported",
                            {"frame_id": frame_id, "job_id": manifest.job_id,
                             "blob": {"sha256": reference.sha256, "size_bytes": reference.size_bytes}, "result": result.to_payload()},
                            principal=actor, expected_revision=state["revision"], key="analysis-map-import:" + manifest.job_id,
                            command={"operation": "import_map", "result": result.to_payload()}, at=datetime.now(UTC))
            unit.commit()
        return self.get_analysis_run(manifest.project_id, manifest.run_id)

    def fail_analysis_map(self, manifest, error):
        with self.uow_factory() as unit:
            workflow = ServerAnalysisWorkflow(unit.event_store)
            state = workflow.load(manifest.run_id)
            frame_id, entry = self._active_map(state, manifest)
            if entry["state"] == "ready":
                raise ConflictError("A completed map cannot fail")
            actor = self._actor(entry["actor_id"])
            self._analysis_authorize(unit, manifest.project_id, actor)
            workflow.append(manifest.run_id, manifest.project_id, "analysis_map.failed",
                            {"frame_id": frame_id, "job_id": manifest.job_id, "error": str(error)[:10000]},
                            principal=actor, expected_revision=state["revision"], key=None,
                            command={"operation": "map_failed"}, at=datetime.now(UTC))
            unit.commit()
        return self.get_analysis_run(manifest.project_id, manifest.run_id)

    @staticmethod
    def _verified_map_bytes(blobs, reference):
        from kraken_manager.infrastructure.analysis.map_validation import read_verified_map
        try:
            return read_verified_map(blobs, reference)
        except (OSError, ValueError) as exc:
            raise ConflictError(str(exc)) from exc

    def analysis_map_bytes(self, project_id, run_id, frame_id):
        try:
            self.get_analysis_run(project_id, run_id)
            state = ServerAnalysisWorkflow(self.events).load(run_id)
            entry = state["maps"].get(frame_id)
            if entry is None or entry["state"] != "ready":
                raise NotFoundError("Rendered map is not available for this frame")
            return self._verified_map_bytes(self.uow_factory.blobs, BlobRef(**entry["blob"]))
        except Exception as exc:
            self._translate_planning_error(exc)
            raise
