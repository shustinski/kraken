"""Local map tasks replay directly from the existing project event stream."""
from copy import deepcopy

from kraken_manager.domain.artifacts import BlobRef
from kraken_core.analysis_runtime import validate_worker_runtime
from .map_validation import read_verified_map, validate_map_bytes


class AnalysisMapStoreMixin:
    def map_tasks(self, run_id):
        tasks = {}
        for event in self.events.load_stream(f"analysis-run:{run_id}"):
            kind = event.data.get("event_type", "")
            if not kind.startswith("analysis_map."):
                continue
            payload = deepcopy(event.data["payload"])
            frame_id = payload["frame_id"]
            if kind == "analysis_map.requested":
                tasks[frame_id] = {**payload, "state": "queued", "acknowledged": False}
            else:
                task = tasks[frame_id]
                if task["manifest"]["job_id"] != payload["job_id"]:
                    raise ValueError("Map event refers to a superseded attempt")
                if kind == "analysis_map.acknowledged":
                    task["acknowledged"] = True
                else:
                    task.update(payload)
                    task["state"] = {"analysis_map.running": "running", "analysis_map.imported": "ready",
                                     "analysis_map.failed": "failed", "analysis_map.cancelled": "cancelled"}[kind]
        return tasks

    def request_map(self, manifest):
        with self.events.lock.hold(self.events.lock_timeout):
            run = self.get_run(manifest.run_id)
            if run is None or run.project_id != manifest.project_id or len(manifest.frames) != 1:
                raise ValueError("Map request does not belong to this project/run")
            frame_id = manifest.frames[0].frame_id
            original = next((frame for part in self.all_partitions(run.run_id) for frame in part.frames
                             if frame.frame_id == frame_id), None)
            if original != manifest.frames[0] or manifest.recipe != run.manifest.recipe:
                raise ValueError("Map request changed the pinned analysis inputs or recipe")
            if not self.metrics_for_frame(run.run_id, frame_id):
                raise ValueError("Map requires a successfully analysed frame")
            original_partition = next(part for part in self.all_partitions(run.run_id) if original in part.frames)
            expected_parameters = tuple(p for p in original_partition.parameters if not p.key.startswith("_"))
            actual_parameters = tuple(p for p in manifest.parameters if p.key != "_render_frame_id")
            requested = tuple(p.value for p in manifest.parameters if p.key == "_render_frame_id")
            if (actual_parameters != expected_parameters or requested != (frame_id,)
                    or manifest.run_fingerprint != run.fingerprint):
                raise ValueError("Map request changed the pinned analysis parameters")
            tasks = self.map_tasks(run.run_id)
            previous = tasks.get(frame_id)
            if previous and previous["manifest"] == manifest.to_payload():
                return previous
            if previous and previous["state"] in {"queued", "running", "ready"}:
                return previous
            for event in self.events.load_stream(f"analysis-run:{manifest.run_id}"):
                payload = event.data.get("payload", {})
                if event.data.get("event_type") == "analysis_map.requested" and payload["manifest"]["job_id"] == manifest.job_id:
                    raise ValueError("Retired map request cannot be replayed as a new attempt")
            self._append_event(manifest.run_id, "analysis_map.requested", {
                "frame_id": frame_id, "manifest": manifest.to_payload(), "error": "",
                "idempotency_key": "analysis-map-request:" + manifest.job_id})
            return self.map_tasks(run.run_id)[frame_id]

    def mark_map_state(self, manifest, state, *, error=""):
        if state not in {"running", "failed", "cancelled"}:
            raise ValueError("Unsupported map state")
        with self.events.lock.hold(self.events.lock_timeout):
            task = self._map_attempt(manifest)
            if task["state"] == state:
                return
            if task["state"] not in {"queued", "running"}:
                raise ValueError("Terminal map result cannot change state")
            self._append_event(manifest.run_id, "analysis_map." + state, {
                "frame_id": manifest.frames[0].frame_id, "job_id": manifest.job_id, "error": str(error)[:10000],
                "idempotency_key": f"analysis-map-state:{manifest.job_id}:{state}"})

    def _map_attempt(self, manifest):
        task = self.map_tasks(manifest.run_id).get(manifest.frames[0].frame_id)
        if task is None or task["manifest"] != manifest.to_payload():
            raise ValueError("Map attempt was superseded or belongs to another run")
        return task

    def import_map(self, manifest, result, data):
        if (result.job_id, result.run_id, result.partition_id, result.project_id) != (
                manifest.job_id, manifest.run_id, manifest.partition_id, manifest.project_id):
            raise ValueError("Rendered map belongs to another partition attempt")
        artifact = result.rendered_map
        if artifact is None or result.outcome.value != "succeeded" or artifact.frame_id != manifest.frames[0].frame_id:
            raise ValueError("Result did not publish the requested map")
        validate_worker_runtime(manifest, result, pinned=self.pinned_runtime(manifest.run_id))
        validate_map_bytes(data, artifact.sha256, artifact.size_bytes)
        with self.events.lock.hold(self.events.lock_timeout):
            task = self._map_attempt(manifest)
            if task["state"] == "ready":
                if task["result"] != result.to_payload():
                    raise ValueError("Repeated map result changed its content")
                return
            if task["state"] not in {"queued", "running"}:
                raise ValueError("Cancelled or failed map cannot import a late result")
            stored = self.blobs.put(data, expected_sha256=artifact.sha256)
            self._append_event(manifest.run_id, "analysis_map.imported", {
                "frame_id": artifact.frame_id, "job_id": manifest.job_id, "result": result.to_payload(),
                "blob": {"sha256": stored.blob.sha256, "size_bytes": stored.blob.size_bytes},
                "idempotency_key": "analysis-map-import:" + manifest.job_id})

    def acknowledge_map(self, manifest):
        with self.events.lock.hold(self.events.lock_timeout):
            task = self._map_attempt(manifest)
            if task["acknowledged"]:
                return
            if task["state"] not in {"ready", "cancelled", "failed"}:
                raise ValueError("Only terminal map tasks can be acknowledged")
            self._append_event(manifest.run_id, "analysis_map.acknowledged", {
                "frame_id": manifest.frames[0].frame_id, "job_id": manifest.job_id,
                "idempotency_key": "analysis-map-ack:" + manifest.job_id})

    def map_bytes(self, run_id, frame_id):
        task = self.map_tasks(run_id).get(frame_id)
        if task is None or task["state"] != "ready":
            raise ValueError("Rendered map is not available")
        return read_verified_map(self.blobs, BlobRef(**task["blob"]))
