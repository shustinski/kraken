"""Local on-demand map workflow using the existing analysis Agent gateway."""
from dataclasses import replace
import logging
from uuid import UUID, uuid5

from kraken_core.analysis_protocol import AnalysisParameter
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest


class AnalysisMapsCoordinatorMixin:
    def request_map(self, run_id, frame_id, *, idempotency_key):
        partition = next((part for part in self.store.all_partitions(run_id)
                          if any(frame.frame_id == frame_id for frame in part.frames)), None)
        if partition is None:
            raise ValueError("Frame does not belong to this analysis run")
        identifier = str(uuid5(UUID(run_id), "analysis-map:" + idempotency_key))
        manifest = replace(partition, job_id=identifier, partition_id=identifier, partition_index=0, partition_count=1,
                           frames=tuple(frame for frame in partition.frames if frame.frame_id == frame_id),
                           parameters=tuple(p for p in partition.parameters if not p.key.startswith("_"))
                           + (AnalysisParameter("_render_frame_id", frame_id),), require_worker_runtime=True,
                           expected_runtime=self.store.pinned_runtime(run_id))
        task = self.store.request_map(manifest)
        self.refresh_maps(run_id)
        return self.store.map_tasks(run_id)[task["frame_id"]]

    def refresh_maps(self, run_id):
        tasks = getattr(self.store, "map_tasks", lambda _id: {})(run_id)
        for task in tasks.values():
            if task["acknowledged"]:
                continue
            manifest = AnalysisPartitionJobManifest.from_payload(task["manifest"])
            try:
                try:
                    job = self.gateway.get(manifest.job_id)
                except KeyError:
                    if task["state"] in {"ready", "failed", "cancelled"}:
                        self.store.acknowledge_map(manifest)
                        continue
                    job = self.gateway.submit(manifest, self._source_paths(manifest))
                if task["state"] == "ready":
                    if job.state == "importing":
                        self.gateway.complete_import(job)
                    elif job.state in {"queued", "staging", "running", "partial"}:
                        continue
                    self.store.acknowledge_map(manifest)
                    continue
                if task["state"] in {"failed", "cancelled"}:
                    if job.state not in {"failed", "cancelled", "succeeded", "recovery_required"}:
                        self.gateway.cancel(job)
                    self.store.acknowledge_map(manifest)
                    continue
                if task["state"] == "queued":
                    self.store.mark_map_state(manifest, "running")
                if job.state == "importing":
                    result, data = self.gateway.map_result(manifest)
                    self.store.import_map(manifest, result, data)
                    self.gateway.complete_import(job)
                    self.store.acknowledge_map(manifest)
                elif job.state in {"failed", "cancelled", "recovery_required", "partial"}:
                    self.store.mark_map_state(manifest, "cancelled" if job.state == "cancelled" else "failed",
                                              error=job.error or "Worker did not complete the requested map")
            except ValueError as exc:
                current = self.store.map_tasks(run_id)[manifest.frames[0].frame_id]
                if current["state"] in {"queued", "running"}:
                    self.store.mark_map_state(manifest, "failed", error=str(exc))
                else:
                    raise
            except (OSError, RuntimeError) as exc:
                # Uncertain transport state never revokes the durable intent.
                logging.getLogger(__name__).warning("Map %s remains pending: %s", manifest.job_id, exc)

    def cancel_maps(self, run_id):
        for task in getattr(self.store, "map_tasks", lambda _id: {})(run_id).values():
            if task["state"] in {"queued", "running"}:
                manifest = AnalysisPartitionJobManifest.from_payload(task["manifest"])
                self.store.mark_map_state(manifest, "cancelled")
        self.refresh_maps(run_id)
