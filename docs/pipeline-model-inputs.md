# Published model inputs and dispatch recovery

Pipeline inference can bind an auxiliary managed model without changing the
existing primary frame port or the one-input-per-frame V1 manifest contract.
Use either `model_input: {version_id: "<artifact-version-uuid>"}` or
`model_input: {producer_id: "<training-operation-uuid>"}` on
`frames.binary-segment.v2` / `frames.confidence.v1`. A producer must be an
explicit dependency with a model output port. Primary image dependencies remain
in `input_operation_id`. The existing dependency scheduler handles both ports.

Models remain ordinary immutable layer attachments. A model has managed bytes,
declared `artifact_role=model` or `plugin_output_role=model`, and a supported
model filename. Conflicting roles, external files, archived series, another
project/layer, unsafe filenames and ambiguous producer outputs are rejected.
This metadata identifies a published model; it does not certify its accuracy or
the safety/compatibility of a model deserializer.

The job parameters pin `model_version_id`, `model_sha256`, `model_filename` and
the existing display label `model_version` (defaults to the version UUID).
Client-supplied hashes/names must match the artifact catalog. Managed model
references reject client source/staging paths. The model series/version joins
`input_heads`, so a replacement head marks the attempt stale according to the
existing workflow policy. Automatic recomputation does not silently select a
different explicitly pinned model; the definition must select the new version.

Submission revalidates the model after project authorization. Model lineage is
included in the quality gate and every imported inference output's
`ArtifactVersion.input_version_ids` alongside its own frame image. A managed
model is an auxiliary port, not another frame input. Existing Agent staging
adapters must fetch the exact version, verify its authoritative SHA-256 and
produce `model_relative_path=inputs/model/<safe filename>`; the application
never translates a model reference into an arbitrary source path.

## AnalysisRun recovery and remaining server gap

Local `AnalysisRunCoordinator.refresh` now repairs queued partition intents
persisted before an interrupted Agent handoff. It resubmits the existing job ID
only after an explicit Agent 404/KeyError. Transport failures retain the queued
intent. Cancelled and failed partitions are excluded from routine polling and
require explicit retry. Cancelling a not-yet-enqueued partition records its
cancellation without creating an Agent job.

This change does **not** implement remote AnalysisRun execution. The current
PostgreSQL analysis adapter is a result projection, not a complete transactional
AnalysisRunStore. Hub still disables remote start. Completing server dispatch
requires authorized start/retry/cancel commands, durable run/partition creation,
the existing server queue's typed analysis-manifest handoff and result-bundle
import under the current lease generation. It must preserve source bindings,
recipe/runtime fingerprints, immutable input hashes and project RBAC. There
must be no second scheduler or queue. Existing local analysis start/polling is
also a UI threading boundary to move off the GUI thread before large runs.

Focused regression tests exercise model port round trips, producer typing,
metadata spoofing/cross-project rejection, authorization ordering, scheduler
secondary binding, actual filesystem job submission/result lineage and local
AnalysisRun crash/cancel recovery. Their model bytes and output bytes are
explicit fixtures; these tests perform no training, inference or quality scoring.

Verified 2026-10-26 with `tools/verify_repository.py` (isolated child processes,
Windows ErrorMode 3): `test_pipeline_model_inputs`,
`test_analysis_dispatch_recovery`, `test_analysis_run_workflow`,
`test_pipeline_execution`, `test_pipeline_runtime_integration`,
`test_pipeline_postgres`, `test_plugin_job_application`, and both manager
architecture modules: **48 passed, 0 skipped, 0 failures**. The PostgreSQL tests
used disposable UUID schemas in the task-owned loopback database. Evidence:
`.codex_tmp_scratch/production-audit/pipeline-model-final/results.json` and its
JUnit/log files. Scoped Ruff checks passed. This gate verifies application
binding/lineage and recovery; managed-model transport staging is verified
separately by its owning Agent integration changes.

## Hub selection

Recognize/confidence actions now open an asynchronous catalog picker with model
name, filename, creation time, immutable version and SHA-256. It shows the
current source version IDs/hashes and states that submission rechecks access
and pins its actual source frame selection. Picking a published model sends
only `model_version_id`; it never exports a managed model to an external path.
The explicit external-file action and the picker's external-file option retain
the previous hash-pinned file workflow. Catalog reads run in the existing Hub
background worker registry and the picker guards close while a read is active.

The layer workflow editor provides named model-version and training-producer
choices, adds the producer dependency automatically, preserves unavailable
pinned references for explicit correction, and displays exact source/model
versions in attempt history. The application still validates all references on
submission; the UI is not an authorization boundary.

UI gate `.codex_tmp_scratch/production-audit/published-model-ui-final`:
5 modules / **50 passed**, including real workflow persistence and existing
server-image/controller tests. No training or geometry evaluation was run.
