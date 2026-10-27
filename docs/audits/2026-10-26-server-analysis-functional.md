# Server AnalysisRun functional gate

This is functional verification on controlled binary fixtures, without training
or evaluation of microscopy/vectorization quality. Full production readiness
remains unconfirmed.

The existing PostgreSQL Agent queue now accepts native Karakal partition jobs.
An internal queue envelope retains actor, capability and resource reservations;
the Agent receives the native manifest. The same generation claim fences input
downloads, bundle uploads, publications and completion. Source inputs are resolved
on the server from immutable project/layer representation versions.

Canonical `analysis-run:<uuid>` events retain creation, dispatch, cancellation,
retry and imported bundle references. Existing `analysis_*` PostgreSQL tables
are transaction-bound projections. A retry gets a new job ID; old events/blobs
remain available. Only the superseded partition's disposable frame/metric rows
are cleared. Explicit projection replay verifies bundle SHA/size and exact frame
coordinates before restoring measurements.

Implemented HTTP routes:

- `POST/GET /api/v1/projects/{project_id}/analysis-runs`
- `GET /api/v1/projects/{project_id}/analysis-runs/{run_id}`
- `GET .../{run_id}/frames?metric_key=...`
- `GET .../{run_id}/compare/{other_run_id}?metric_key=...`
- `POST .../{run_id}/commands` with `retry`, `cancel` or `repeat`

Mutations require project authorization and an idempotency key; commands also
require the current revision. Hub's remote client and dual catalog expose these
operations. `RemoteAnalysisController` now connects the existing workspace panel
to these routes without creating a local analysis store. Network calls and CSV
exports run in retained background threads. Lost mutation replies can be retried
with the same idempotency key; parent destruction cannot destroy a running thread.

Evidence is under `.codex_tmp_scratch/production-audit-20261026/`:

- `server-analysis-recovery`: real Karakal subprocess over HTTP, PostgreSQL import,
  reopen/replay, cancellation, retry/repeat and immutable queue conflict checks.
- `server-analysis-errors-verified`: the same actual worker receives a corrupt
  PNG. It records failure instead of inventing a metric; retry does not duplicate
  frame rows, and replay preserves the current attempt. Adjacent queue, quality
  projection and complete Hub/Agent server-workspace suites also pass.
- `functional-analysis-and-service`: native workflow and remote client **3 tests
  passed**, alongside dispatch-recovery, published-model UI and service-startup
  tests. Earlier red receipts are retained; fixture corrections did not weaken
  assertions or suppress failures.
- `pinned-model-authorization-verified`: exact auxiliary model download/ticket
  checks with real PostgreSQL; stale claim, wrong project/layer/role/hash/name
  are refused. Manually published artifact parameters now persist through the
  existing server upload handler instead of being silently discarded.
- `remote-analysis-acceptance`: **5 tests passed**. Three Qt transport tests cover
  responsiveness, CSV, lost replies and parent destruction. Two tests use the real
  migrated PostgreSQL backend and HTTP routes: actual Qt creation/cancellation,
  anonymous/outsider refusal, viewer read-only access, stale revision refusal and
  project scope isolation. These are functional checks, not quality measurements.

The remote discrepancy map is now implemented and verified through the real
worker/HTTP/PostgreSQL and Qt paths: [map acceptance](2026-10-26-analysis-map-functional.md).
The local map is also implemented and verified: [local map acceptance](2026-10-26-local-analysis-map-functional.md).
Measured worker receipts, pin/refusal, comparison/recheck and runtime UI are
implemented: [runtime acceptance](2026-10-26-worker-runtime-functional.md).
Remaining work includes broader concurrent/fault testing,
fresh packaging and the complete cross-subsystem acceptance scenario. Server-
declared runtime metadata is not attestation of an external worker's environment.
