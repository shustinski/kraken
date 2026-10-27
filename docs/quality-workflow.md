# Quality issues and review decisions

Quality findings are durable `quality-issue:<UUID>` streams in the existing
project EventStore. The local filesystem unit of work and PostgreSQL unit of
work supply authorization, compare-and-swap, idempotency and atomic persistence.
PostgreSQL also publishes these events through the existing transactional outbox.
No second artifact repository, execution queue or CIF package workflow is created.

An issue identifies the affected `WorkScope` and immutable `source_version_ids`.
Scope can contain project layers, frame IDs or `frame_coordinates`, and a region
with an explicit coordinate system. Artifact references are resolved against the
project and checked against its layer/frame scope. Optional `review_batch_id`
and assignment `work_order_id` link existing workflows in the same project and
affected layers. Those links are associations; they do not automatically complete
the order, accept a CIF batch, or assert that its entire selection is corrected.

The implemented lifecycle is:

1. `create`: record the finding, severity, description, scope and affected versions.
2. `assign`: choose an active performer and optionally an existing correction order.
3. `correct`: record new immutable versions for every affected artifact series.
4. `recheck`: an independent reviewer records evidence, a note and a boolean outcome.
5. `accept`: close only after a successful recheck; `reject` returns work for reassignment.
6. `exception`: a reviewer records an explicit, justified waiver instead of acceptance.

Accepted issues and approved exceptions are immutable. Rejected findings can be
assigned again, and all prior correction/review decisions remain in history.
An independent reviewer must differ from the author who recorded the correction.
These are recorded human review decisions; `passed: true` does not itself execute
or attest to a new Karakal analysis. Findings created from analysis use the
measured recheck workflow below and reject a manually supplied `passed` value.

Permissions use the existing role catalog:

| Command | Permission |
| --- | --- |
| Create, recheck | `manage_review` |
| Assign | `assign_work` |
| Correct | `return_review` |
| Accept, reject, exception | `accept_review` |

The selected `X-Kraken-Role` is checked for the actual command in the application
service. Inactive principals, foreign references, stale revisions, missing evidence,
unjustified decisions and reuse of an idempotency key for another command are rejected.
A linked assignee can submit their own correction; another corrector cannot submit
it on their behalf. A user with `manage_review` may record a delegated/manual return,
with their own identity preserved as the correction author.

## Local and server entry points

`EmbeddedProjectService` exposes `create_quality_issue`, `mutate_quality_issue`,
`get_quality_issue`, `list_quality_issues`, and `quality_issue_history`.

The authenticated server provides:

| Method | Path relative to `/api/v1/projects/{project_id}` |
| --- | --- |
| POST | `/quality-issues` |
| GET | `/quality-issues` |
| GET | `/quality-issues/{issue_id}` |
| GET | `/quality-issues/{issue_id}/history` |
| POST | `/quality-issues/{issue_id}/commands` |

Create requests require `Idempotency-Key`. Mutation requests require both
`Idempotency-Key` and `If-Match`, containing the current issue revision (zero on
creation). Identical retries return the original result even after later changes.
Different requests using the same key conflict. A mutation body contains only
`operation` and `arguments`.

Example creation body:

```json
{
  "title": "Interrupted conductor",
  "description": "Continuity is absent at the marked region",
  "severity": "major",
  "scope": {"layer_ids": ["<layer UUID>"], "frame_coordinates": [[1, 1]]},
  "source_version_ids": ["<immutable artifact version UUID>"]
}
```

Both list APIs support `artifact_version_id` and `frame_id` filters. Artifact
lookup includes original versions, correction versions and historical review
evidence. History includes every snapshot, actor, operation, timestamp and stream
revision. Quality data survives rebuilding the disposable SQLite projection cache.

## Verification and remaining scope

`tests/test_quality_workflow.py` covers local transactions, review outcomes,
rollback/retry, concurrent writers, forbidden references, role checks, reverse
lookup, cache rebuild and the real FastAPI application routes. Real PostgreSQL
tests in `tests/test_quality_postgres.py` require `KRAKEN_TEST_POSTGRES_URL`; they
use fixture-owned UUID schemas, Alembic migrations and temporary files, then drop
only those schemas. They verify concurrent CAS, SQL rollback including the outbox
and idempotency table, and the complete authenticated HTTP acceptance lifecycle.

Hub now includes a dedicated asynchronous quality desk, local/remote client routing and
artifact/performer selection. `test_quality_dialog.py` and `test_quality_client.py`
exercise the shown UI and transport. Open major/critical/blocker issues gate matching
plan acceptance and pipeline inputs; an exception is an explicit audited review decision.

Automatic issue creation and recheck from completed AnalysisRun metrics still require
an explicitly selected measurement rule; the implemented controls and API below
consume actual stored results without generating substitute measurements.

## Measured findings and rechecks

The quality desk offers **Load runs**, a computed metric selector, an explicit
comparison and threshold, and a target source binding. The threshold starts empty.
Selecting **Create findings from measurements** evaluates the selected completed
run and creates one finding for each failing frame. The ordinary severity selector
applies to those findings. No universal engineering threshold is supplied.

The target binding must participate in the run's analysis expression. A finding
references only that target's immutable artifact versions; unaffected comparison
inputs remain pinned by the origin run. Its rule records the metric key, raw or
normalized value field, comparator, threshold, target binding and computation
signature. The description records the actual observed value and frame coordinates.

For example, a user-selected rule can be sent as:

```json
{
  "run_id": "<completed AnalysisRun ID>",
  "rule": {
    "metric_key": "iou",
    "value_field": "raw_value",
    "comparison": "lt",
    "threshold": 0.9,
    "target_binding": "A"
  },
  "severity": "major"
}
```

Here `lt` means that a measured value below the chosen threshold creates a finding;
the numeric value is an example, not a default. Supported comparisons are `lt`,
`le`, `gt`, and `ge`; value fields are `raw_value` and `goodness`.

Creation and recheck require a fully imported completed run, exact frame coverage,
ready records, matching coordinates and source metadata, and finite measured values.
Missing metrics, failed frames, incomplete imports and inconsistent manifests are
errors. A zero-finding evaluation is still recorded in the project event log, so
its idempotency key cannot later be silently reused with a different rule.

After the assigned correction has been recorded, an independent reviewer chooses
the actual replacement run and **Verify correction with selected run**. The service
requires the same recipe, parameters and runtime identity; every recorded correction
version must occur as the measured target input and every unaffected reference input
must remain unchanged. The service calculates the verdict from stored observations
and the original rule. The API accepts only `run_id` and a reviewer `note`, never
a client-supplied score or verdict. Review history records the run ID, measured values
and affected frames. Acceptance still requires an authorized independent review.

Additional endpoints relative to `/api/v1/projects/{project_id}`:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/quality-artifacts?layer_id=...&x=...&y=...` | Read actual versions of one frame without scanning every frame's versions |
| GET | `/quality-analysis-runs` | List project-owned runs and their actually computed metric keys |
| GET | `/quality-analysis-runs/{run_id}/readiness` | Validate completion and quality readiness of that run's actual inputs |
| POST | `/quality-issues/from-analysis` | Evaluate the explicit rule and atomically record findings/evaluation receipt |
| POST | `/quality-issues/{issue_id}/analysis-recheck` | Recheck recorded correction versions against an actual replacement run |

Measured mutations use the existing permissions, `Idempotency-Key`, and UOW. Recheck
also requires `If-Match`. Local, remote and DualCatalog service methods share the
same contracts; the Hub performs I/O in background workers.

## Input readiness and immutable defect lineage

`QualityService.require_clear_inputs` and `blocking_issues` support actual version
IDs and optional projections/scope. Project-owned provenance is validated, including
parent/input edges, cycles and missing references. Matching unresolved major,
critical and blocker findings prevent dispatch or acceptance; an authorized exception
is explicit and audited.

Acceptance certifies the recorded correction branch. It does not repair the bytes
of the original immutable defective version. Therefore direct use of that version,
an unrelated recomputation from it, or a mixed input that still consumes it remains
blocked. A descendant of the accepted correction is allowed: traversal stops at the
specific approved correction IDs for that finding, while other input branches are
still checked. Scope-based acceptance can proceed with accepted correction evidence.

Run readiness reports `computational_ready`, `ready`, a reason and blocking issue
IDs. A corrected measured run can become ready after acceptance while the original
defective-input run remains unsuitable for reuse.

## Disposable analysis cache migration

Local analysis originally shared `.index/read.sqlite3` with metadata projections.
An ordinary metadata rebuild could replace that file without analysis tables.
Analysis now uses `.index/analysis.sqlite3`, a disposable projection reconstructed
from existing `analysis-run:*` events and immutable record-bundle blobs. Canonical
authority remains in the same project log and BlobStore; PostgreSQL uses its
existing analysis tables and transactional metadata database.

Before a legacy metadata cache can be replaced, analysis migration validates its
run IDs and imported partition state/hash against canonical events and replays the
actual bundles. If runs or imported results lack canonical history, migration stops
with an explicit error and preserves the old cache. A failed replay never leaves
an apparently successful empty migrated cache. SQLite connections are closed before
Windows file replacement. Repeated project reads and metadata rebuilds do not erase
analysis results; deleting the disposable analysis cache reconstructs those results
from canonical data on the next open.

`test_quality_analysis.py` and `test_quality_analysis_postgres.py` run the actual
Karakal headless worker on known masks, import its compressed result bundles, and
verify measured IoU changes from 0 to 1 through local and PostgreSQL/HTTP/remote
workflows. They also cover incomparable runtimes, changed reference inputs, spoofed
manual verdicts, missing records, atomic rollback, concurrent evaluation receipts,
zero findings, complete legacy replay and preservation of unrecorded legacy data.
`test_quality_accepted_lineage.py` verifies the acceptance barrier on real immutable
artifact versions and derived artifacts. The dialog test exercises the measured
workflow in a separate offscreen process with Windows crash dialogs disabled.
