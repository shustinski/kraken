# Executable layer workflow

The existing `layer-pipeline:<layer UUID>` event stream now holds immutable
workflow definition versions and durable execution snapshots alongside its manual
operation history. Its PluginJobs use the existing submission handlers, Agent
queue, result import and artifact versions. WorkPlan remains engineering planning;
a workflow run and its computation attempts are separate records.

In Hub, open layer management and choose **Технологический процесс… / Workflow…**.
Choose named representations, capability, expected plugin version, frame scope,
dependencies and optional input producer. The editor validates types, graph cycles,
representation ownership and resource requests. Save a definition before starting.
Pause stops new dispatch; running jobs continue. Cancel reconciles already reserved
dispatches and cancels their authoritative PluginJobs before the run becomes terminal.
Attempt history and event history remain available after retries and new runs.

Persistent Server automatically advances workflows during its lifespan. Local Hub
advances local graphs in a background worker and resumes discovery after reopening.
Closing the workflow dialog does not stop the controller. Normal application close
guards protect active background persistence and Agent work.

## Contracts

* `GET /api/v1/projects/{project}/layers/{layer}/pipeline/workflow`
* `PUT` at the same URL publishes a definition with `If-Match` and `Idempotency-Key`.
* `POST .../commands` accepts `start`, `pause`, `resume`, `cancel`, `approve`, `retry`.
  Approval/retry identify `node_id`; all mutations require revision and idempotency.

The definition includes `project_id`, `layer_id`, increasing `version`, `operations`
and `policy`. An operation declares a stable UUID, title, existing capability,
expected algorithm/plugin version, source/target representation IDs, typed ports,
frame coordinates, dependencies, optional input producer, parameters, CPU/GPU/memory
reservation, manual approval and quality gate. The policy bounds parallel dispatch,
resource reservations, retry attempts/delay and explicitly permits or forbids stale
recomputation. The default forbids automatic stale recomputation and automatic retry.

Input heads are resolved to immutable version IDs and hashes before dispatch intent
commit. Deterministic job IDs and command keys repair a crash between database commit
and Agent handoff. An acknowledgement is persisted only after handoff returns.
Successful computation without verified imported artifact outputs cannot unblock a
dependent operation. Published tool version must match the pinned expected version.
Changed input heads mark completed results and dependents stale. Historical outputs
are retained. Open substantial quality findings block dispatch when the operation's
quality gate is enabled. Repair/exploratory steps can explicitly disable that gate
in their audited definition; this does not accept the defect or certify connectivity.

Contour vectorization supports explicit `execution_mode: headless` using its normal
processing use case and result publisher. Interactive launching remains available.
Source metadata and the published plugin version must agree.

## Verification and remaining work

`test_pipeline_execution.py` checks graph/type/resource rules, durable replay,
manual confirmation, pause, bounded retry, stale detection and cancellation.
`test_pipeline_runtime_integration.py` uses actual filesystem UOW/job handlers,
including Agent handoff failure, restart and a quality gate. `test_pipeline_postgres.py`
uses migrated fixture-owned PostgreSQL schemas, simultaneous reservations and the
authenticated FastAPI application. `test_layer_pipeline_dialog.py` exercises shown
Qt commands, worker close guards and local continuation outside the dialog.

The explicit acceptance command is:

```powershell
$env:KRAKEN_RUN_PIPELINE_LIVE='1'
.venv/Scripts/python.exe tools/verify_repository.py tests/test_pipeline_live_contour.py --timeout 150 --report .test_runtime/pipeline-live
```

It executes three real Contour jobs on two synthetic images through a real Agent
subprocess, imports six CIF versions, checks the dependency join, reopens Hub's
composition during execution and verifies unchanged sources. It is not physical
ground truth or a throughput test. The recording-gateway tests separately establish
transport behavior and do not claim an algorithm execution.

Resource declarations bound reservations within one workflow. With all four explicit
queue admission limits configured, the Server also serializes logical reservations
across its PostgreSQL job queue. An unconfigured queue reports `unenforced`. These
are not OS-enforced memory limits or allocation across physical GPU hosts. Multi-input model
ports, Karakal AnalysisRun dispatch, remote multi-agent failure injection, long-run
load, global allocation fairness and release/installer acceptance remain open.
