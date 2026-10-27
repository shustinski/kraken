# Engineering process templates

A process template creates planned engineering work in an existing recovery
plan. It does not launch algorithms, submit plugin jobs or replace the existing
layer processing graph. Generated work can later reference existing WorkOrders
and PluginJobs through the ordinary planning commands.

## Author and apply in Hub

Open recovery plans, choose **Create template**, enter a name, and add steps.
For each step select its scope, planned effort and an optional mandatory
acceptance criterion. A step can wait for acceptance of a previous step or be a
child of a broader stage. The editor rejects acceptance deadlocks.

Choose a saved template and a plan. Enter the selected layer IDs in the plan's
layer field. For a template with frame steps, also enter explicit coordinates
such as `1,1; 2,1` in the template frame field. Optionally choose the performer
using the plan's performer field; this assigns the generated steps to that
performer. **Generate work items** creates the complete batch in one change.
The work remains planned until the usual execution and acceptance commands.

**New template version** preserves the old definition and publishes its next
version. Existing work keeps its original version and content digest. Applying
a newer definition is a separate explicit action and creates another batch;
it does not alter existing work or acceptance evidence.

## Scope and dependency semantics

- `project`: one work item for the entire selected scope.
- `layer`: one work item per selected layer.
- `frame`: one work item per selected frame per selected layer. Frame selection
  is mandatory; the service never silently expands the whole project grid.

Dependencies join matching scopes. Frame-to-frame edges join the same layer
and frame; layer-to-frame edges fan out within the same layer; frame-to-layer
edges fan in within the same layer; a project-scoped successor waits for all
matching lower-scope predecessors. A parent must have the same or broader scope
than its children. Children must be accepted before their parent is accepted.

These are engineering planning relationships. Matching logical frame IDs does
not estimate physical image registration or establish electrical connectivity.

A materialization is bounded to 10,000 work items and 100,000 dependencies.
Oversized requests fail without inserting a partial plan. Split the selected
scope into deliberate batches when necessary.

## API

Project membership is required to read definitions. Publishing and applying
require the existing `ASSIGN_WORK` permission. Mutations use `Idempotency-Key`;
updates also use `If-Match` containing the expected template version or plan
revision. GitLab mutations retain the existing transport live identity check.

- `GET /api/v1/projects/{project_id}/work-plan-templates`
- `GET /api/v1/projects/{project_id}/work-plan-templates/{template_id}?version=1`
- `POST /api/v1/projects/{project_id}/work-plan-templates` creates version 1.
- `PUT /api/v1/projects/{project_id}/work-plan-templates/{template_id}` publishes
  the next immutable version, with `If-Match` set to the preceding version.

Example definition:

```json
{
  "title": "Mask reconstruction",
  "tasks": [
    {"key": "segment", "title": "Segment image", "multiplicity": "frame", "effort_hours": 0.5},
    {"key": "review", "title": "Review mask", "multiplicity": "frame",
     "criteria": [{"description": "Engineer verified versioned mask", "required": true}]},
    {"key": "stitch", "title": "Stitch layer", "multiplicity": "layer"}
  ],
  "dependencies": [
    {"predecessor_key": "segment", "successor_key": "review"},
    {"predecessor_key": "review", "successor_key": "stitch"}
  ]
}
```

Apply through the existing plan command endpoint:

```json
{
  "operation": "apply_template",
  "arguments": {
    "template_id": "<template UUID>",
    "version": 1,
    "scope": {"layer_ids": ["<layer UUID>"], "frame_coordinates": [[1, 1], [2, 1]]},
    "assignments": {"review": {"performer_id": "<performer UUID>"}}
  }
}
```

Scope layers, performers and template ownership are resolved against the
project. Definitions cannot inject work IDs, runtime state, job references,
reviewer identity or acceptance evidence. Unknown fields are rejected.

## Durability and audit

Definitions are snapshots in `work-template:<UUID>` streams in the existing
EventStore. Every published version records its author and command identity.
Application appends exactly one `work_plan.apply_template` event to the existing
plan stream. The plan records the source template UUID, version, SHA256 digest,
application UUID and generated item UUIDs. Item IDs are deterministic for an
idempotent command. A retry with the same key and arguments replays its original
result; a different command using that key conflicts.

Both filesystem and PostgreSQL use their existing UnitOfWork, without a new
schema, queue or scheduler. Concurrent changes use the event-store revision
compare-and-swap. Historical plan payloads without template provenance continue
to load as before.

Tests: `tests/test_work_templates.py`, `tests/test_work_templates_dialog.py` and
`tests/test_work_templates_postgres.py`. PostgreSQL tests require
`KRAKEN_TEST_POSTGRES_URL` and create isolated UUID schemas. Use
`tools/verify_repository.py` to run each test module in a separate process.
