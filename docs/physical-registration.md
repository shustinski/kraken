# Physical registration and contact evidence

The physical model uses the existing immutable artifacts and project event store.
It does not replace layer stitching attachments, import frame numbers as physical
coordinates, or introduce a separate database. Each `MultilayerModelChangedV1`
event records a complete versioned snapshot, actor, command fingerprint and
idempotency key in the `multilayer:{project_id}` stream. Writes require a matching
aggregate revision; the normal unit of work also commits the PostgreSQL outbox.

The local, remote Hub and persistent Server expose `get_multilayer_model` and
`mutate_multilayer_model`. HTTP endpoints are:

* `GET /api/v1/projects/{project_id}/multilayer`
* `POST /api/v1/projects/{project_id}/multilayer/commands`, with
  `Idempotency-Key` and `If-Match`.

The command body contains `operation` and `arguments`. Supported operations are
`set_stack`, `calibrate`, `register_affine`, `propose_contact`, and `review_contact`.
Structure and registration writes require `MANAGE_STRUCTURE`; contact reviews
require `ACCEPT_REVIEW`. The reviewer identity comes from authentication. External,
archived, foreign-layer and inactive input versions are rejected. Inputs are
anchored by artifact UUID and SHA-256. Contact evidence must be a managed version
on the separate contact layer and cannot reuse an input UUID or its content hash.

All physical lengths are **metres**. Stack layers have an explicit Z, thickness,
technology type and conductor designation. Z order, uniqueness and nonoverlapping
thickness intervals are checked. Conductivity is an engineering assertion, not
inferred from layer order. Calibration records pixel scales, physical origin, Y
axis direction and optional declared uncertainty against a specific input version.
Unknown uncertainty remains `null`.

`register_affine` takes source/target layer IDs, explicit corresponding pixel
coordinates and `max_error_m`. The infrastructure solver lazily loads NumPy from
the `registration` extra, converts points through their respective calibrations,
normalizes coordinates and solves least squares by SVD. It checks rank, design and
transform condition, invertibility and every control residual. Fewer than three
points produce an auditable `insufficient` result without a transform. Degenerate,
ill-conditioned and excessive-residual inputs produce `rejected` results without a
transform. Outlier observations are never silently removed.

RMS and maximum residual are measured **at the supplied controls**. If all input
uncertainties are supplied, an allowance adds the observed maximum residual,
source calibration uncertainty amplified by the fitted linear operator, target
uncertainty and maximum declared correspondence uncertainty. Its scope is
`supplied_control_points_only`: it is not a certified global registration bound,
statistical confidence interval, or accuracy guarantee away from those controls.
Three exact controls provide no independent residual validation degrees of freedom.

Changing a calibration, stack or active source artifact makes the affected
registration and contact assessment stale. A newer registration supersedes the
previous revision. The old coefficients, controls, reviews and evidence remain in
history. No automatic recomputation runs. Idempotent responses retain their original
snapshot but assess present currentness against the latest model and active inputs.

The Hub Management menu opens **Physical layers and contacts**. It supports named
layers, managed layer-attachment revisions with SHA-256, physical stack editing,
calibration, manually entered control points, fit diagnostics, contact candidates
and independent evidence review. It deliberately leaves missing measurements blank.
API calibration also supports managed frame artifacts; the panel currently selects
layer attachments, suitable for mosaics and contact observations. Long service calls
run outside the Qt thread and are tracked by the Hub close guard.

Verification uses explicitly synthetic image bytes and known mathematical
transforms. Unit tests cover scale/origin/Y orientation, rotation/shear/translation,
rank and conditioning, residual outliers and uncertainty. Real local/HTTP workflows
cover provenance, review, stale revisions, RBAC, persistence and reopen. Dedicated
Russian/English Qt processes exercise the shown panel. PostgreSQL tests run in
fixture-owned UUID schemas and verify concurrent CAS, reopen, outbox and rollback.

This is a registration and evidence foundation. It does not automatically discover
correspondences in microscopy, prove physical contact from overlapping drawings,
extract transistor circuits, or demonstrate recovery of an actual chip.
