# Electrical evidence and partial netlists

Hub exposes the electrical model alongside the physical multilayer model. The
same commands are available through local composition, RemoteCatalog/DualCatalog
and `/api/v1/projects/{project}/electrical` (GET), `/commands` (POST), `/export` (GET).
Writes require `If-Match`, `Idempotency-Key`, the expected physical-model revision,
and project authorization. Events use `electrical:<project>` in the existing
EventStore; masks/evidence use existing immutable ArtifactVersions and BlobStore.

Conductor extraction requires an active version-specific calibrated binary mask,
an explicit foreground value (0 or 255) and explicit 4/8 adjacency. Pillow decodes
1/L/LA images; SciPy labels exact connected pixel components. Grayscale values
are not silently thresholded. An explicit nodata marker and nonopaque LA pixels
remain unknown; components record contact with nodata, image boundaries and
diagonal-only bridges. Each component retains exact pixel runs/digest, mask SHA,
algorithm version, calibration and native-grid dimensions. Input limits are
64 MiB encoded bytes and 16,777,216 pixels; larger inputs require an explicit
tiling workflow rather than an approximate fallback.

Pixel adjacency does not certify a conducting material. Separate material reviews
require distinct active evidence bytes and an audit note. Physical contacts require
the physical model's contact-layer evidence/review and an explicit correspondence
to named conductor components. The contact footprint checks the selected supports
in SI coordinates; it never searches for or proves a connection by nearest distance.
Terminals require explicit component, port, direction, optional instance/cell and
independent observation bytes. Changed calibration/artifact/registration anchors,
retracted evidence, inconsistent net names and contradictory contacts/pins remain
stale/unknown/conflicted and block inappropriate export.

The intermediate graph includes reviewed nets, nodes, provenance, unknowns and
conflicts. EDIF export identifies itself as `partial_reviewed_subgraph`; it does
not claim to reconstruct an entire chip. Synthetic tests explicitly bind A/Y to a
reviewed conductor/contact fixture and roundtrip the exported EDIF through strict
Krona (Y=A). They do not prove real semiconductor structure. The current MOS detector proposes candidates only for its supported rectangular,
single-channel gate geometry against distinct diffusion components. It records
registration/calibration revisions, overlap/uncertainty and unresolved bulk; it
never assigns polarity or source/drain from geometric proximity. An authorized
reviewer must supply explicit roles and independent process/implant/engineer
annotation evidence before realization. Recalibration or replacement of evidence
invalidates the realization. Local replay, HTTP/PostgreSQL CAS and shown Qt review
are covered by `test_device_candidate*` (18 workflow/value/API/Qt tests).
Acceptance against independently verified real semiconductor geometry and broader
device families remains open.

`test_electrical_graph.py`, `test_electrical_workflow.py`,
`test_electrical_postgres.py` and `test_electrical_dialog.py` cover masks, missing
evidence, conflicts/staleness, real managed BlobStore, EDIF/Krona, reopen, migrated
PostgreSQL, HTTP, concurrent CAS/rollback and shown RU/EN Qt windows. A frozen
Desktop self-test executes exact 4/8/nodata extraction. Frozen Server's initial
NumPy import from an AnyIO worker hung in native extension creation (captured
thread dumps). Bundled entry points now initialize NumPy/SciPy on the main thread
before workers. The same frozen HTTP extraction then passed in 0.1293s, and the
12-check distribution scenario passed with restart persistence. Source tests alone
were insufficient to find or certify this fix.

Install `electrical` and `registration` extras for source deployments. Windows
builds verify NumPy/SciPy/Pillow before packaging and include the actual runtime.
