# ADR 0002: Semantic storage ports and canonical migration

Status: accepted — 2026-07-17

## Decision

Kraken does not expose one CRUD `Storage`. The application defines separate
`EventStore`, `ProjectionStore`, `BlobStore`, `UnitOfWork`, identity/ACL, locks,
background jobs, packages and snapshot ports.

File projects use atomic immutable event segments, SHA-256 objects, a project
lock and a rebuildable SQLite index. Shared projects use PostgreSQL and a
separate blob adapter. `StorageProfile.capabilities` advertises behavior and
limits before a use case starts.

Every migration passes through `KrakenMigrationBundleV1`. It preserves event
identities, revisions, timestamps and hashes and supports checkpoints. This
avoids pairwise database converters and makes source/destination verification
the same operation for every backend.

## Consequences

SQLite inside a file project is never an authority. A successful cutover cannot
delete its source. External references are reported but not copied because the
referenced bytes are outside Kraken's authority.

