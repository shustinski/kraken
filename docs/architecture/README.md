# Kraken Project Manager architecture

Kraken is split into independently deployable composition roots:

- `kraken_hub` composes the PyQt desktop and embedded local services;
- `kraken_server` composes HTTP/WebSocket transport, PostgreSQL adapters and workers;
- `KrakenBlobGateway` is the Rust data plane for large immutable upload/download streams;
- `kraken_agent` runs durable plugin jobs against isolated staging workspaces;
- plugins consume versioned manifests and cannot open project storage or databases.

The dependency rule is enforced by tests:

```text
kraken_manager.domain
          ↑
kraken_manager.application
       ↗      ↖
infrastructure  presentation.qt
       ↖      ↗
 composition roots
```

`domain` contains no filesystem paths, framework objects, SQL/HTTP concepts or
plugin process APIs. `application` exposes semantic ports rather than a generic
storage abstraction. Backend selection and credentials belong only to a
composition root.

## Data authority

For a local project, immutable JSONL event segments and content-addressed blobs
are authoritative. `.index/index.sqlite3` is a disposable read model. A project
lock makes the backend single-writer.

For a shared project, PostgreSQL is authoritative for events, projections,
identity, ACL and jobs. Blob bytes are supplied by a separate `BlobStore`.
The transactional outbox decouples projection and notification work without an
external broker at the supported v1 scale.

The Python server authorizes a transfer and issues a short-lived HMAC ticket.
Desktop or Agent then talks directly to the Rust gateway. A completed upload is
registered in PostgreSQL only after its size and SHA-256 object are present in
the shared BlobStore. Legacy clients retain the Python proxy endpoints.

Coordinates and dimensions are immutable. Empty grid cells do not become rows;
only artifacts, work state and other facts about a frame are projected.

## Security boundary

Every read requires a session. Shared mutations require all of:

1. a GitLab OIDC principal identified by `issuer + sub`;
2. a successful live `userinfo` request immediately before mutation/commit;
3. a Kraken project role granting the requested permission.

Local principals are hard-denied for shared mutation even if an invalid ACL row
exists. Plugins receive copied input files and safe relative names in a staging
workspace, never project paths, database credentials or blob-store handles.

## Versioning

Events, plugin jobs, plugin results, review packages, frame selections and
migration bundles have independent public schema versions. Database migrations
never rewrite event payloads; pure upcasters translate older payloads on read.
Managed artifact versions and blobs are immutable.

