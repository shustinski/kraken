# Kraken Server

Run Alembic migrations before the first start. The built-in persistent
composition is selected with:

```text
KRAKEN_SERVER_COMPOSITION=kraken_server.composition:postgresql_composition
```

It requires `KRAKEN_DATABASE_URL` and `KRAKEN_BLOB_ROOT`.
`KRAKEN_GITLAB_ISSUER` enables GitLab identities
and shared mutations; `KRAKEN_GITLAB_CA_FILE` may point to the closed-network
CA bundle. Without a GitLab issuer, authenticated server-local accounts keep
read-only access to the shared catalog.

The metadata transaction covers domain events, temporal/current projections,
ACL and outbox. Managed blobs are immutable and content-addressed; an aborted
transaction can only leave an unreferenced blob, removable by an integrity
maintenance job.

`--development` is the only way to enable the ephemeral in-memory backend.
Production startup fails closed when the composition or authentication
resolver is missing.

Create the first Server Admin without a hard-coded password:

```text
kraken-admin bootstrap-admin --database-url <postgres-url> --username admin --display-name Administrator
```
