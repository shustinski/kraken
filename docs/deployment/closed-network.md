# Closed-network deployment checklist

1. Mirror the selected extras (`desktop`, `server`, `postgres`, `reports`,
   `packages`) and all platform wheels inside the closed network.
2. Configure the internal GitLab issuer, OIDC client, CA bundle and exactly the
   scopes `openid profile email`. Keep client/server credentials in OS keyring
   or the deployment secret store.
3. Set `KRAKEN_DATABASE_URL`, apply `alembic upgrade head`, provision the blob
   root with restrictive OS ACL, then run `kraken-admin bootstrap-admin` once.
4. Terminate TLS at the approved internal endpoint. The Agent control API must
   remain on loopback and use a newly generated control token.
5. Create a backup, restore it into an isolated target and run full event/blob
   integrity verification before enabling shared mutations.
6. Load-test the actual hardware baseline and record p95 viewport latency,
   worker throughput, 50-client concurrency and a 10-million-event report.

Backups are incomplete unless they include PostgreSQL, managed blobs, signing
keys and deployment configuration. A backup is not considered operational until
the restore drill succeeds.

