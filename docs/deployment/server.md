# Kraken Server deployment

Kraken Desktop clients connect to Kraken Server over HTTPS. They never receive
the PostgreSQL password and never access the managed blob directory directly.

## Packaged Windows installation

The normal operator workflow does not require Python or `uv`:

1. Install PostgreSQL and create an empty database owned by a dedicated Kraken
   database user.
2. Run `KrakenServerSetup-<version>.exe` as a Windows administrator.
3. At the end of installation, start **Configure Kraken Server** and supply:
   the PostgreSQL URL, blob directory, first administrator login and password.
4. The setup command runs all schema migrations, stores the database URL with
   machine-scoped Windows DPAPI, writes
   `C:\ProgramData\Kraken\Server\server.toml`, installs the `KrakenServer`
   Windows service and starts it.
5. Publish the service through an approved HTTPS endpoint. Either configure a
   reverse proxy with WebSocket support or provide `tls_cert_file` and
   `tls_key_file` in `server.toml` and bind Kraken Server to the approved
   network interface.

Only HTTPS/WSS (normally TCP 443) is exposed to workstations. PostgreSQL and
the blob directory are accessible only to the Kraken Server service account.

An unattended equivalent is:

```text
KrakenAdmin.exe setup-server --database-url <postgres-url> --blob-root D:\KrakenData\blobs --username admin --display-name Administrator --install-service --server-executable "C:\Program Files\Kraken Server\KrakenServer.exe"
```

Passwords are prompted interactively. Omitting `--database-url` also prompts
for it without placing the secret in the command history.

If every administrator loses access, a local Windows administrator may run:

```text
KrakenAdmin.exe recover-admin --username admin
```

Recovery only restores an existing account. It never creates an unknown
remote administrator.

## Desktop connection

Install `KrakenDesktopSetup-<version>.exe` on each workstation. On first
connection enter the HTTPS address, Kraken login and password. The address is
remembered; the password is not. The **Administration** page is visible only
when the authenticated server principal has the `server_admin` system role.

The page creates/disables accounts, resets passwords, revokes sessions and
grants/revokes `server_admin`. Every operation is authorized by Kraken Server
and appended to the administration audit log. An administrator cannot disable
or demote themselves, so the last active administrator cannot be removed from
the UI.

## Source/development operation

Source deployments remain supported. Run Alembic before the first start and
select the built-in persistent composition:

```text
KRAKEN_SERVER_COMPOSITION=kraken_server.composition:postgresql_composition
KRAKEN_DATABASE_URL=postgresql+psycopg://...
KRAKEN_BLOB_ROOT=/srv/kraken/blobs
KRAKEN_PROJECT_ACCESS_MODE=acl
alembic upgrade head
kraken-server
```

`--development` is the only way to enable the ephemeral backend. Production
startup fails closed when persistent storage or authentication is missing.

## Backups

A complete backup includes PostgreSQL, managed blobs, the `.server` signing
key directory and `server.toml` plus its DPAPI-protected secret. Machine-scoped
DPAPI data must be decrypted/re-protected during a migration to another host.
A backup is operational only after an isolated restore drill succeeds.
