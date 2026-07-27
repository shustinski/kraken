# Kraken

Kraken is a clean-architecture project manager for sparse frame grids. It can
run as an offline, single-writer Desktop application or as a PostgreSQL-backed
shared service. Contour, NeuralImage and the remaining applications stay
isolated plugins and exchange only versioned manifests through Kraken Agent.

## Layout

```text
src/
  kraken_manager/        # domain, application, infrastructure, Qt presentation
  kraken_hub/
  kraken_server/
  kraken_agent/
  kraken_core/           # technical runtime and plugin wire protocol

plugins/
  neuralimage/
    pyproject.toml
    README.md
    src/...
    tests/...
    resources/...
    packaging/...
    scripts/
  contour/
    pyproject.toml
    README.md
    src/...
    tests/...
    resources/...
    packaging/...
    scripts/
  krona/
    pyproject.toml
    README.md
    src/...
    tests/...
    resources/...
    packaging/...
    scripts/
  csliser/
    pyproject.toml
    README.md
    src/...
    tests/...
    resources/...
    packaging/...
    scripts/
  karakal/
    pyproject.toml
    README.md
    src/...
    tests/...
    resources/...
    packaging/...
    scripts/
```

The active plugin package names are `neuralimage`, `contour`, `krona`,
`csliser`, and `karakal`.

## UV Setup

Install only the profiles needed on a workstation:

```powershell
uv sync --extra desktop
```

For a server with PostgreSQL and reports:

```powershell
uv sync --extra server --extra postgres --extra reports --extra packages
```

Install root development tools (this also includes the Desktop test runtime):

```powershell
uv sync --extra dev
```

If `uv` warns that hardlinks are unavailable, use copy mode:

```powershell
uv sync --link-mode=copy
```

## Plugin UV Setup

Each plugin is also an independent Python project. Initialize a plugin
environment from its folder:

```powershell
cd plugins\krona
uv sync
uv run python -m krona --help
```

With development dependencies:

```powershell
cd plugins\contour
uv sync --extra dev
uv run pytest
```

The plugins that use shared Kraken code declare `kraken` as a workspace
dependency, so `uv sync` inside the plugin also installs the root
`kraken_core` package.

## Run

Create the first workstation account (there is no self-registration), then
start Desktop:

```powershell
uv run kraken-admin bootstrap-local --username admin --display-name "Administrator"
uv run kraken-hub
```

The previous plugin-only launcher remains available behind
`kraken-hub --legacy-launcher`; `--list` lists registered plugins.

Production server startup is fail-closed. Run Alembic first and select the
built-in composition as documented in
[`docs/deployment/server.md`](docs/deployment/server.md). Ephemeral state is
available only with an explicit `kraken-server --development`.

Kraken Agent uses a durable local SQLite queue and authenticated loopback
channel. Start it with `kraken-agent`; its registry determines which V1 plugin
operations may run.

The Desktop project catalog supports audited rename/archive/restore, managed
directory imports with `<x>_<y>`, row-major or regex mapping, sparse matrix
statuses, representation switching, activity reports, integrity scans and
canonical backup/restore bundles. Managed files are copied into the immutable
SHA-256 BlobStore; a source folder is never treated as the authoritative
project state.

The project workspace uses the shared virtualized frame matrix. Its thumbnail
cache is storage-neutral and supports `sqlite://`, `files://`, and `memory://`
adapters; see [`docs/frame-matrix.md`](docs/frame-matrix.md).

Shared mutations under `/api/v1` require `Idempotency-Key`, optimistic
`If-Match`, a GitLab principal and a live GitLab `userinfo` check. Project,
layer, representation and ACL lifecycle changes all emit versioned audit
events; local/server accounts remain read-only for shared content.

Kraken Hub checks for its own updates at startup when an update manifest is
configured. Set it in the Hub with **Update source…**, pass
`--update-url https://example.org/kraken/version.json`, or set the
`KRAKEN_UPDATE_URL` environment variable. A choice of **Later** suppresses the
same offer only for the current process; it is offered again after the next
launch.

**Plugins** — always from that plugin’s directory (separate `uv sync` per plugin the first time):

```powershell
cd plugins\contour
uv sync
uv run python -m contour
```

Same pattern for `krona`, `neuralimage`, `csliser`, `karakal` (replace the folder and module name). Do not set `PYTHONPATH` by hand for normal use; `uv` installs dependencies.

### Debugging in VS Code / Cursor

1. In the repo root: `uv sync` (creates `.venv` with dependencies).
2. **Python: Select Interpreter** → `.\.venv\Scripts\python.exe` (or `.venv/bin/python` on Linux/macOS), **not** `Program Files\Python...`.
3. **Run and Debug** → **Kraken Hub** → F5. The launch config forces this `.venv` so it still works if the wrong interpreter is selected.

If you see `No module named kraken_hub`, the global/system Python is being used. Use `uv run python -m kraken_hub` from the repo root, or select the project `.venv` as above.

## Build

Each plugin owns its build scripts:

```powershell
.\plugins\contour\scripts\build_windows.ps1
```

```bash
./plugins/contour/scripts/build_linux.sh
```
