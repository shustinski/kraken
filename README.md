# Kraken

Kraken is a PyQt6 hub plus a shared runtime for independently installed
plugins. The repository is a monorepo: the hub and common code live under
`src/`, while every application lives under `plugins/<plugin_name>/`.

## Layout

```text
src/
  kraken_hub/
  kraken_core/

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

Install project dependencies from the repository root:

```powershell
uv sync
```

Install root development tools, such as test and lint dependencies:

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

**Hub (main app)** — from the repository root after `uv sync`:

```powershell
uv run python -m kraken_hub
```

`--list` lists registered plugins.

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
