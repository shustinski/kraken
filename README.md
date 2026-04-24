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

Run commands through the managed environment:

```powershell
uv run python -m kraken_hub --list
uv run python -m kraken_hub
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

From the repository root:

```powershell
uv run python -m kraken_hub --list
uv run python -m kraken_hub
```

Run a plugin from its own folder:

```powershell
cd plugins\krona
uv run python -m krona
```

Run a plugin from the repository root by adding the plugin source path:

```powershell
$env:PYTHONPATH = "src;plugins\krona\src"
uv run python -m krona
```

Common plugin commands:

```powershell
cd plugins\neuralimage
uv run python -m neuralimage --help

cd plugins\contour
uv run python -m contour --help

cd plugins\krona
uv run python -m krona --help
```

## Build

Each plugin owns its build scripts:

```powershell
.\plugins\contour\scripts\build_windows.ps1
```

```bash
./plugins/contour/scripts/build_linux.sh
```
