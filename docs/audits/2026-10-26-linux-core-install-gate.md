# Linux headless core installed-wheel gate

Date: 2026-10-26. This gate checks a bounded functional subset of Kraken on
Linux, without desktop or numerical dependencies. It does not replace the full
distribution gate. No model training or vectorization-quality assessment was
performed.

## Result and exact scope

A fresh container installed the rebuilt wheel and its complete requested
dependency closure **with Docker networking disabled** (`--network=none`) and
pip indexes disabled (`--no-index`). `pip check` passed. Both installed
`/usr/local/bin/kraken-agent --help` and `kraken-server --help` exited zero.

The installed-wheel regression run passed **129 tests and 12 subtests**, with
one existing Starlette/httpx deprecation warning and no skips, in **13.49 s**.
Total installation/check/test time was **19.237 s**.

The 19 test modules cover POSIX project locking and crash recovery, event-cache
coherence, filesystem event/blob storage contracts, real symlink path/package
attacks, domain/application behavior, Agent execution/callback/lease recovery,
subprocess execution-mode dispatch, queue configuration, artifact lineage,
plugin/analysis protocol formats, and server service/HTTP route contracts.
Subprocess mode tests run a small deterministic diagnostic probe, not training.

Requested wheel extras are `agent,server,packages`. Desktop/Qt, image/numerical
algorithms, PostgreSQL integration, graphical installers and Linux Rust gateway
binaries are outside this result. The full dependency/test profile remains the
tool's default; `--profile core` explicitly selects this smaller gate.

Image: `python@sha256:35caba83af2fe6da9a5d922f9b0156f21d1246a48278c58a08bd253303ba42e8`.
Actual interpreter: CPython **3.14.9**, Linux.
Tested wheel SHA-256:
`c9f6a1c0814eb7c86bc8eee4f3f6469e01e2f56f1d309acda9cbd39658bde403`.
The imported manager module was
`/usr/local/lib/python3.14/site-packages/kraken_manager/__init__.py`, not the
source-tree package. Tests use a hashed source/test snapshot; Krona's auxiliary
source snapshot is explicitly recorded in the report.

## Real defect found and fixed

The initial online core installation and 129 tests passed, but the first
offline CLI check correctly failed. Importing
`kraken_core.process_lifetime` constructed `ctypes.WINFUNCTYPE` unconditionally;
that attribute does not exist on Linux, so the installed `kraken-agent --help`
crashed before parsing arguments. The failed report/log is retained at
`.codex_tmp_scratch/production-audit-20261026/linux-core-offline/`.

`process_lifetime.py` now constructs the Windows callback type only on Windows,
matching its already platform-guarded public lifetime-binding API. The rebuilt
wheel passed the actual Linux CLI checks. A separate regression imports the
module in a POSIX environment with a Windows callback factory that raises if
used; no Windows API is touched. Existing real Windows parent/child console
lifetime tests also passed: **2 passed, 0.62 s**.

## Tooling changes and cache reuse

`tools/verify_linux_distribution.py` now supports explicit core/full profiles,
bounded installation timeouts, repeated read-only `--pip-cache` inputs, an
optional wheelhouse, and an offline mode. Complete HTTP-cache wheel bodies are
validated as ZIPs, named from their embedded wheel metadata, checked for
conflicting contents, and copied into the new report's input wheelhouse with
SHA-256/provenance recorded. Partial downloads are not treated as wheels.

The prior retry's cache supplied completed wheels, including cryptography;
missing small headless/server dependencies were obtained in a separate bounded
online preparation run. The final gate used only saved wheels and no network.
The earlier full-install network timeouts and the initial missing-server-extra
collection failure remain failed evidence; none is counted as a pass. No GUI,
NumPy, SciPy or multi-gigabyte training packages were installed for this gate.

All containers were newly created with unique `kraken-linux-audit-*` names,
`--rm`, and mounts limited to their own input/evidence directories. No user
container or user project was changed. The single locally built wheel received
read permission for the host account running Docker; no source-data permissions
were changed.

## Reproduction and evidence

Wheel build used the existing setuptools backend:

```powershell
.venv/Scripts/python.exe -m build --wheel --no-isolation --skip-dependency-check --outdir .codex_tmp_scratch/production-audit-20261026/linux-core-fixed-wheel
```

The build frontend's dependency check was skipped because its build-only
`wheel` distribution was absent; the installed setuptools backend successfully
built the wheel using its bundled support. The output ZIP CRC and platform
guard content were checked. Runtime dependency checks were not skipped.

Final gate command:

```powershell
.venv/Scripts/python.exe tools/verify_linux_distribution.py --profile core --offline --installation-timeout 120 --pip-cache .codex_tmp_scratch/production-audit-20261025/linux-wheel-retry/pip-cache --pip-cache .codex_tmp_scratch/production-audit-20261026/linux-core-server-online/pip-cache --wheel .codex_tmp_scratch/production-audit-20261026/linux-core-fixed-wheel/kraken-0.1.0-py3-none-any.whl --report .codex_tmp_scratch/production-audit-20261026/linux-core-fixed-offline
```

Evidence directory:
`.codex_tmp_scratch/production-audit-20261026/linux-core-fixed-offline/`.
It contains `report.json`, `tests.xml`, `tests.log`, `installation.log`,
`pip-installation.json`, `dependency-check.log`, both CLI help logs,
`installed.txt`, `cached-wheels.json`, `source-manifest.json`, `container.json`
and the exact input wheel/test/tool snapshot.

Tool regression command
`.venv/Scripts/python.exe tools/test_verify_linux_distribution.py` passed
**4 tests**. Ruff passed for both tooling files and `process_lifetime.py`.
