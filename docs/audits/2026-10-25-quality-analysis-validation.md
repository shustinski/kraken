# Quality / AnalysisRun validation — 2026-10-25

Implemented and exercised local, server, remote-client and Hub quality evaluation
against actual imported Karakal measurements. Explicit rule fields are required;
the threshold control has no default. Analysis-linked findings reject manually
supplied verdicts. Replacement runs must analyze the recorded corrections while
preserving unaffected references and computation identity.

The regression workflow executes the real headless worker on known 8×8 masks:
the original measured IoU is 0 and the corrected measured IoU is 1. The worker
writes real compressed result bundles, which the actual filesystem/PostgreSQL
AnalysisRun stores import before quality evaluation. This validates workflow and
pixel-metric decisions; it does not claim electrical sign-off or large-image
benchmark performance.

## Executed checks

The full focused run used the owned PostgreSQL instance on loopback port 55439,
with `KRAKEN_TEST_POSTGRES_URL` configured. PostgreSQL fixtures migrated unique
UUID schemas, used temporary files, and dropped their own schemas in teardown.

```powershell
.venv/Scripts/python.exe -c "import ctypes; ctypes.windll.kernel32.SetErrorMode(3); import pytest; raise SystemExit(pytest.main(['tests/test_quality_workflow.py','tests/test_quality_client.py','tests/test_quality_dialog.py','tests/test_quality_postgres.py','tests/test_quality_analysis.py','tests/test_quality_analysis_postgres.py','tests/test_quality_accepted_lineage.py','tests/test_artifact_lineage.py','tests/test_analysis_run_workflow.py','-q','--tb=short']))"
```

Result: **40 passed**, one upstream Starlette/httpx deprecation warning, 60.40 s.
The run includes the actual local Agent/worker/gateway import smoke, forced
PostgreSQL compare-and-swap races, rollback of findings/outbox/evaluation receipt,
and safe offscreen Qt child processes. Windows fault dialogs were disabled before
native imports.

Ruff checks passed for all added/changed quality, analysis-store and lineage
modules and regression tests. The pre-existing broad facade imports in
`infrastructure/projections/rebuild.py` were not mass-refactored.

## Fixed defects and protected invariants

- A metadata rebuild previously replaced the SQLite file containing analysis
  results. Analysis now has a separate disposable cache, replayed from the same
  canonical events and bundles. Tests repeatedly rebuild metadata and reconstruct
  the complete legacy analysis cache without losing measurements.
- Legacy run/partition results without matching canonical history produce an
  explicit migration error, preserving the old cache. Failed replay does not
  leave an apparently successful empty new cache. Windows SQLite inspection
  connections close before atomic file replacement.
- Completed counters alone cannot certify a run with missing actual frame records.
- A claimed correction cannot be rechecked by analyzing the original version,
  changing the reference, using a different runtime/recipe/parameters, supplying
  a nonexistent metric, or submitting `passed: true` directly.
- Acceptance certifies a correction branch. Exact old defective inputs and
  unrelated/mixed descendants remain gated; actual descendants of the approved
  correction clear the gate. Missing/cyclic/foreign provenance remains fail-closed.

## Scope boundary

No production data or root import/performance profile was modified. These checks
cover deterministic quality decisions over existing AnalysisRun results and the
shown Hub workflow. Launching new remote AnalysisRuns remains the responsibility
of the existing analysis execution subsystem; this slice does not create a second
analysis engine, scheduler or authoritative database.
