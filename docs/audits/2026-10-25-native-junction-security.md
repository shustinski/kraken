# Native Windows junction security regression audit

Date: 2026-10-25. Scope: Agent transfer staging, filesystem path helpers,
managed blob enumeration, review package reading, and regression checks for
callbacks, authorization, quality/AnalysisRun and project locking.

## Reproduced defects and fixes

The new `tests/test_security_native_junctions.py` creates actual Windows NTFS
directory junctions with `_winapi.CreateJunction`; it does not require symbolic
link privileges. Cleanup removes only the temporary junction entry with
`os.rmdir`, never recursively removes its target. POSIX runs use directory
symlinks for the equivalent paths.

* Failed Agent input exclusive creation used to delete a pre-existing file.
  Through an input junction, this deleted an external sentinel. Downloads also
  created new files outside staging. Transfer destinations now reject reparse
  components; cleanup only considers a file actually created by this operation
  with the same device/inode identity. Input hash mismatch removes its own
  partial file. External input copies use the existing staging API.
* Agent output upload followed an output directory junction. Upload now obtains
  a rooted regular-file handle and uses that same handle and `fstat` throughout
  ticket creation, transfer and completion.
* `os.walk(..., followlinks=False)` still descended Windows junctions in blob
  enumeration. Its directory and file filters now reject every reparse point.
* Containment helpers accepted `..` and multi-component/absolute parts. A rooted
  reader could read a neighboring file through `root/../private`. Both helpers
  validate individual relative components, including separators and drive/ADS
  syntax, before filesystem operations. The root itself is checked before
  canonicalization.
* Append opening did not recheck intermediate components after obtaining the
  handle. A deterministic junction substitution test proved that a plugin log
  could escape staging. Append now rechecks containment before exposing the
  handle and closes it when validation fails.

The initial native-junction suite reproduced four failures and three passes;
subsequent tests separately reproduced the upload, traversal and append defects
before their fixes. Actual staging output and review package junction rejection
already worked and remains covered.

## Verification

Final new regression suite:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_security_native_junctions.py -q --tb=short
```

**20 passed** (0.29 s). Includes real directory junction attacks, traversal,
deterministic append substitution, successful byte transfer and failed-download
cleanup.

Broader scoped regression run, with Windows native crash dialogs disabled:

```powershell
.venv/Scripts/python.exe -c "import ctypes; ctypes.windll.kernel32.SetErrorMode(3); import pytest; raise SystemExit(pytest.main(['tests/test_security_native_junctions.py','tests/test_security_paths_packages.py','tests/test_agent_input_media.py','tests/test_agent_execution_safety.py','tests/test_agent_runtime.py','tests/test_agent_jobs.py','tests/test_manager_filesystem_blob.py','tests/test_filesystem_lock_recovery.py','tests/test_work_package_archives.py','tests/test_review_return.py','tests/test_review_workflow_usecases.py','tests/test_work_order_security.py','tests/test_quality_analysis.py','tests/test_quality_accepted_lineage.py','tests/test_analysis_run_workflow.py','-q','--tb=short']))"
```

**129 passed, 3 skipped, 9 subtests passed** (74.85 s). This run collected the
first 18 new tests; the final two successful-transfer/hash-cleanup tests were
added afterward and are included in the independent 20-pass run above. The
three skips are existing file/directory symbolic-link tests requiring Windows
privilege 1314; the actual directory junction tests do run.

Ruff passed for the three changed production files and the new test file.
No locking/EventStore, corpus/import, packaging or primary audit files were
changed in this audit. Existing lock crash/reopen/multiprocess/legacy tests ran
successfully in the scoped regression suite.

## Limits

These path checks harden Kraken's own file operations. They do not make a
same-user arbitrary native plugin an OS sandbox. Fully atomic resistance to
continuous adversarial ancestor replacement requires native handle-relative
operations or filesystem permission/isolation boundaries; checks around an
open cannot provide that general guarantee. The deterministic append race
test verifies no bytes are written after the detected replacement, although
opening with `O_CREAT` can leave an empty external file before the post-check
rejects it. File symbolic-link tests remain separately skipped on this Windows
account. Rust blob gateway and packaged rebuild checks belong to the parent
agent's independent audit and are not claimed by these results.
