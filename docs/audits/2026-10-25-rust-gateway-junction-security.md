# Rust Blob Gateway native junction audit

Date: 2026-10-25. Tested the repository-built Windows MSVC release executable
through real loopback HTTP requests carrying Python-issued HMAC tickets and
actual `_winapi.CreateJunction` links. All storage and link targets were inside
the tests' own temporary directories. Cleanup removed only junction entries;
child termination affected only processes started by these tests.

## Failure proof

The first run of `tests/test_blob_gateway_junctions.py` against the pre-fix
release binary produced **9 failures**:

* Download followed a digest-prefix junction and returned HTTP 200 with external
  content. Existing-object upload/deduplication followed the same junction.
* Upload committed a new immutable object through a prefix junction outside the
  configured object root.
* Runtime `.gateway-staging`, `.transfers`, `.deleting`, and nested project
  transfer-directory junctions were accepted.
* A junction at the configured root or staging root was accepted at startup.
  This exposed external files to staging cleanup, including old upload files.

The final suite also checks both digest-prefix levels, root substitution after
startup, and a junction above an initially nonexistent root. Startup tests
explicitly preserve an external old-file sentinel.

## Fix

`blob_gateway/src/main.rs` now checks each existing directory component before
using or creating it, without canonicalizing away a reparse root. Windows
reparse attributes and symbolic links are rejected, as are special files.
Startup validates roots before staging cleanup. The same checks cover project
deletion markers, transfer directories and leases, staging creation and cleanup,
blob commit/deduplication, and reads.

Windows file opens use `FILE_FLAG_OPEN_REPARSE_POINT`; file metadata and ancestor
checks occur around opening. Reads and upload deduplication consume the checked
handle itself. Download size comes from that same handle. Cleanup refuses
unsafe ancestors or leaf entries. No dependency or Cargo lock changes were
needed.

## Verification

MSVC environment: Visual Studio 18 BuildTools `vcvars64.bat`.

```powershell
$env:RUSTUP_HOME='D:\code\kraken\.codex_tmp_scratch\toolchains\rustup'
$env:CARGO_HOME='D:\code\kraken\.codex_tmp_scratch\toolchains\cargo'
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && D:\code\kraken\.codex_tmp_scratch\toolchains\cargo\bin\cargo.exe build --manifest-path blob_gateway/Cargo.toml --release --locked'
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && D:\code\kraken\.codex_tmp_scratch\toolchains\cargo\bin\cargo.exe test --manifest-path blob_gateway/Cargo.toml --locked'
```

Release build succeeded. **3 Rust unit tests passed**, including deletion-barrier
lease lifetime and same-size corrupt-object deduplication rejection.

```powershell
.venv/Scripts/python.exe -c "import ctypes; ctypes.windll.kernel32.SetErrorMode(3); import pytest; raise SystemExit(pytest.main(['tests/test_blob_gateway_junctions.py','tests/test_blob_gateway.py','-q','--tb=short']))"
```

**21 passed, 1 upstream Starlette/httpx warning, 27.34 s.** Includes 13 new real
junction scenarios and the 8 existing ticket/configuration/upload/download/
Range/deduplication/parallel-stream checks. Ruff passes for the new Python test.

Tested release binary SHA-256:
`84b0bf932bc5264fbe9f8fe3dcf3534f6dd91f5e1e871a09670c38949d4a4ef4`.

Rustfmt was unavailable in the existing toolchain (`cargo-fmt.exe` component
not installed), so no successful formatter check is claimed.

## Limits

This is protection for the gateway's own filesystem operations, not an OS
sandbox against arbitrary native code running as the same user. Ancestor
validation is portable path-based validation; it does not make directory
creation, hard-link commit or deletion atomic against continuous ancestor
replacement. Those stronger guarantees require native directory-handle-relative
operations and/or filesystem isolation. The tests prove rejection of the real
static and between-request junction substitutions described above, without
claiming general race freedom. Windows final-file symbolic-link coverage still
depends on privileges; directory junction tests run without those privileges.
Frozen installer artifacts built before this fix need a new build and are not
covered by this source/release-binary validation.
