# Beta Stabilization W3 Implementation Plan — «палитра живых сценариев принятия»

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-earn every `beta: true` registry verdict with post-freeze evidence by implementing and running the palette of live acceptance scenarios on the beta stand (hybrid rule), re-binding `docs/quality/registry/actions.json` rows (status/tests/evidence) to fresh JUnit/receipts, and delivering the W3 section of the checkpoint — the W3 wave of the beta-stabilization spec.

**Architecture:** The palette is a set of pytest scenario files, not one linear flow. Local-only actions (account bootstrap, preferences, roots, local backup) are re-earned by offscreen GUI-tier tests; everything that touches the server flow is re-earned by live-tier tests against the permanent stand (Server + PostgreSQL schema `kraken_beta_test` + agent; Blob Gateway when supplied) reusing the existing env contracts: `KRAKEN_LIVE_WORKER_CONFIG` (`tests/test_live_server_worker_acceptance.py`: `server_url`, `session_secret`, `machine_secret`, `protected_project_id`, `results_directory`, receipts chained via `results_directory/receipt.json`) and `KRAKEN_UI_TEST_CONFIG` (`tests/test_live_ui_workflow.py`: `server_url`, `username`, `password`, `source_directory`). New palette files are added to the tier file-lists in `scripts/quality-gates.ps1`, so `tools/quality/coverage_report.py::action_verdict` derives each action's verdict from the fresh JUnit and the registry `tests` bindings re-point at palette node ids. Gateway-dependent scenarios form a separate task group behind an explicit prerequisite (exe supply); non-gateway scenarios run and complete independently of it.

**Tech Stack:** Python 3.14 (`.venv\Scripts\python.exe`), pytest / pytest-qt (offscreen, `QT_QPA_PLATFORM=offscreen`), PowerShell 5.1, PostgreSQL (local, schema `kraken_beta_test`), FastAPI server, `kraken-agent`, Rust Blob Gateway (external supply), `tools/quality/{coverage_report,gate_compare,generate_quality_docs}.py`.

**Spec:** `docs/superpowers/specs/2026-11-17-beta-stabilization-design.md` (binding: «Палитра действий», «Правило свежести доказательств», exclusions list, `domain_auth=false`).
**Gap source (W3 backlog):** `docs/quality/checkpoint-2026-11-17-beta.md` — 15 PASS / 45 NOT_TESTED / 1 BLOCKED (HUB-CHECK-UPDATES → W4). Controller ruling stands: the checkpoint verdict table is the sole readiness source; W3 re-binds registry statuses to post-freeze evidence only for scenarios that passed.

**Rebaseline (2026-11-19, `a83b6fb`):** Tasks 1–4 are committed (`9d8ac64`, `23276a8`, `a258fab`, `233ecad`); Task 5's documentation is committed (`e349723`), and the S01 test exists as `tests/quality/test_s01_local_identity.py` (`72f458b`). These tasks are historical, not instructions to recreate files or repeat commits. Before continuing, verify their registry bindings and evidence against the current checkout. The gateway exe is present and the stand runbook records a successful gateway-mode acceptance on 2026-11-19; Task 18 is a configuration audit, not a request for an external kit. The 15/45/1 count is the W2 baseline, not a current verdict; recompute it from the current registry and JUnit before reporting progress. The worktree contains unrelated edits: preserve them.

## Global Constraints

- Python floor: `requires-python = ">=3.14"` (pyproject.toml) — all tooling runs on the repo venv `.venv\Scripts\python.exe`.
- Dev machine is a **closed network**: pypi.org and download.pytorch.org unreachable. Nothing in this plan may require network access. Do not attempt `uv lock`/`uv sync` that mutates `uv.lock` or `.venv` (D21 stays external). No new pip/uv/cargo dependencies.
- Never touch `uv.lock`, `.venv`, the working database `kraken_local` (schema `public`), or `config/local/*` (read-only templates). `D:\Heggs\Source` is read-only. Stand data lives only in PostgreSQL schema/database `kraken_beta_test` and `.test_runtime\beta-stand*` (git-ignored).
- PowerShell 5.1 only: native commands run through the `Invoke-Native` pattern; no `python -c` with embedded double quotes; where `-File` mis-binds comma arguments use `powershell -NoProfile -Command "& script.ps1 ..."` (fixed properly in Task 1, canonical form kept everywhere).
- Qt: `QT_QPA_PLATFORM=offscreen` for all scenario runs (gate sets it; direct pytest runs must set it too).
- Gate evidence convention: per-tier `receipt.json` + `junit.xml` + `pytest.txt` under `docs/quality/evidence/<date>/gate-<tier>/`; palette scenario evidence under `docs/quality/evidence/<date>/palette/<scenario-id>/run-1/` and `run-2/`; `--basetemp` under `.test_runtime\` (git-ignored). The two runs must retain separate JUnit, log and receipt files. Use distinct `-OutDir` values for full gate runs because `quality-gates.ps1` clears a tier's previous evidence before each run.
- **Skip never equals PASS.** A scenario whose mapping skipped is NOT_TESTED, not PASS. PARTIAL tiers need an explicit cause accepted by the owner.
- **Twice-in-a-row convention:** every palette segment and the final full gate run **twice in a row without manual intervention between runs**; compare test node ids and outcomes from both preserved JUnit files and tier receipts, including skips, failures and collection errors. A second green run alone does not prove the first one.
- Each live scenario creates uniquely named state and cleans up its projects, jobs and issued tokens in fixture teardown (including failure paths). Check for active jobs before and after each scenario; never rely on dropping the shared stand database to clean up between runs.
- Registry and scenario user-facing text in Russian (record fields `expected_behavior`, checkpoint sections, matrix rows). Machine keys (`beta`, `live`, ids) unchanged.
- Feature freeze in force: only test/acceptance infrastructure and narrow fixes (RED→GREEN with a defect entry, numbering continues from D23).
- Excluded from W3 entirely (spec «Границы беты»): Karakal chain acceptance, stitch, physical layers, electrical, Krona, CSliser, KateGB, offline exchange, legacy migration, Windows service, Hello, layer workflow, QC dialog, dev-session, attach project. LDAP stays off (`domain_auth=false` on `/api/v1/health`).

## Review Focus

Failure classes the palette could hide — each pinned by a test in the named task:

1. **Silent tier mis-binding** — `-Tier a,b` through `powershell -File` binds one bogus element and validates the wrong tier set. Task 1 pins: `-Tier live,smoke` via `-File` runs both tiers, unknown tier names abort with exit 2.
2. **False gateway green** — a gateway scenario counting PASS while uploads actually ran in proxy mode. Task 18/19/20 pin: acceptance asserts `transfer["mode"] == "gateway"` and the Rust exe health (`service == "kraken-blob-gateway"`); without the exe the group self-skips to PARTIAL and the final report states it — never PASS.
3. **Cross-scenario state pollution** — one scenario's leftovers (projects, active jobs, issued tokens) poison the next. Tasks 10–15 pin: unique `palette-<id>-<runid>` project names, teardown that runs even after assertion failure, pre/post-flight assertion `no queued/staging/running/recovery_required jobs`, and the protected-project invariance snapshot (`protected_unchanged`) on every live scenario.
4. **Concurrent-user ACL leaks** — the two-user scenario accidentally passes by sharing one session/token. Task 15 pins: two simultaneous `RemoteHttpClient` sessions with distinct principal ids; observer-before-role gets 403/409 (`error_conditions` of SRV-PROJECTS/HUB-PROJECT-ROLES), and only the assigned performer's «Моя работа» lists the job.
5. **Registry re-bind drift** — statuses flipped to PASS without a fresh JUnit binding (the pre-freeze failure mode). Task 23 extends the existing `resolve_node_id` validation to every `beta: true` PASS row, verifies every mapped node passed in fresh retained JUnit, and checks that coverage comparison refuses PASS→FAIL regressions.

---

### Task 1: Comma-tier argument fix in `scripts/quality-gates.ps1` (chore, W3 backlog T9-first)

**Files:**
- Modify: `scripts/quality-gates.ps1` (param block lines ~167–178; validation after `$Tier = @($Tier | Select-Object -Unique)` line ~198)
- Modify: `docs/quality/beta-stand.md`, `docs/quality/checkpoint-2026-11-17-beta.md` (only the runbook command note, if present)

**Interfaces:**
- Consumes: existing tier names (`smoke,unit,integration,gui,stateful,e2e,extended,regression,live`).
- Produces: `-Tier` accepts comma-joined elements however PowerShell 5.1 `-File` passes them; unknown names abort loudly. All later tasks may use the canonical form `powershell -NoProfile -ExecutionPolicy Bypass -Command "& scripts\quality-gates.ps1 -Tier live -SkipCoverage"` and the `-File` comma form.

- [ ] **Step 1: Reproduce the trap** — run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\quality-gates.ps1 -Tier live,smoke -SkipCoverage`; expect the current ValidateSet error on the single value `live,smoke` (record the error text to `.test_runtime\comma-trap.txt`).
- [ ] **Step 2: Remove `[ValidateSet(...)]`** from the `$Tier` parameter (keep `[string[]]$Tier, [Parameter(Mandatory=$true)]`) and insert manual validation immediately after `$Tier = @($Tier | Select-Object -Unique)`:

```powershell
# PS 5.1: `powershell -File script.ps1 -Tier a,b` binds the WHOLE
# "a,b" as one string[] element, so ValidateSet cannot see the parts.
# Re-split every element on commas, then validate.
$knownTiers = @('smoke','unit','integration','gui','stateful','e2e','extended','regression','live')
$Tier = @($Tier | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$unknownTiers = @($Tier | Where-Object { $knownTiers -notcontains $_ } | Select-Object -Unique)
if ($unknownTiers.Count -gt 0) {
    Write-Host ('[gate] unknown tier(s): ' + ($unknownTiers -join ', ') + '; known: ' + ($knownTiers -join ', '))
    exit 2
}
if ($Tier.Count -eq 0) { Write-Host '[gate] no tiers specified'; exit 2 }
$Tier = @($Tier | Select-Object -Unique)
```

  Update the `.PARAMETER Tier` docstring: comma- or space-separated, works through both `-File` and `-Command`.
- [ ] **Step 3: Verify GREEN** — (a) `-File ... -Tier live,smoke -SkipCoverage` runs registry+live+smoke and ends `GATE RESULT: ...` with exit code matching tier statuses (live PARTIAL without env is exit 0); (b) `-File ... -Tier nonsense -SkipCoverage` prints `unknown tier(s): nonsense` and exits 2; (c) `-Command "& scripts\quality-gates.ps1 -Tier gui -SkipCoverage"` still behaves as before.
- [ ] **Step 4: Commit** — `git commit -m "fix(quality): re-split comma tiers in quality-gates.ps1 (PS 5.1 -File mis-binding)"`

### Task 2: Probe timeout in `tests/offline/test_pytorch_resolution.py` (chore, W3 backlog T5-minor)

**Files:**
- Modify: `tests/offline/test_pytorch_resolution.py`

**Interfaces:**
- Consumes: existing probe `_offline_resolution_available()` (D21 gate) and pin test `test_offline_resolution_probe_matches_dry_run`.
- Produces: bounded probe (no unbounded `uv` hang in any tier); semantics unchanged (any failure/timeout ⇒ False ⇒ deterministic skip).

- [ ] **Step 1: Add the timeout** — in `_offline_resolution_available()` add `timeout=120,` to the `subprocess.run(...)` call. `subprocess.TimeoutExpired` already derives from `SubprocessError`, which the existing `except (OSError, subprocess.SubprocessError): return False` catches — no other change.
- [ ] **Step 2: Verify** — `.venv\Scripts\python.exe -m pytest tests/offline/test_pytorch_resolution.py -q`; expect the same result as the 2026-11-18 baseline (2 skipped on this machine, exit 0); run twice — identical.
- [ ] **Step 3: Commit** — `git commit -m "fix(tests): bound the offline-resolution probe with timeout=120 (D21 gate)"`

### Task 3: `-Restart` ordering guard in `scripts/start-beta-stand.ps1` (chore, W3 backlog T6-minor)

**Files:**
- Modify: `scripts/start-beta-stand.ps1` (move the restart block at lines ~197–202 to after the `check-database-name` guard, i.e. after line ~229)
- Test: `tests/test_beta_stand_guard.py` (extend, following the file's existing subprocess/script-text style; if it only tests `beta_stand.py`, add a script-order pin asserting `check-database-name` appears before the `-Restart` teardown invocation in the script text — match the file's established mechanism)

**Interfaces:**
- Consumes: `tools/quality/beta_stand.py check-database-name` (strict schema-mode check).
- Produces: `-Restart -DatabaseName <refused>` cannot run `stop-beta-stand.ps1` (which drops databases/schemas) before the name guard passes.

- [ ] **Step 1: Write the failing pin** — add the ordering test to `tests/test_beta_stand_guard.py`; run `.venv\Scripts\python.exe -m pytest tests/test_beta_stand_guard.py -q`; expect FAIL.
- [ ] **Step 2: Move the block** — relocate `if ($Restart -and (Test-Path -LiteralPath $statePath)) { ... stop-beta-stand.ps1 ... }` to sit immediately after the `$nameCheck` guard block and before `New-Item -ItemType Directory ... $standDir`. Keep behavior otherwise identical.
- [ ] **Step 3: Verify GREEN** — pin test passes; then run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start-beta-stand.ps1 -Restart -DatabaseName kraken_local`; expect `FAIL preconditions: -DatabaseName refused: ...` and exit 1 with **no** `[beta-stand] -Restart: running stop-beta-stand.ps1 first` line printed.
- [ ] **Step 4: Commit** — `git commit -m "fix(scripts): check -DatabaseName before -Restart teardown in start-beta-stand.ps1"`

### Task 4: nt-skipif pin on the D22 DOS-name regression test (chore, W3 backlog T8-minor)

**Files:**
- Modify: the D22 regression file registered in `docs/quality/registry/defects.json` (defect D22, `regression_tests` field) that exercises Windows DOS short names / `remove_stored_path`

**Interfaces:**
- Consumes: `kraken_defect = "D22"` marker (regression tier discovery via `tools/quality/list_defect_files.py`).
- Produces: the DOS-name assertions skip deterministically on non-NT filesystems instead of failing; unchanged on Windows (all CI/dev runs here).

- [ ] **Step 1: Locate** — `.venv\Scripts\python.exe -m tools.quality.list_defect_files.py` output + grep for the D22 file; identify the test(s) whose behavior is NT-specific (short-name / `\\?\` handling).
- [ ] **Step 2: Add the pin** — decorate the NT-specific test(s) with `@pytest.mark.skipif(os.name != "nt", reason="Windows short-name semantics required")` (add `import os` if missing). Do not touch the rest of the file.
- [ ] **Step 3: Verify** — `.venv\Scripts\python.exe -m pytest <d22-file> -q` on this machine: all still PASS (nt machine, no skips). Commit: `git commit -m "test(quality): nt-skipif on the D22 DOS-name regression test"`

### Task 5: Palette harness groundwork — UI test config, env recipe, tier-list registration policy

**Files:**
- Create: `docs/quality/palette-w3.md` (palette index: every scenario id, its env, its actions, its evidence path — mirrors the checkpoint «W3 сценарий» column)
- Modify: `docs/quality/beta-stand.md` (new section «KRAKEN_UI_TEST_CONFIG для палитры W3»)
- Create (throwaway, NOT committed): `.test_runtime\make-ui-test-config.py`, `.test_runtime\beta-stand\ui-test-config.json`, `.test_runtime\palette-source\` (100 numbered PNGs)

**Interfaces:**
- Consumes: stand outputs `.test_runtime\beta-stand.env.json`, `.test_runtime\beta-stand\state.json` (`beta_username`, `beta_password`), the `KRAKEN_UI_TEST_CONFIG` contract from `tests/test_live_ui_workflow.py` (header docstring: `server_url`, `username`, `password`, `source_directory` with 100 numbered images), the plugin-chain contract from `tests/test_live_server_plugin_chain.py` (`f"{stage}_machine_secret"`, `model_reuse_receipt`, `results_directory/receipt.json` chaining — the last two are supplied in Task 18).
- Produces: a documented, repeatable env recipe used by every live scenario task; `docs/quality/palette-w3.md` as the palette «учёт» block (spec: каждая строка палитры — идентифицируемый блок).

- [ ] **Step 1: Inspect the stand** — confirm its existing state and gateway health before deciding whether a restart is needed. The gateway exe is already present; do not assume a missing-exe failure or restart a running shared stand solely for this step.
- [ ] **Step 2: Verify the UI source + config** — generate 100 numbered 64×64 PNGs and `ui-test-config.json` if absent; derive `source_directory` from the resolved checkout path (`Path.cwd()` when invoked from the repo), never a fixed drive letter. Verify all 100 files, JSON keys and path existence. Reuse existing generated files if valid.
- [ ] **Step 3: Document the env recipe** in `docs/quality/beta-stand.md` — the canonical caller preamble for the palette (used verbatim by scenario tasks, date-substituted):

```powershell
$repo = (Resolve-Path .).Path
$env:KRAKEN_LIVE_WORKER_CONFIG = Join-Path $repo '.test_runtime\beta-stand\live-worker-config.json'
$env:KRAKEN_TEST_POSTGRES_URL = (Get-Content (Join-Path $repo '.test_runtime\beta-stand.env.json') -Raw | ConvertFrom-Json).KRAKEN_TEST_POSTGRES_URL
$env:KRAKEN_UI_TEST_CONFIG = Join-Path $repo '.test_runtime\beta-stand\ui-test-config.json'
$env:QT_QPA_PLATFORM = 'offscreen'
# Run the scenario command in this same PowerShell process.
```

  State explicitly: the stand provisions ONE account (`beta-stand`); additional users are registered by the scenarios themselves via `POST /api/v1/auth/accounts` on loopback (open registration, `domain_auth=false`; see `src/kraken_server/interface/http/routes/auth.py`).
- [ ] **Step 4: Write `docs/quality/palette-w3.md`** with the scenario index table (id, env gate, action ids, test file, evidence path) initialized from this plan's final mapping table.
- [ ] **Step 5: Commit** — `git commit -m "docs(quality): W3 palette harness — UI test config recipe and palette index"`

### Task 6: Scenario S01 — локальный вход и создание аккаунта (offline GUI)

**Covers:** HUB-BOOTSTRAP-ACCOUNT, HUB-LOGIN (+ error branches «A local account already exists», «Неверные учётные данные или аккаунт временно заблокирован.»).

**Files:**
- Use existing `tests/quality/test_s01_local_identity.py` for S01; add S02–S04 tests alongside it or in focused files under `tests/quality/` according to the current GUI tier registration.
- Modify: `scripts/quality-gates.ps1` only if the existing `tests/quality/test_s01_local_identity.py` is absent from the GUI tier.
- Modify: `docs/quality/registry/actions.json` (rows below)

**Interfaces:**
- Consumes: local account store + login window driven via pytest-qt with `configure_settings_directory(tmp_path)` isolation (pattern used across the gui tier); no stand.
- Produces: gui-tier JUnit node ids from `tests/quality/test_s01_local_identity.py` bound to the two registry rows.

- [x] **Step 1: Write the S01 tests** — implemented in `tests/quality/test_s01_local_identity.py`; inspect actual test names and assertions before binding the registry.
  - `test_s01_bootstrap_creates_first_local_account_and_session` — with an empty settings dir, the bootstrap window creates the first account; assert active local session and persisted account.
  - `test_s01_second_local_account_rejected` — after one account exists, a second creation attempt is rejected with the message containing `A local account already exists` (registry `error_conditions` text).
  - `test_s01_login_wrong_password_blocked_message` — login with a wrong password shows exactly `Неверные учётные данные или аккаунт временно заблокирован.`
  - `test_s01_login_correct_password_opens_session` — happy path.
- [ ] **Step 2: Verify S01 twice** — run `tests/quality/test_s01_local_identity.py` via the existing `scripts/run_s01_palette.ps1` or its equivalent, preserving separate evidence for each run. Fix product code only via RED→GREEN defect if behavior mismatches the registry promise.
- [ ] **Step 3: Check GUI tier registration** for the actual S01 file; add it only if absent, then run the GUI tier. Expect exit 0 with no failures and S01 node ids present in JUnit.
- [ ] **Step 4: Re-bind registry rows** — `HUB-BOOTSTRAP-ACCOUNT` and `HUB-LOGIN` use the actual collected node ids in `tests/quality/test_s01_local_identity.py`; mark PASS only after the two preserved runs and GUI tier confirm them. Evidence points to the corresponding JUnit files.
- [ ] **Step 5: Regenerate docs + commit** — `.venv\Scripts\python.exe -m tools.quality.generate_quality_docs --write`; `git commit -m "test(quality): palette S01 — локальный вход и создание аккаунта (HUB-BOOTSTRAP-ACCOUNT, HUB-LOGIN re-earned)"`

### Task 7: Scenario S02 — настройки, источник обновлений, вид матрицы, хранилища (offline GUI)

**Covers:** HUB-PREFERENCES, HUB-UPDATE-SOURCE, MGR-MATRIX-VISUAL, HUB-CONFIGURE-ROOTS.

**Files:**
- Modify: the S01 test module under `tests/quality/`, or add a focused sibling module if that keeps the scenarios clearer.
- Modify: `docs/quality/registry/actions.json`

- [ ] **Step 1: Write the tests** (same isolation idiom): `test_s02_preferences_dialog_persists_server_url_and_theme` (Вид → «Настройки…»: server address, accounts list, theme round-trip into QSettings; secrets go to keyring — assert only that no secret value lands in QSettings), `test_s02_update_source_dialog_stores_manifest_url` (Вид → «Источник обновлений…»: URL/path written to QSettings), `test_s02_matrix_visual_mode_persisted` (Вид → рамка/заливка ячейка → QSettings), `test_s02_configure_roots_writes_settings` («Настроить хранилища…» with two temp roots → QSettings contains both).
- [ ] **Step 2: Run twice, green** — run the actual S02 module; retain JUnit and logs for each run.
- [ ] **Step 3: Re-bind registry rows** — HUB-PREFERENCES, HUB-UPDATE-SOURCE, MGR-MATRIX-VISUAL, HUB-CONFIGURE-ROOTS: PASS only for each action's actual collected test node ids and fresh JUnit evidence.
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette S02 — настройки, источник обновлений, вид матрицы, корни хранилищ"`

### Task 8: Scenario S03 — привязка хранилища и локальный резервный контур (offline GUI)

**Covers:** HUB-BIND-WORKSPACE, HUB-EXPORT-BACKUP, HUB-IMPORT-BACKUP (+ error branch «Испорченная копия отклоняется ошибкой проверки»).

**Files:**
- Modify: the S01 test module under `tests/quality/`, or add a focused sibling module.
- Modify: `docs/quality/registry/actions.json`

- [ ] **Step 1: Write the tests** — `test_s03_bind_workspace_dialog_attaches_two_roots` (project creation prompts «Настроить хранилище проекта»; two temp roots attach; assert both roots bound), `test_s03_export_backup_creates_kraken_backup_folder` («Создать резервную копию» → `kraken-backup-*` directory with manifest), `test_s03_import_backup_restores_project` (export then import into a clean store → project visible in catalog), `test_s03_import_corrupted_backup_rejected` (corrupt the copied archive's payload file, expect the verification error dialog/error from `error_conditions`, no partially restored project).
- [ ] **Step 2: Run twice, green.**
- [ ] **Step 3: Re-bind** HUB-BIND-WORKSPACE, HUB-EXPORT-BACKUP, HUB-IMPORT-BACKUP rows → `status: "PASS"`, tests = their `test_s03_*` node ids, evidence = `docs/quality/evidence/<run-date>/gate-gui/junit.xml`.
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette S03 — привязка хранилища, локальный экспорт/импорт резервной копии с веткой отказа"`

### Task 9: Scenario S04 — проверка целостности и админ-обновление каталога (offline GUI)

**Covers:** HUB-SCAN-INTEGRITY (+ error branch «Обнаружены ошибки»), HUB-ADMIN-REFRESH.

**Files:**
- Modify: the S01 test module under `tests/quality/`, or add a focused sibling module.
- Modify: `docs/quality/registry/actions.json`

- [ ] **Step 1: Write the tests** — `test_s04_scan_integrity_reports_clean` (Администрирование → «Проверить целостность» on a healthy local store → «Проверка завершена»), `test_s04_scan_integrity_reports_errors` (corrupt a stored journal/event file per the store's format → dialog reports «Обнаружены ошибки»), `test_s04_admin_refresh_lists_projects_and_audit` («Обновить каталог» → project list + ≤500 audit rows populated).
- [ ] **Step 2: Run twice, green.**
- [ ] **Step 3: Re-bind** HUB-SCAN-INTEGRITY, HUB-ADMIN-REFRESH rows → PASS + `test_s04_*` bindings + gui junit evidence.
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette S04 — проверка целостности (в т.ч. ветка ошибок) и админ-обновление каталога"`

### Task 10: Scenario S05/S06/S07 — серверный каталог: проекты, слои, репрезентации (live)

**Covers:** SRV-PROJECTS, SRV-LAYERS, SRV-REPRESENTATIONS (+ error branches «Чужой If-Match — 409 revision.conflict», «Наблюдатель не переименовывает проект — 403» context, invalid upload rejection at artifact preflight).

**Files:**
- Create: `tests/test_beta_palette_server_api.py` (docstring: `"""Палитра W3 — серверные маршруты на живом стенде (KRAKEN_LIVE_WORKER_CONFIG). S05/S06/S07/S08."""`)
- Modify: `scripts/quality-gates.ps1` (`$liveFiles += 'tests/test_beta_palette_server_api.py'` — same commit as the first collected test so the live tier never hits exit 5)
- Modify: `docs/quality/registry/actions.json`

**Interfaces:**
- Consumes: `_connect(config, directory)` and the protected-project invariance snapshot from `tests/test_live_server_worker_acceptance.py`; unique project names `palette-s05-<uuid4hex8>`. A delete request is a behavior assertion, not cleanup: fixture finalizers must drain jobs and revoke scenario tokens. If physical project removal is unavailable, run the two acceptance passes against a disposable isolated stand/schema and tear it down afterward; do not accumulate projects on the permanent stand.
- Produces: live JUnit ids `tests/test_beta_palette_server_api.py::test_s05_*` etc.; the `RemoteHttpClient._call` verb/payload/idempotency-key/if-match style of the existing live tests.

- [ ] **Step 1: Write S05** — `test_s05_project_crud_and_revision_conflict`: create project (2×1) → list contains it → read → workspace binding → rename (fresh `If-Match`) → archive → catalog hides unless archived shown → restore → **stale `If-Match` rename → HTTP 409 `revision.conflict`** → delete request (`«Запросить удаление»` semantics: project remains, заявка recorded). Assert `application/problem+json` shape on the 409.
- [ ] **Step 2: Write S06** — `test_s06_layer_routes_full_surface`: on a `palette-s06-*` project: add layer via scan route (PNG folder), external layer route, import route, rename, reorder, archive, delete-by-name; final project GET still readable. `test_s06_deleted_layer_references_gone`: after layer delete, `GET /layers/{id}` → 404/410-class problem response and project history retains the deletion event (grounded: SRV-LAYERS routes list includes delete; registry error branch for layer delete requires exact name mirrored in S11 GUI).
- [ ] **Step 3: Write S07** — `test_s07_representations_import_and_viewport`: create representation (image), import plan route, import route, inputs preparation route, frames listing, viewport route; `test_s07_upload_stale_revision_409`: artifact series upload preflight with stale `If-Match` → 409 (SRV-ARTIFACTS error branch; runs in proxy mode — do NOT assert transfer mode here, gateway-mode assertion belongs to G16/G17).
- [ ] **Step 4: Run twice on the stand** — canonical preamble (Task 5); run pytest with separate `--junitxml=docs/quality/evidence/<run-date>/palette/S05-S07/run-1/junit.xml` and `run-2/junit.xml`, distinct `--basetemp` paths and retained logs. Expect exit 0, 0 skips, identical node outcomes and `protected_unchanged`.
- [ ] **Step 5: Re-bind** SRV-PROJECTS, SRV-LAYERS, SRV-REPRESENTATIONS rows → PASS only after the mapped node ids pass in both retained S05–S07 JUnit files; reference both evidence paths.
- [ ] **Step 6: Regenerate docs + commit** — `git commit -m "test(quality): palette S05-S07 — серверные проекты, слои, репрезентации (live)"`

### Task 11: Scenario S08 — команды, websocket, заметки/пакеты, протокол заданий, планы, статистика (live)

**Covers:** SRV-COMMANDS (+403), SRV-WEBSOCKET, SRV-NOTES-CHECKS-PACKAGES, SRV-JOBS-AGENT-PROTOCOL (+wrong token 401, duplicate complete 403, CAS 409), SRV-PLANS (+409), SRV-HISTORY-STATISTICS.

**Files:**
- Modify: `tests/test_beta_palette_server_api.py`
- Modify: `docs/quality/registry/actions.json`

- [ ] **Step 1: Write S08 tests** —
  - `test_s08_project_command_accepted_and_forbidden`: `POST /api/v1/projects/{id}/commands` as owner → 200 + journal event; as a freshly registered no-role account → 403 (`error_conditions`: «Отказ без прав — 403»).
  - `test_s08_websocket_ping_subscribe_unsubscribe`: WS `/ws` with the session token: ping → pong; subscribe project → `catalog_changed` event observed after a catalog-mutating call from a second HTTP session; unsubscribe → no further events.
  - `test_s08_notes_checks_package_return`: note create/update; check record; package route producing the check ZIP and the return-ZIP round trip; work-order (поручение) routes — SRV-NOTES-CHECKS-PACKAGES full surface except legacy migration (out of beta).
  - `test_s08_jobs_agent_protocol_error_branches`: against a queued job created via the work-order route: `POST /api/v1/agent/lease` with a garbage bearer token → **401**; CAS conflict — two racing lease claims, loser gets **409**; after a legitimate one-shot `ObservedWorker` completes the job, repeating `complete` → **403** (duplicate). Use the `ObservedWorker` class with `allowed_jobs` set to exactly this job id.
  - `test_s08_plans_crud_and_revision_conflict`: work-plan routes (plan create, template, analytics), stale `If-Match` → 409.
  - `test_s08_history_and_statistics`: history route returns the journal events created by S05-style mutations on the scenario project; statistics route computes per-project aggregates.
- [ ] **Step 2: Run twice** on the stand (same command as Task 10 step 4, junit to `.../palette/S08/junit.xml`); exit 0, 0 skips, matching.
- [ ] **Step 3: Re-bind** SRV-COMMANDS, SRV-WEBSOCKET, SRV-NOTES-CHECKS-PACKAGES, SRV-JOBS-AGENT-PROTOCOL, SRV-PLANS, SRV-HISTORY-STATISTICS rows → PASS with the `test_s08_*` bindings and evidence path above.
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette S08 — команды, websocket, пакеты, протокол заданий, планы, статистика (live, ветки 401/403/409)"`

### Task 12: Scenario S09 — здоровье агента с токеном подключения (live)

**Covers:** AGT-HEALTH (+ error branch «Запрос без токена отклонён»).

**Files:**
- Create: `tests/test_beta_palette_agent_health.py` (docstring: `"""Палитра W3 — HTTP health агента на живом стенде. S09."""`)
- Modify: `scripts/quality-gates.ps1` (`$liveFiles += ...`, same commit)
- Modify: `docs/quality/registry/actions.json`

**Interfaces:**
- Consumes: the standalone loopback control mode of `python -m kraken_agent` with `--data-dir <temp> --connection-file <temp>/connection.json --independent`. Do not pass `--server-url`: that selects `ServerAgentWorker.run_forever()` and does not start the HTTP control server or write a connection file. Remote worker authentication is exercised in the separate server/worker scenarios.
- Produces: AGT-HEALTH binding; documented agent bring-up recipe for W4 smoke.

- [ ] **Step 1: Write the test** — `test_s09_agent_health_ok_with_token_and_rejected_without`: fixture starts the standalone agent, waits for its connection file, requests `<connection-file url>/api/v1/health` with its control token → `status: ok`; without a token → 401. Teardown calls authenticated `/api/v1/shutdown`, waits for process exit and verifies the port closes. If retaining the stand assertion, verify `/api/v1/health` still answers independently.
- [ ] **Step 2: Run twice** on the stand (junit `.../palette/S09/junit.xml`), exit 0, no skips, identical.
- [ ] **Step 3: Re-bind** AGT-HEALTH → PASS + binding + evidence.
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette S09 — health агента с токеном и без (live)"`

### Task 13: Scenario S10 — вкладочный вход и жизненный цикл серверного проекта через Hub (live UI)

**Covers:** HUB-LOGIN-TABBED, HUB-CONNECT-SAVED-SERVER, HUB-REFRESH-CATALOG, HUB-OPEN-PROJECT, HUB-RENAME-PROJECT, HUB-ARCHIVE-PROJECT, HUB-RESTORE-PROJECT, HUB-DELETE-PROJECT (+ «Неполное подтверждение именем отклоняется»), HUB-PROJECT-PROPERTIES (+ LOGIN-TABBED error «Неверные учётные данные — то же сообщение блокировки»).

**Files:**
- Create: `tests/test_beta_palette_hub_ui_project_cycle.py` (docstring: `"""Палитра W3 — окно входа и серверный каталог проектов через Hub (офскрин). S10."""`)
- Modify: `scripts/quality-gates.ps1` (`$liveFiles += ...`, same commit)
- Modify: `docs/quality/registry/actions.json`

**Interfaces:**
- Consumes: `KRAKEN_UI_TEST_CONFIG` fixture pattern from `tests/test_live_ui_workflow.py` (`live_ui` equivalents: `DesktopController` over `DualCatalogService`, `modal()` dialog driver, `select_catalog_project()`); `GET /api/v1/health` `domain_auth: false`.
- Produces: S10 JUnit ids; scenario projects cleaned by a fixture finalizer, or isolated in a disposable stand for the pair of runs if the API has no physical project removal. Verify no active jobs after teardown.

- [ ] **Step 1: Write the tests** —
  - `test_s10_tabbed_login_local_and_server_tabs_visible_ldap_hidden`: login window shows tabs «Локальный» and «Сервер»; with health answering `domain_auth: false`, no LDAP tab is present; server-tab login with the stand account creates/uses the mirror account and lands in the catalog (dual catalog: server projects visible).
  - `test_s10_saved_server_reconnect_attaches_server_client`: with saved address+account, post-login state shows the connected server client (dual catalog), per HUB-CONNECT-SAVED-SERVER `expected_behavior`.
  - `test_s10_wrong_server_credentials_blocked_message`: wrong password → the same blocking message as the registry error condition.
  - `test_s10_project_cycle_through_ui`: create `palette-s10-<uuid8>` → «Обновить» (catalog reload after out-of-band HTTP project creation is reflected) → double-click open (page «Проект», menus «Действия»/«Слои…» enabled) → rename via dialog → «В архив» (hidden unless «Показывать архивные») → «Восстановить» → «Свойства» (read-only dialog shows name+state) → «Удалить»: type partial name → rejected; type the exact full name → delete request accepted, project leaves the active catalog per server semantics.
- [ ] **Step 2: Run twice** on the stand with the canonical preamble + `KRAKEN_UI_TEST_CONFIG`; junit `.../palette/S10/junit.xml`; exit 0, 0 skips, identical.
- [ ] **Step 3: Re-bind** the nine rows (HUB-LOGIN-TABBED, HUB-CONNECT-SAVED-SERVER, HUB-REFRESH-CATALOG, HUB-OPEN-PROJECT, HUB-RENAME-PROJECT, HUB-ARCHIVE-PROJECT, HUB-RESTORE-PROJECT, HUB-DELETE-PROJECT, HUB-PROJECT-PROPERTIES) → PASS with `test_s10_*` bindings + evidence.
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette S10 — вкладочный вход и жизненный цикл серверного проекта (live UI)"`

### Task 14: Scenario S11 — мультиобъектность слоёв: 2 проекта, 2+ слоя, 2+ варианта, использовать-then-удалить (live UI)

**Covers:** HUB-MATRIX-CONTEXT, HUB-REORDER-LAYERS, HUB-REPRESENTATION-LIFECYCLE; re-earns in-flow HUB-ADD-LAYER, HUB-LAYER-CONTEXT, HUB-ADD-REPRESENTATION (currently PASS — see Task 23 policy; their bindings gain the palette node ids as additional evidence), + error branches «Смесь PNG и JPG блокирует скан», «Удаление требует точного имени слоя», «Импорт неподдерживаемого вида отклоняется».

**Files:**
- Create: `tests/test_beta_palette_hub_ui_layers.py` (docstring: `"""Палитра W3 — слои, варианты изображений, контекст матрицы (live UI). S11."""`)
- Modify: `scripts/quality-gates.ps1` (`$liveFiles += ...`, same commit)
- Modify: `docs/quality/registry/actions.json`

**Interfaces:**
- Consumes: `KRAKEN_UI_TEST_CONFIG`; source PNGs from `source_directory`; the layers graph menu helpers `_select_representation_in_graph` / `node_menu` (import from existing live test modules).
- Produces: the non-vectorized half of the spec's multi-object rule (2 projects, 2+ layers per project with 2+ image variants and primary switch, use-then-delete layer with the project readable); project names `palette-s11a-<uuid8>` / `palette-s11b-<uuid8>`.

- [ ] **Step 1: Write the tests** —
  - `test_s11_two_projects_two_layers_each`: create two server projects; add 2 layers to each from PNG folders (distinct images from `source_directory`); assert matrix shows all layers of the opened project.
  - `test_s11_two_variants_switch_primary_and_context_menu`: on one layer add a second image representation («Добавить изображение»), switch the active variant via the Layers menu graph (the combo is hidden — use `_select_representation_in_graph`), assert matrix/task targets follow the active variant; open the matrix cell context menu and assert items «Выгрузка файлов», «Выдать работу…», «Свойства кадра…», «Свойства вектора…» (presence only; actions exercised in S12/G17); «Свойства вектора…» greyed with no vector is acceptable — assert menu structure only.
  - `test_s11_lifecycle_rename_note_activate_archive_deactivate`: on a board node: переименование, заметка, активация, деактивация, архив of a representation; history events recorded (SRV-REPRESENTATIONS already re-earned — assert via UI + journal).
  - `test_s11_use_then_delete_layer_project_stays_readable`: use a layer (upload frames / open matrix), delete it via the layer menu with the exact name (partial name → rejected per `error_conditions`), project still opens and reads; second layer intact.
  - `test_s11_mixed_png_jpg_scan_blocked` and `test_s11_unsupported_representation_import_rejected`: error branches from HUB-ADD-LAYER / HUB-ADD-REPRESENTATION `error_conditions`.
- [ ] **Step 2: Run twice** (junit `.../palette/S11/junit.xml`), exit 0, 0 skips, identical, `protected_unchanged`.
- [ ] **Step 3: Re-bind** HUB-MATRIX-CONTEXT, HUB-REORDER-LAYERS (add a drag-order step inside `test_s11_two_projects_two_layers_each` asserting the reorder event), HUB-REPRESENTATION-LIFECYCLE → PASS; HUB-ADD-LAYER / HUB-LAYER-CONTEXT / HUB-ADD-REPRESENTATION: append the S11 node ids to existing `tests` and add the palette evidence path (status stays PASS, confirmed again by the final full-gate in Task 23).
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette S11 — мультиобъектность слоёв и варианты изображений (live UI)"`

### Task 15: Scenario S12 — два пользователя: выдача → «Моя работа» → коррекция → приёмка → повторная выдача; план восстановления (live UI)

**Covers:** HUB-MY-WORK, HUB-WORK-PLAN; re-earns HUB-ISSUE-WORK and HUB-REVIEW-RETURN in-flow (PASS list, binding augmentation); error branches «Отказ без каталога» (выдача и приёмка) + ACL leak pin.

**Files:**
- Create: `tests/test_beta_palette_hub_ui_work_flow.py` (docstring: `"""Палитра W3 — серверная выдача работы двумя пользователями, «Моя работа», приёмка, план восстановления (live UI). S12."""`)
- Modify: `scripts/quality-gates.ps1` (`$liveFiles += ...`, same commit)
- Modify: `docs/quality/registry/actions.json`

**Interfaces:**
- Consumes: `KRAKEN_UI_TEST_CONFIG`; second user provisioning **in-scenario**: `POST /api/v1/auth/accounts` with `{"username": "palette-performer-<uuid6>", "password": <generated>, "display_name": "Палитра Корректор"}` on loopback (registration is open at `domain_auth=false`), then owner grants the «корректор» project role via the ACL route/UI (HUB-PROJECT-ROLES mechanism — already PASS-verified in gui tier, used here as infrastructure, not re-earned); **two concurrent sessions**: two `RemoteServerProjectService` clients + two `DesktopController` instances alive in the same test process (distinct principal ids asserted).
- Produces: the two-concurrent-users half of the multi-object rule; the state cycle выдача → коррекция → приёмка → повторная выдача того же слоя.

- [ ] **Step 1: Write the tests** —
  - `test_s12_two_users_issue_work_mywork_accept_reissue`: user A (owner) opens `palette-s12-<uuid8>`, «Выдать работу…» server-side fixing the task to user B (исправление векторов); assert B's session (second controller) «Моя работа» lists the task and A's does not; B runs the listed actions (обновить, открыть, выгрузить, пометку коррекции, «на сервер» return), A «Загрузить проверенные файлы…» (приёмка слоя после ручной коррекции), journal records приёмка, then A issues the SAME layer again (повторная выдача) and B sees it. Assert both sessions concurrently live (both principals authenticated before the first action completes).
  - `test_s12_issue_without_folder_rejected` and `test_s12_review_return_without_folder_rejected`: the «Отказ без каталога» error branches of HUB-ISSUE-WORK / HUB-REVIEW-RETURN.
  - `test_s12_observer_cannot_see_private_project_until_role`: freshly registered user C (no role) does not see `palette-s12-*` in the server catalog; after observer grant, read-only visible; observer rename attempt → 403 (pins Review Focus #4).
  - `test_s12_work_plan_recovery_setup_and_export`: «План восстановления…» on the project — create a plan selecting which layers export on frame issue, template reuse, board visible, CSV export lands on disk with the plan rows (beta flow item «настроить план восстановления слоя»).
- [ ] **Step 2: Run twice** (junit `.../palette/S12/junit.xml`), exit 0, 0 skips, identical.
- [ ] **Step 3: Re-bind** HUB-MY-WORK, HUB-WORK-PLAN → PASS with bindings + evidence; append S12 node ids to HUB-ISSUE-WORK / HUB-REVIEW-RETURN `tests`/`evidence` (status unchanged PASS).
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette S12 — выдача/коррекция/приёмка двумя пользователями, план восстановления (live UI)"`

### Task 16: Scenario S13 — плагины: процесс из Hub и CLI-дымы Contour/NeuralImage

**Covers:** HUB-PLUGINS, PLUGIN-CONTOUR-HELP, PLUGIN-NEURALIMAGE-VERSION.

**Files:**
- Create: `tests/test_plugin_cli_smoke.py` (docstring: `"""Палитра W3 — CLI-доступность плагинов бета-объёма. S13 (CLI)."""`)
- Create: `tests/test_beta_palette_hub_plugins.py` (docstring: `"""Палитра W3 — запуск процесса плагина из Hub (live UI). S13 (Hub)."""`)
- Modify: `scripts/quality-gates.ps1` (smoke tier run list → `@('tests/test_desktop_runtime_smoke.py', 'tests/test_plugin_cli_smoke.py')` as one invocation; `$liveFiles += 'tests/test_beta_palette_hub_plugins.py'`)
- Modify: `docs/quality/registry/actions.json`

- [ ] **Step 1: Write the CLI smoke tests** — `test_contour_help_exits_zero` (`subprocess.run([sys.executable, "-m", "contour", "--help"])` → exit 0, stdout non-empty), `test_neuralimage_version_exits_zero` (`-m neuralimage --version` → exit 0). No torch CUDA init in these paths — if `--version` imports torch, assert it still exits 0 on CPU (it does in the e2e family); do NOT invoke `torch.cuda` (D15).
- [ ] **Step 2: Write the Hub plugin test** — `test_s13_plugins_dialog_opens_plugin_process`: live-UI fixture; «Плагины» → «Открыть» for Contour; assert a distinct plugin subprocess exists (psutil-free: `QProcess` state or the controller's plugin process handle) and terminates cleanly on dialog close; no agent job created (`remote.plugin_jobs()` unchanged).
- [ ] **Step 3: Run both twice** — CLI: `.venv\Scripts\python.exe -m pytest tests/test_plugin_cli_smoke.py -q`; then smoke tier `powershell -NoProfile -ExecutionPolicy Bypass -Command "& scripts\quality-gates.ps1 -Tier smoke -SkipCoverage"` (expect the two `KRAKEN_SMOKE_*`-gated skips → PARTIAL is acceptable for the tier; the new CLI tests themselves pass — confirm in `gate-smoke/junit.xml`); Hub test via canonical preamble with junit `.../palette/S13/junit.xml`.
- [ ] **Step 4: Re-bind** HUB-PLUGINS → PASS with the live binding; PLUGIN-CONTOUR-HELP / PLUGIN-NEURALIMAGE-VERSION → PASS with `tests/test_plugin_cli_smoke.py::test_contour_help_exits_zero` / `::test_neuralimage_version_exits_zero` and `docs/quality/evidence/<run-date>/gate-smoke/junit.xml`.
- [ ] **Step 5: Regenerate docs + commit** — `git commit -m "test(quality): palette S13 — плагины: процесс из Hub и CLI-дымы (Contour, NeuralImage)"`

### Task 17: Scenario S14 — резервное копирование и восстановление сервера (integration, требует provisioning)

**Covers:** ADMIN-SERVER-BACKUP-RESTORE (+ error branch «Восстановление на занятой базе отклоняется»).

**Files:**
- Modify: `docs/quality/beta-stand.md` (new section «KRAKEN_TEST_PG_BIN и роль с CREATEDB для S14»)
- Modify: `docs/quality/registry/actions.json`

**Interfaces:**
- Consumes: `tests/test_server_backup.py` (fixture `recovery`: needs `KRAKEN_TEST_POSTGRES_URL` AND `KRAKEN_TEST_PG_BIN`; creates/drops two `kraken_backup_test_*` databases — requires a role with `CREATE DATABASE`; the stand's `kraken_local_app` role does NOT have it — this is the documented W2 gap).
- Produces: re-earned ADMIN-SERVER-BACKUP-RESTORE or an explicitly owner-signed exception.

- [ ] **Step 1: Ask the owner once** for the PostgreSQL superuser (or a role grant) on the local cluster; provision a dedicated role `kraken_backup_test_owner` with `CREATEDB` (psql one-liner recorded in the runbook, secret stored DPAPI under `.test_runtime\`, never in git). If the owner declines or is unavailable, jump to Step 5 (exception path).
- [ ] **Step 2: Record `KRAKEN_TEST_PG_BIN`** — locate `pg_dump.exe`/`pg_restore.exe` next to the cluster (`Get-Command pg_dump` or `C:\Program Files\PostgreSQL\<version>\bin`), append both `KRAKEN_TEST_PG_BIN` and the backup-role URL into `.test_runtime\beta-stand.env.json` via `beta_stand.py env-json` extension or documented manual edit; document in the runbook.
- [ ] **Step 3: Run the suite twice** — canonical preamble with both variables set: `.venv\Scripts\python.exe -m pytest tests/test_server_backup.py -q --junitxml=docs/quality/evidence/<run-date>/palette/S14/junit.xml`; expect 0 skips, exit 0; the occupied-database branch is covered by `test_native_consistent_backup_restore_and_repeat` (`pytest.raises(ServerBackupError)` on restoring over a non-empty target) — if no existing test covers it explicitly, add one (RED→GREEN) before binding.
- [ ] **Step 4: Re-bind** ADMIN-SERVER-BACKUP-RESTORE → `status: "PASS"`, `tests: ["tests/test_server_backup.py"]`, evidence = the palette junit; commit `git commit -m "test(quality): palette S14 — native backup/verify/restore сервера (ADMIN-SERVER-BACKUP-RESTORE re-earned)"`
- [ ] **Step 5: Exception path (only if Step 1 failed)** — keep `status: "NOT_TESTED"`, set `exception_reason` to «native backup/restore требует роль с CREATEDB и KRAKEN_TEST_PG_BIN — provisioning вне возможностей закрытой машины; решение заказчика перенести на W4»; do NOT count toward readiness; commit `git commit -m "docs(quality): S14 backup/restore — явное ограничение, ждёт provisioning роли (решение заказчика)"` and surface it in the Task 23 report.

### Task 18: Gateway prerequisite (GATE) — поставка `kraken-blob-gateway.exe` + токен-расширение стенда

**This is a prerequisite audit, not an acceptance scenario.** The executable is already present in the current checkout; confirm its health and gateway transfer mode before Tasks 19–22. Supply from an external kit only if this check fails.

**Files:**
- Modify: `scripts/start-beta-stand.ps1` (issue per-capability machine tokens + extend `live-worker-config.json`)
- Modify: `tests/test_beta_stand_guard.py` or the stand's companion test file (pin the extended config contract)
- Modify: `docs/quality/beta-stand.md` (supply paths)

**Interfaces:**
- Consumes: the existing `blob_gateway\target\release\kraken-blob-gateway.exe` and the stand's gateway health/transfer checks. Rebuild offline only if the binary is absent or invalid.
- Produces: gateway-enabled stand (`GET http://127.0.0.1:8081/health` → `service == "kraken-blob-gateway"`); only the beta-required NeuralImage machine capability and model reuse receipt if absent from the current config. Exclude Karakal capabilities and chain tests from W3.

- [ ] **Step 1: Verify existing gateway** — inspect binary, `/health`, stand config and an actual `transfer.mode == "gateway"` upload. Record a blocker only if the current setup fails.
- [ ] **Step 2: Pin test for missing config extension** — if needed, extend stand tests for the NeuralImage secret and optional model reuse receipt; run RED before implementation.
- [ ] **Step 3: Implement missing beta configuration only** — if necessary, issue `beta-stand-worker-ni` (`frames.binary-segment.v2`), add `neuralimage_machine_secret` and `model_reuse_receipt` to `$liveConfig`, and revoke the issued token in the stop script. Do not issue a Karakal token. Tests green.
- [ ] **Step 4: Validate the stand with the gateway** — restart only when configuration changed and after checking the shared stand can safely be stopped. Expect ALL lines PASS, including gateway health and a real gateway-mode upload.
- [ ] **Step 5: Produce the reuse receipt** — canonical preamble + `KRAKEN_RUN_TRAINING_LIVE=1`; run `.venv\Scripts\python.exe -m pytest tests/test_neuralimage_headless_training.py -q`; expect exit 0 and the receipt file present; wire its path into `live-worker-config.json` (re-run the stand script; it reuses components and refreshes the config).
- [ ] **Step 6: Chain sanity** — canonical preamble; run only the NeuralImage chain node id and its required worker acceptance predecessor, preserving their receipts. Expect exit 0 and no skips; Karakal tests remain outside W3.
- [ ] **Step 7: Commit if changed** — commit only necessary stand configuration and tests; otherwise record the audit evidence.

### Task 19: Scenario G16 — Blob Gateway PUT/GET/HEAD с билетом и пережитие перезапуска (live, gateway)

**Covers:** BLOB-PUT-GET-RESTART (+ error branches «Нет билета, неверный HMAC, чужая операция, просрочка — 401»); re-confirms PERSISTENT-GATEWAY-UPLOAD and SRV-ARTIFACTS in gateway mode.

**Files:**
- Modify: `tests/test_blob_gateway.py` (extend `test_rust_gateway_streams_python_signed_upload_and_range_download` with a process-restart + identical-bytes GET step, or add `test_rust_gateway_survives_restart_returns_same_bytes` following the file's existing subprocess fixture style)
- Modify: `docs/quality/registry/actions.json`

- [ ] **Step 1: RED** — add the restart test: start the Rust gateway subprocess (existing fixture mechanism), PUT a blob with a valid HMAC ticket, kill the process, restart on the same storage dir, GET → identical bytes + HEAD matches digest; ticket-less GET → 401. Run with the gateway env: expect FAIL before any needed fixture change (the exe now exists in this group only).
- [ ] **Step 2: GREEN** — implement the fixture restart support in the test file only (no product change unless a real defect surfaces → defect entry from D24).
- [ ] **Step 3: Run twice** — `.venv\Scripts\python.exe -m pytest tests/test_blob_gateway.py -q --junitxml=docs/quality/evidence/<run-date>/palette/G16/junit.xml`; expect 0 skips (exe present), identical results; then the full gateway-mode confirmation: `.venv\Scripts\python.exe -m pytest tests/test_live_server_worker_acceptance.py -q` (its `_upload_frame` asserts `transfer["mode"] == "gateway"`) twice.
- [ ] **Step 4: Re-bind** BLOB-PUT-GET-RESTART → PASS, `tests: ["tests/test_blob_gateway.py::test_rust_gateway_streams_python_signed_upload_and_range_download", "tests/test_blob_gateway.py::test_ticket_round_trip_rejects_tampering", "<restart test node id>"]`, evidence `["docs/quality/evidence/<run-date>/palette/G16/junit.xml"]`.
- [ ] **Step 5: Regenerate docs + commit** — `git commit -m "test(quality): palette G16 — blob gateway PUT/GET/HEAD, 401-ветки и перезапуск процесса"`

### Task 20: Scenario G17 — векторизация слоя агентом + серверный контур Contour, 2 слоя отдельно (live, gateway)

**Covers:** AGT-OP-VECTORIZE (+ отмена/падение → failed, проект не испорчен); completes the vectorized half of the multi-object rule (2+ слоя, каждый векторизуется отдельно, задание берёт активный вариант).

**Files:**
- Create: `tests/test_beta_palette_vectorize_multi.py` (docstring: `"""Палитра W3 — векторизация двух слоёв отдельными заданиями агента (live, gateway). G17."""`)
- Modify: `scripts/quality-gates.ps1` (`$liveFiles += ...`, same commit)
- Modify: `docs/quality/registry/actions.json`

**Interfaces:**
- Consumes: the `live_worker` fixture pattern and `ObservedWorker` (import per plugin-chain precedent); a scenario project `palette-g17-<uuid8>` with 2 layers (each with 2 image variants, active variant switched — reuse S11 provisioning at API level), upload frames in gateway mode (`_upload_frame` asserts mode).
- Produces: two separate `frames.vectorize.v1` jobs (one per layer), each leased by the one-shot worker, CIF results imported as vector representations of the CORRESPONDING layer; active-variant selection observable (job inputs reference the active representation).

- [ ] **Step 1: Write the tests** — `test_g17_two_layers_vectorized_separately_active_variant`: enqueue vectorize per layer via the server job/work-order routes, run the one-shot `ObservedWorker` (registry: `frames.vectorize.v1` → `(sys.executable, "-m", "contour.worker")`), wait both jobs complete, assert each layer got its own vector representation artifact with distinct digests and the receipt written to `results_directory`; `test_g17_cancel_marks_failed_project_intact`: enqueue, cancel via the job route while queued/running, assert job state `failed`/`cancelled` per protocol and project revision/history unpolluted (registry error branch).
- [ ] **Step 2: Run twice** (gateway stand up, background agent OFF per runbook), junit `.../palette/G17/junit.xml`; exit 0, 0 skips, `protected_unchanged`.
- [ ] **Step 3: Re-bind** AGT-OP-VECTORIZE → PASS, `tests: ["tests/test_live_server_worker_acceptance.py::test_contour_machine_worker_publishes_versions_and_hub_reopens", "tests/test_beta_palette_vectorize_multi.py::test_g17_two_layers_vectorized_separately_active_variant", "tests/test_beta_palette_vectorize_multi.py::test_g17_cancel_marks_failed_project_intact", "tests/test_pipeline_live_contour.py"]`, evidence `["docs/quality/evidence/<run-date>/gate-live/junit.xml", "docs/quality/evidence/<run-date>/palette/G17/junit.xml"]` (the pipeline file binding requires `KRAKEN_RUN_PIPELINE_LIVE=1` during the final gate — Task 23).
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette G17 — векторизация двух слоёв отдельными заданиями (live, gateway)"`

### Task 21: Scenario G18 — подготовка выборки агентом + интерактивная выборка Contour (live, gateway)

**Covers:** AGT-OP-DATASET-PREPARE, PLUGIN-CONTOUR-DATASET-INTERACTIVE (+ отказ для кадра без парного CIF; отмена задания). Headless Contour поддерживает `frames.dataset.prepare.v1` и создаёт ZIP выборки; прежнее требование отвергать саму операцию противоречило реализации и существующему тесту `plugins/contour/tests/unit/test_headless_dataset.py`.

**Files:**
- Create: `tests/test_beta_palette_agent_ops.py` (docstring: `"""Палитра W3 — агентские операции выборки (live, gateway). G18."""`)
- Modify: `scripts/quality-gates.ps1` (`$liveFiles += ...`, same commit)
- Modify: `docs/quality/registry/actions.json`

- [ ] **Step 1: Write the agent test** — `test_g18_dataset_prepare_worker_imports_dataset`: scenario project `palette-g18-<uuid8>` with a vectorized layer (G17 provisioning mechanism at API level); `ObservedWorker` registry `{"frames.dataset.prepare.v1": PluginProcessSpec(..., (sys.executable, "-m", "contour.worker"))}`; enqueue via the job routes with a plan referencing the vector representation; worker completes; assert the dataset artifact appears in the project (registry: «Результат импортирован в проект (выборка)») and `allowed_jobs`-guarded lease observed.
- [ ] **Step 2: Write the error/interactive tests** — `test_g18_headless_worker_rejects_unpaired_cif` verifies a malformed input fails clearly without a dataset ZIP; `test_g18_interactive_prepare_via_hub_gateway_path` launches the real Contour process in interactive mode, drives the return action and verifies the imported ZIP and gateway tickets. `test_g18_cancel_queued_dataset_job_preserves_project` covers queued-job cancellation. Both headless and interactive modes must accept valid `frames.dataset.prepare.v1` inputs.
- [ ] **Step 3: Run twice** (junit `.../palette/G18/junit.xml`), exit 0, 0 skips, matching.
- [ ] **Step 4: Re-bind** AGT-OP-DATASET-PREPARE → PASS with `test_g18_dataset_prepare_worker_imports_dataset` and `test_g18_cancel_queued_dataset_job_preserves_project`; PLUGIN-CONTOUR-DATASET-INTERACTIVE → PASS with `test_g18_interactive_prepare_via_hub_gateway_path` and `test_g18_headless_worker_rejects_unpaired_cif`; evidence = two retained palette JUnit files and the live gate JUnit.
- [ ] **Step 5: Regenerate docs + commit** — `git commit -m "test(quality): palette G18 — подготовка выборки агентом и интерактивная ветка Contour (live, gateway)"`

### Task 22: Scenario G19 — обучение модели и бинарная сегментация на живом контуре (live, gateway, bounded CPU)

**Covers:** AGT-OP-TRAIN, AGT-OP-BINARY-SEGMENT (+ отмена/падение — reuse of the fault branch already observed for training in the e2e D15.1 gate: keep the CUDA-free bounded CPU path).

**Files:**
- Modify: `docs/quality/registry/actions.json`; add `tests/test_beta_palette_train_live.py` because the existing bounded training pipeline uses a local agent and does not prove the `--server-url` contract of AGT-OP-TRAIN.

- [ ] **Step 1: Run the bounded training + inference suites twice** — canonical preamble + `KRAKEN_RUN_MODEL_FUNCTIONAL=1` and `KRAKEN_RUN_TRAINING_LIVE=1`: `.venv\Scripts\python.exe -m pytest tests/test_pipeline_live_neural_model.py tests/test_neuralimage_headless_training.py tests/test_live_server_plugin_chain.py -q -k "not karakal and not stitch"` with junit `.../palette/G19/junit.xml`; also run `tests/test_beta_palette_train_live.py` twice against the Server Gateway. Expected: `dataset.model.train.v1` bounded one-epoch CPU training completes both locally and through a server machine lease; `frames.binary-segment.v2` inference publishes new binary PNG masks; the chain test reuses the verified checkpoint (`model_reuse_receipt`, epoch_upper_bound == 1). Exit 0, 0 skips (CUDA must not be touched — D15 family).
- [ ] **Step 2: Verify outputs** — project gains the model checkpoint artifact and new binary PNG masks, all readable via artifact download routes with sha256 match (the live training and chain tests assert digests).
- [ ] **Step 3: Re-bind** AGT-OP-TRAIN → `tests: ["tests/test_beta_palette_train_live.py::test_g19_server_training_publishes_verified_checkpoint", "tests/test_pipeline_live_neural_model.py"]`; AGT-OP-BINARY-SEGMENT → `tests: ["tests/test_pipeline_live_neural_model.py", "tests/test_live_server_plugin_chain.py::test_neuralimage_reuses_verified_training_and_publishes_real_inference"]`; both PASS, evidence = the two complete palette G19 JUnits plus the two server training JUnits.
- [ ] **Step 4: Regenerate docs + commit** — `git commit -m "test(quality): palette G19 — обучение и бинарная сегментация на живом контуре (bounded CPU, gateway)"`

### Task 23: Финал — полный гейт дважды, аудит свежести, W3-секция чекпоинта, отчёт владельцу

**Files:**
- Modify: `docs/quality/checkpoint-2026-11-17-beta.md` (fill «W3 сценарий» column for all rows; append section «## Итог W3 — палитра живых сценариев принятия» in Russian)
- Modify: `docs/quality/registry/actions.json` (final consistency: every re-verified row PASS with fresh bindings; HUB-CHECK-UPDATES stays `BLOCKED` with `exception_reason` ending «…приёмка — W4 (тестовый источник обновлений)»; failures keep NOT_TESTED + updated `exception_reason`)
- Modify: `tests/test_quality_registry.py` (pin: every `beta: true` PASS action has non-empty `tests` and each node id resolves with the existing `resolve_node_id` helper); add a repeatable evidence audit for fresh passed JUnit nodes.
- Create: `docs/quality/evidence/<final-date>/` (full-gate artifacts, committed)

- [ ] **Step 1: Write the anti-drift pin** in `tests/test_quality_registry.py` and confirm it passes against the current registry (all rows re-bound so far comply).
- [ ] **Step 2: Full gate, run 1** — confirm the stand is healthy; restart only if necessary and safe. Set ALL env: `KRAKEN_LIVE_WORKER_CONFIG`, `KRAKEN_TEST_POSTGRES_URL`, `KRAKEN_UI_TEST_CONFIG`, `KRAKEN_RUN_PIPELINE_LIVE=1`, `KRAKEN_RUN_MODEL_FUNCTIONAL=1`, `KRAKEN_RUN_TRAINING_LIVE=1`; run `scripts\quality-gates.ps1 -Tier smoke,unit,integration,gui,stateful,e2e,regression,live -OutDir docs\quality\evidence\<final-date>\full-run-1`. Expect 0 FAIL; record each PARTIAL cause explicitly.
- [ ] **Step 3: Full gate, run 2** — repeat the identical tier/env command with `-OutDir docs\quality\evidence\<final-date>\full-run-2`, without manual intervention or stand restart. Compare both retained JUnit sets and receipts by tier and test node id, including skips, failures and errors; results must match.
- [ ] **Step 4: Coverage + freshness audit** — verify both runs' `coverage-report.json`/`coverage-compare.json`: 0 PASS→FAIL regressions; for each `beta: true` PASS row, resolve every node id with `resolve_node_id`, require a passed result in fresh retained JUnit and check JUnit timestamp/mtime after the freeze point. Keep this audit as a committed, repeatable quality test/tool. Verify the W2 PASS rows still show PASS in both full runs; triage regressions RED→GREEN and rerun rather than re-binding by hand.
- [ ] **Step 5: W3 checkpoint section (Russian)** — append «## Итог W3 — палитра живых сценариев принятия»: per-scenario table (id, дата прогона, junit-путь, вердикт), итоговый расклад `beta:true` (N PASS / M NOT_TESTED с причинами / 1 BLOCKED → W4), gateway-статус (поставлен/нет), список явных исключений (S14 fallback, HUB-CHECK-UPDATES), «Следующий шаг: W4 — поставка». Fill the «W3 сценарий» column of the readiness table with the ids from the mapping table below (ids не выдумываются — только из этой таблицы).
- [ ] **Step 6: Commit evidence + checkpoint** — `git add docs/quality/evidence/<final-date> docs/quality/checkpoint-2026-11-17-beta.md docs/quality/registry/actions.json docs/quality/acceptance-matrix.md docs/quality/functional-inventory.md docs/quality/palette-w3.md tests/test_quality_registry.py; git commit -m "docs(quality): W3 итог — палитра прогнана, реестр перепривязан к свежим доказательствам"`
- [ ] **Step 7: Report to the owner** — compact table: PASS/NOT_TESTED/BLOCKED counts vs the 15/45/1 baseline; the two-user + multi-object proof points; gateway status and what remains for W4 (installer, update source, smoke); every exception needing a signature.

---

## Mapping table: task → scenario id → action ids (all 45 NOT_TESTED + 15 PASS policy + 1 BLOCKED)

| Task | Scenario | Actions re-earned |
|---|---|---|
| 6 | S01 | HUB-BOOTSTRAP-ACCOUNT, HUB-LOGIN |
| 7 | S02 | HUB-PREFERENCES, HUB-UPDATE-SOURCE, MGR-MATRIX-VISUAL, HUB-CONFIGURE-ROOTS |
| 8 | S03 | HUB-BIND-WORKSPACE, HUB-EXPORT-BACKUP, HUB-IMPORT-BACKUP |
| 9 | S04 | HUB-SCAN-INTEGRITY, HUB-ADMIN-REFRESH |
| 10 | S05 | SRV-PROJECTS |
| 10 | S06 | SRV-LAYERS |
| 10 | S07 | SRV-REPRESENTATIONS |
| 11 | S08 | SRV-COMMANDS, SRV-WEBSOCKET, SRV-NOTES-CHECKS-PACKAGES, SRV-JOBS-AGENT-PROTOCOL, SRV-PLANS, SRV-HISTORY-STATISTICS |
| 12 | S09 | AGT-HEALTH |
| 13 | S10 | HUB-LOGIN-TABBED, HUB-CONNECT-SAVED-SERVER, HUB-REFRESH-CATALOG, HUB-OPEN-PROJECT, HUB-RENAME-PROJECT, HUB-ARCHIVE-PROJECT, HUB-RESTORE-PROJECT, HUB-DELETE-PROJECT, HUB-PROJECT-PROPERTIES |
| 14 | S11 | HUB-MATRIX-CONTEXT, HUB-REORDER-LAYERS, HUB-REPRESENTATION-LIFECYCLE (+ in-flow: HUB-ADD-LAYER, HUB-LAYER-CONTEXT, HUB-ADD-REPRESENTATION) |
| 15 | S12 | HUB-MY-WORK, HUB-WORK-PLAN (+ in-flow: HUB-ISSUE-WORK, HUB-REVIEW-RETURN) |
| 16 | S13 | HUB-PLUGINS, PLUGIN-CONTOUR-HELP, PLUGIN-NEURALIMAGE-VERSION |
| 17 | S14 | ADMIN-SERVER-BACKUP-RESTORE (или явное исключение, Step 5) |
| 18 | GATE (prereq, no actions) | — audit gateway and required NeuralImage config; extend only if needed |
| 19 | G16 | BLOB-PUT-GET-RESTART (+gateway-mode confirmation of PERSISTENT-GATEWAY-UPLOAD, SRV-ARTIFACTS) |
| 20 | G17 | AGT-OP-VECTORIZE |
| 21 | G18 | AGT-OP-DATASET-PREPARE, PLUGIN-CONTOUR-DATASET-INTERACTIVE |
| 22 | G19 | AGT-OP-TRAIN, AGT-OP-BINARY-SEGMENT |
| 23 | FINAL | coverage regen + live-tier PASS + checkpoint + report |

Count check against the checkpoint gap table: 45 NOT_TESTED = S01:2 + S02:4 + S03:3 + S04:2 + S05:1 + S06:1 + S07:1 + S08:6 + S09:1 + S10:9 + S11:3 + S12:2 + S13:3 + S14:1 + G16:1 + G17:1 + G18:2 + G19:2 = 45. ✔

**15 PASS policy (hybrid rule):** HUB-CREATE-PROJECT, HUB-PROJECT-ROLES, HUB-ADD-PERFORMER, HUB-EDIT-PERFORMER, HUB-DELETE-PERFORMER, HUB-REVIEW-RETURN, HUB-ADD-LAYER, HUB-LAYER-CONTEXT, HUB-ADD-REPRESENTATION, HUB-ISSUE-WORK, HUB-STATISTICS, SRV-HEALTH-SESSION, SRV-PRINCIPALS, SRV-ARTIFACTS, PERSISTENT-GATEWAY-UPLOAD — evidence dated 2026-11-18 already satisfies post-freeze freshness; each must remain PASS on the final full-gate JUnit of Task 23 Steps 2–4 (their mapped tiers re-run there). Rows touched by palette flows (S11/S12/G16) additionally gain palette bindings. Any that fail in the final gate → RED→GREEN triage, not manual re-bind. **HUB-CHECK-UPDATES** stays `BLOCKED` → W4 (тестовый источник обновлений — поставка W4), with the exception_reason kept current.

---

This plan was originally checked at `fefb38a`; the rebaseline above supersedes those historical assumptions. Recheck all remaining paths, routes and stand prerequisites against the checkout before execution.
