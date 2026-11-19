# Beta Stabilization W1–W2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the beta-scope quality gate machine-readable (`beta` registry flag + `live` tier), make the existing suites deterministic on the closed-network dev machine, stand up the repeatable server stand, and drive integration/gui/stateful tiers to zero FAIL — the W1–W2 waves of the beta-stabilization spec.

**Architecture:** The existing quality-registry toolchain (`docs/quality/registry/*.json` → `tools/quality/registry_schema.py` → `tools/quality/coverage_report.py` → `scripts/quality-gates.ps1`) gains a second axis: every action carries `beta: true|false`; gate coverage and PASS→FAIL regression comparison count only `beta: true` rows. A new `live` gate tier executes the live-stack suites (self-skipping to PARTIAL without the stand). One permanent stand (PostgreSQL test DB + Server + Blob Gateway + agent) makes every integration/live run repeatable.

**Tech Stack:** Python 3.14 (uv workspace, `.venv`), pytest / pytest-qt (offscreen), PowerShell 5.1, PostgreSQL (local), FastAPI server, Rust blob gateway, `kraken-agent`.

**Spec:** `docs/superpowers/specs/2026-11-17-beta-stabilization-design.md`

## Global Constraints

- Python floor: `requires-python = ">=3.14"` (pyproject.toml) — all tooling runs on the repo venv `.venv\Scripts\python.exe`.
- No new pip/uv dependencies; the quality tooling stays hand-rolled (see `registry_schema.py` docstring).
- Dev machine is a **closed network**: pypi.org and download.pytorch.org are unreachable (DNS fail). Nothing in this plan may require network access. `uv lock --check` currently FAILS on this machine (contour `build-system.requires = ["setuptools>=68", "wheel", "pybind11>=3.1"]` was added after the lock was generated). Mitigation measured 2026-11-17: `UV_CACHE_DIR=G:\kraken-transfer\uv-cache` makes the build deps resolvable offline; full lock regeneration still needs one network touch (cu132 index pages not cached) plus a `requires-python = ">=3.14,<3.15"` fence (Task 5, D21). Do not attempt `uv lock`/`uv sync` that mutates `uv.lock` or `.venv`.
- PowerShell 5.1 only: native commands must run through the existing `Invoke-Native` helper pattern; no `python -c` with embedded double quotes (all Python logic lives in `tools/quality/*.py`).
- Gate evidence convention: per-tier `receipt.json` + `junit.xml` + `pytest.txt` under `docs/quality/evidence/<date>/gate-<tier>/`; pytest `--basetemp` under `.test_runtime\quality-<uuid>\` (git-ignored); `QT_QPA_PLATFORM=offscreen`, `PYTHONUTF8=1`.
- Skips are never PASS (existing gate rule) — a tier with skips is PARTIAL.
- Working database `kraken_local` must not be dropped/modified; tests use separate schemas/databases. `D:\Heggs\Source` is read-only.
- Feature freeze: only fixes and acceptance infrastructure are committed to `main` while this plan runs.

## Review Focus

Failure classes the spec implies that no single task's happy-path test covers — each is pinned by a test in the named task:

1. **Hidden network need** — a unit/offline test that silently requires PyPI makes the suite machine-dependent (D16/D17 class). Task 5 pins: both `test_pytorch_resolution` tests skip deterministically (never fail) whenever offline uv resolution is unavailable on the machine.
2. **Registry migration half-applied** — a schema rule requiring `beta` while `actions.json` rows lack it breaks the gate on day one. Task 1 pins: validation of the real `docs/quality/registry/actions.json` passes with the flag on every row.
3. **Coverage denominator drift** — re-scoping filters from `mandatory` to `beta` could silently drop SRV rows or whole metrics from the report. Task 3 pins: `beta: false` actions leave all metric denominators while `beta: true` SRV actions stay in `api_operations`.
4. **`live` tier lying green without a stand** — a tier whose files all skip must be PARTIAL, not PASS. Task 4 pins the no-env run reports PARTIAL and exit 0.
5. **Stand leftovers** — a crashed server/agent process or a stale token poisons the next run. Task 6 pins: the stand script is idempotent (re-run against a live stand converges to the same state) and prints teardown; acceptance smoke passes twice in a row.

---

### Task 1: `beta` flag in registry schema + data migration

**Files:**
- Modify: `tools/quality/registry_schema.py`
- Modify: `docs/quality/registry/actions.json` (all 80 records)
- Test: `tests/test_quality_registry.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: every action record carries `beta: bool`; `beta: false` records carry `beta_out_reason: str` (non-empty) and MUST NOT carry it when `beta: true`. `ACTION_TIERS` includes `"live"`. The NOT_TESTED-exception rule fires on `beta: true` (replacing the `mandatory`-based condition). Downstream: `coverage_report.py` (Task 3) and `quality-gates.ps1` (Task 4) filter on `beta`.

- [ ] **Step 1: Write failing tests in `tests/test_quality_registry.py`** (follow the file's existing fixture style for synthetic action records):

```python
def test_action_requires_boolean_beta_flag(...):
    # record without "beta" -> validate_registry reports "'beta' must be a boolean"

def test_beta_false_requires_beta_out_reason(...):
    # beta=false, no beta_out_reason -> problem; with non-empty string -> no problem

def test_beta_true_must_not_carry_beta_out_reason(...):
    # beta=true + beta_out_reason -> problem

def test_not_tested_rule_scopes_to_beta_not_mandatory(...):
    # beta=true, status=NOT_TESTED, no exception_reason -> problem
    # beta=false, status=NOT_TESTED, no exception_reason -> NO problem

def test_live_is_a_valid_action_tier(...):
    # tier="live" passes _require_enum
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_quality_registry.py -q`; expect FAIL on the new tests (also note the real-registry validation test still passes only because rules are not yet active).

- [ ] **Step 3: Implement in `registry_schema.py`**: add `"live"` to `ACTION_TIERS` (keep existing order, append); in `_validate_action` add:
  `beta` must be a bool (message: `"'beta' must be a boolean"`); if `beta is False`, `beta_out_reason` must be a non-empty string; if `beta is True`, presence of `beta_out_reason` is a problem; replace the `mandatory`-scoped NOT_TESTED condition with `record.get("status") == "NOT_TESTED" and record.get("beta") is True and not record.get("exception_reason")`. Keep the `mandatory` bool validation unchanged (informational flag).

- [ ] **Step 4: Migrate `docs/quality/registry/actions.json`**: add `"beta": true` to the 55 actions and `"beta": false` + `"beta_out_reason": "вне объёма беты — <что>"` to the 25 listed in the spec §«Классификация существующих 80 действий». Edit with a throwaway script under `.test_runtime/` (json round-trip, `ensure_ascii=False`, indent 2) so key order stays stable; do not reformat untouched fields.

- [ ] **Step 5: Run the suite green** — `pytest tests/test_quality_registry.py -q`; expect all PASS including the test that validates the real registry files.

- [ ] **Step 6: Regenerate generated doc sections and commit**:

```bash
.venv\Scripts\python.exe -m tools.quality.generate_quality_docs
git add tools/quality/registry_schema.py docs/quality/registry/actions.json tests/test_quality_registry.py docs/quality/functional-inventory.md docs/quality/defects.md
git commit -m "feat(quality): beta-scope flag in action registry schema and data (55 in / 25 out)"
```

### Task 2: New beta actions for fresh login feature, admin backup and agent operations

**Files:**
- Modify: `docs/quality/registry/actions.json` (append 6 records)
- Modify: `docs/quality/functional-inventory.md` (regenerated)
- Test: `tests/test_quality_registry.py` (append)

**Interfaces:**
- Consumes: Task 1 schema (`beta`, tier `live`).
- Produces: action ids `HUB-LOGIN-TABBED`, `ADMIN-SERVER-BACKUP-RESTORE`, `AGT-OP-VECTORIZE`, `AGT-OP-DATASET-PREPARE`, `AGT-OP-TRAIN`, `AGT-OP-BINARY-SEGMENT` — all `beta: true`, `status: NOT_TESTED` with `exception_reason` («сценарий ещё не прогнан; палитра W3»). W3 plans bind tests to these ids via the `tests` field.

- [ ] **Step 1: Write the failing pin** — in `tests/test_quality_registry.py` add one test asserting the six ids exist in the real registry, are `beta: true`, and have non-empty `expected_behavior`; run, expect FAIL (ids missing).

- [ ] **Step 2: Append the six records** with fields in the existing record style (Russian text):
  - `HUB-LOGIN-TABBED` — surface `HUB`, tier `gui`, entry «Окно «Вход в Kraken» с вкладками „Локальный“ и „Сервер“»; expected: локальный вход работает всегда; серверный вход создаёт/использует зеркальный аккаунт; LDAP-вкладка скрыта, когда `/api/v1/health` отвечает `domain_auth: false`; error: неверные учётные данные — то же сообщение блокировки.
  - `ADMIN-SERVER-BACKUP-RESTORE` — surface `SRV`, tier `integration`, entry `kraken-admin backup-server / backup-verify / restore-server`; preconditions: отдельная тестовая база; error: восстановление на занятой базе отклоняется.
  - `AGT-OP-VECTORIZE` (`frames.vectorize.v1`, interactive), `AGT-OP-DATASET-PREPARE` (`frames.dataset.prepare.v1`), `AGT-OP-TRAIN` (`dataset.model.train.v1`), `AGT-OP-BINARY-SEGMENT` (`frames.binary-segment.v1`/`v2`) — surface `AGT`, tier `live`, entry «Задание агента <operation> через `--server-url`»; expected: результат импортирован в проект (CIF/выборка/чекпоинт/бинарный CIF); error: отмена задания, падение плагина — задание помечено failed, проект не испорчен.

- [ ] **Step 3: Run green** — `pytest tests/test_quality_registry.py -q`; regenerate docs as in Task 1 step 6.
- [ ] **Step 4: Commit** — `git commit -m "feat(quality): registry rows for tabbed login, admin backup and beta agent operations"`

### Task 3: Coverage report scoped to `beta`

**Files:**
- Modify: `tools/quality/coverage_report.py` (filters in `build_report`, ~lines 507–519; module docstring lines ~32–34)
- Test: `tests/test_quality_coverage_report.py`

**Interfaces:**
- Consumes: `beta` on action records (Task 1).
- Produces: `build_report(...)` denominators include only actions with `beta: true`: `error_units` filter `if action.get("beta")`; `ui_actions` predicate `bool(a.get("beta")) and a.get("surface") != "SRV"`; `api_operations` predicate `bool(a.get("beta")) and a.get("surface") == "SRV"`. Return shape unchanged.

- [ ] **Step 1: Write failing tests** in `tests/test_quality_coverage_report.py` (reuse the file's registry-fixture helpers): a `beta: false` HUB action with `error_conditions` leaves `error_paths` and `ui_actions` denominators; a `beta: false` SRV action leaves `api_operations`; a `beta: true` SRV action is counted in `api_operations`; `mandatory` no longer moves any denominator.
- [ ] **Step 2: Run, verify FAIL** — `pytest tests/test_quality_coverage_report.py -q`.
- [ ] **Step 3: Apply the three filter changes + docstring wording** (mandatory → beta) in `build_report`.
- [ ] **Step 4: Run green** — `pytest tests/test_quality_coverage_report.py tests/test_quality_registry.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat(quality): coverage report denominators scoped to beta actions"`

### Task 4: `live` tier in quality-gates.ps1

**Files:**
- Modify: `scripts/quality-gates.ps1` (`ValidateSet` line ~158; docstring tier mapping; new `$liveFiles` list + execution block next to the integration tier)

**Interfaces:**
- Consumes: gate tier conventions (`Invoke-GateTier`, receipts, skip=PARTIAL).
- Produces: `-Tier live` runs `$liveFiles` as one pytest invocation (all files live in the `tests/` import domain):

```powershell
$liveFiles = @(
  'tests/test_live_server_plugin_chain.py',
  'tests/test_live_server_worker_acceptance.py',
  'tests/test_live_ui_workflow.py',
  'tests/test_pipeline_live_contour.py',
  'tests/test_pipeline_live_neural_model.py',
  'tests/test_neuralimage_headless_training.py'
)
```

  Env-gate variables (`KRAKEN_LIVE_WORKER_CONFIG`, `KRAKEN_UI_TEST_CONFIG`, `KRAKEN_RUN_TRAINING_LIVE`, `KRAKEN_RUN_PIPELINE_LIVE`, `KRAKEN_RUN_MODEL_FUNCTIONAL`) are inherited from the caller environment, exactly like the integration tier inherits `KRAKEN_TEST_POSTGRES_URL`; the gate never starts the stand itself.

- [ ] **Step 1: Add `'live'` to `[ValidateSet(...)]` and the tier-mapping docstring** (state: self-skip without env → PARTIAL; PASS requires a live stand from Task 6).
- [ ] **Step 2: Add the execution block** mirroring the integration tier block (single invocation over `$liveFiles`, receipt + junit + pytest.txt, skip ⇒ PARTIAL).
- [ ] **Step 3: Verify no-env behavior** — `.venv\Scripts\powershell` is not needed; run `powershell -File scripts\quality-gates.ps1 -Tier live -SkipCoverage`; expect: tier PARTIAL (all skipped), exit code 0, evidence dir written.
- [ ] **Step 4: Verify unit tier unaffected** — `powershell -File scripts\quality-gates.ps1 -Tier unit -SkipCoverage`; expect PASS as before (exit 0, 0 skips counted as pass except the D15/D16/D17 probe skips already present — status PARTIAL acceptable only if it was PARTIAL on baseline; compare with today's baseline run).
- [ ] **Step 5: Commit** — `git commit -m "feat(quality): live tier in quality gates (stand-gated, skip=PARTIAL)"`

### Task 5: Deterministic offline tests + D21 dependency-supply defect

**Files:**
- Modify: `tests/offline/test_pytorch_resolution.py`
- Modify: `docs/quality/registry/defects.json`, `docs/quality/defects.md`

**Interfaces:**
- Consumes: uv offline cache state.
- Produces: module-level `_offline_resolution_available() -> bool` (single `functools.lru_cache`, one subprocess per run: execute the same `uv sync --directory <ROOT> --locked --dry-run --offline --all-groups --all-extras` that `_dry_run` performs, against a throwaway `UV_PROJECT_ENVIRONMENT`; exit 0 ⇒ True; any nonzero ⇒ False — this gates on the whole chain, because build-dep availability alone is insufficient: with the kit cache the build deps resolve but the locked resolution still fails on the Python≥3.15 torch fork until D21 is fully fixed) and `_require_offline_resolution()` which calls `pytest.skip("offline uv resolution unavailable on this machine (build deps and/or cu132 index pages missing, or unresolved Python>=3.15 torch fork) — see D21")`. Defined in `test_pytorch_resolution.py` itself, following the existing idiom `tests/offline/test_offline_sync.py::_pypi_is_reachable` (no conftest plumbing).

- [ ] **Step 1: Write the failing pin** — add to `tests/offline/test_pytorch_resolution.py` a test `test_offline_resolution_probe_matches_dry_run`: the probe's boolean equals the recorded result of one direct `_dry_run("--all-groups", "--all-extras")` invocation (True ⇒ it exits 0; False ⇒ it exits nonzero). Run, expect FAIL (no probe yet).
- [ ] **Step 2: Implement the probe in the test module** and add `_require_offline_resolution()` as the first statement in both existing tests. Do NOT touch `_dry_run` itself.
- [ ] **Step 3: Verify determinism** — `pytest tests/offline/test_pytorch_resolution.py tests/offline/test_offline_sync.py -q`; expect on this machine: pytorch tests 2 skipped with the stated reason, file exits 0, no failures. Run twice — same result.
- [ ] **Step 4: Register defect D21** in `docs/quality/registry/defects.json` (component `test_harness` + dependency supply; status OPEN with `exception_reason`: «на закрытой машине тесты гейтятся пробом; полное исправление из трёх частей: (1) uv-кэш из G:\\kraken-transfer содержит pybind11 3.1.0, wheel 0.48.0 и setuptools — при UV_CACHE_DIR=G:\\kraken-transfer\\uv-cache metadata Contour собирается офлайн; (2) в uv.lock нет build-зависимостей contour (лок устарел относительно build-system.requires) — пересобрать `uv lock` нужно один раз с доступом к download.pytorch.org: в uv-кэше комплекта есть байты колёс torch, но нет страниц простого индекса cu132 (simple-v20 содержит только .lock без тела ответа), офлайн-резолвер не может перечислить версии; (3) упаковки корня и neuralimage объявляют requires-python=">=3.14" без верхней границы, и uv строит неразрешимый форк Python>=3.15, где torch 2.12 не публикуется — при пересборке лока поставить fence ">=3.14,<3.15" (torch 2.12.0+cu132 существует только под 3.14); на закрытой машине это правка pyproject, применимая вместе с пересборкой лока»; regression_tests: the two pytorch tests) and a matching section in `docs/quality/defects.md` with RED evidence = baseline run `2 failed` (2026-11-17) and GREEN = deterministic skip + suite exit 0. Note in the defect: `uv lock --check` fails on the dev machine; this is the external prerequisite for W4 packaging and for any future dependency change. The kit's `dependencies/uv-cache` (`G:\kraken-transfer\uv-cache`) also serves as a `UV_CACHE_DIR` accelerator for offline resolution of everything except the cu132 index pages (note it in `docs/quality/beta-stand.md`).
- [ ] **Step 5: Run registry validation** — `pytest tests/test_quality_registry.py -q`; commit both files + helper + tests: `git commit -m "fix(tests): probe-gate offline pytorch-resolution tests on closed network (D21)"`

### Task 6: Permanent beta stand (Server + PostgreSQL + Gateway + agent)

**Files:**
- Create: `scripts/start-beta-stand.ps1`, `scripts/stop-beta-stand.ps1`
- Create: `docs/quality/beta-stand.md`

**Interfaces:**
- Consumes: `config/local/server.toml`, `config/local/database-url.secret`, `config/local/blob-gateway.secret` (exist, git-ignored); `kraken-admin` subcommands (`init`, `setup-server`, `bootstrap-admin`, agent tokens); live config JSON format documented in `tests/test_live_server_worker_acceptance.py` (header + `_config()`, ~lines 1–120: `KRAKEN_LIVE_WORKER_CONFIG` file with `server_url`, `session_secret`, …); `KrakenBlobGateway.exe` (blob_gateway build); `kraken-agent --server-url`.
- Produces: a repeatable bring-up that leaves in the caller environment (printed + written to `.test_runtime/beta-stand.env.json`, git-ignored): `KRAKEN_TEST_POSTGRES_URL` (a dedicated `kraken_beta_test` DB, never `kraken_local` data), `KRAKEN_LIVE_WORKER_CONFIG` (path to the JSON), plus a machine token for the leased worker. `stop-beta-stand.ps1` revokes tokens, stops processes, drops `kraken_beta_test`.

- [ ] **Step 1: Write `start-beta-stand.ps1`** (idempotent: detect already-listening 8080/8081 and live agent file-connection, reuse; steps: create/reset test DB → `kraken-admin migrate` on it → start server → start gateway → `kraken-admin` agent-token issue → start `kraken-agent --server-url` → write env json + config json). Print a one-line status per component (`PASS`/`FAIL: <reason>`).
- [ ] **Step 2: Write `docs/quality/beta-stand.md`** — how to start/stop, what env each gate tier needs, the twice-in-a-row convention from the spec, the rule «рабочую kraken_local не трогать».
- [ ] **Step 3: Verify smoke twice** — start the stand, then run `pytest tests/test_live_server_worker_acceptance.py -q` twice in a row; expect: no skips, exit 0 both times (this is the stand's own acceptance). Stop; verify ports closed and token revoked (subsequent auth attempts get HTTP 401).
- [ ] **Step 4: Commit** — `git commit -m "feat(quality): repeatable beta stand scripts and runbook"`

### Task 7: Beta package build attempt (early packaging risk pull)

**Files:**
- Modify: `packaging/build_windows.ps1` (only if the attempt fails for a fixable reason)
- Create: evidence note in `docs/quality/checkpoint-2026-11-17-beta.md` (started here, finished in Task 10)

**Interfaces:**
- Consumes: existing `.venv` (do NOT re-sync it — see D21 constraint), `packaging/KrakenDesktop.spec`, `packaging/KrakenServer.spec`, Inno Setup (`packaging/installer/*.iss`) if present on the machine.
- Produces: dated record of exit codes per build step and artifact paths; on failure, a defect entry (component per failing stage) with RED evidence. No silent «попробовали — не вышло».

- [ ] **Step 1: Run `powershell -File packaging\build_windows.ps1`** capturing full output to `.test_runtime/build-attempt-1.txt`.
- [ ] **Step 2: If FAIL — classify and fix narrowly** (missing tool vs. spec bug vs. D21 dependency supply). Fix what is fixable offline; register the rest as a defect with the exact error text.
- [ ] **Step 3: Record result in the checkpoint file** (exit code, artifacts, plugin set present in the build — verify contour + neuralimage are in, kategb/krona/csliser/karakal status noted as-is; trimming them from the package is NOT this task).
- [ ] **Step 4: Commit** the checkpoint note.

### Task 8: Integration tier green on the stand

**Files:**
- No fixed files — every failure gets its own RED→GREEN commit pair; statuses/evidence go to `docs/quality/registry/actions.json` + `docs/quality/acceptance-matrix.md`.

**Interfaces:**
- Consumes: Task 6 stand (`KRAKEN_TEST_POSTGRES_URL`), Task 3 scoping.
- Produces: `-Tier integration` with the env set: exit 0 and **0 FAIL**; each previously-skipped SRV action now backed by a JUnit node dated today.

- [ ] **Step 1:** Start stand; run `powershell -File scripts\quality-gates.ps1 -Tier integration -SkipCoverage` with `KRAKEN_TEST_POSTGRES_URL` set.
- [ ] **Step 2:** For each FAIL: reproduce → write/extend a failing regression test → minimal fix → GREEN → commit (defect registry entries follow the D1–D20 format; numbers continue from D21).
- [ ] **Step 3:** Re-run the whole tier until exit 0 with 0 FAIL; record PASS/PARTIAL statuses into the registry (with `tests` bindings) and regenerate docs.
- [ ] **Step 4: Commit** registry + matrix updates: `git commit -m "test(quality): integration tier green on beta stand"`.

### Task 9: GUI and stateful tiers green on the stand

Same shape as Task 8 with `powershell -File scripts\quality-gates.ps1 -Tier gui,stateful -SkipCoverage`; Qt failures follow the same RED→GREEN loop. Expected end state: exit 0, 0 FAIL; offscreen suites pass with the stand reachable for their server-backed pages.

- [ ] Steps 1–4 as in Task 8 for tiers `gui,stateful`. Commit: `git commit -m "test(quality): gui and stateful tiers green on beta stand"`.

### Task 10: Full-gate baseline after the freeze point + W2 checkpoint

**Files:**
- Create/modify: `docs/quality/checkpoint-2026-11-17-beta.md` (final state)
- Modify: registry statuses + `docs/quality/evidence/<date>/coverage-report.json`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: one full run `powershell -File scripts\quality-gates.ps1 -Tier smoke,unit,integration,gui,stateful,e2e,regression,live` with the stand up; committed coverage report as the **freshness baseline** (every `beta: true` PASS is dated at/after the freeze point — hybrid rule); a gap list feeding the W3 (palette) and W4 (delivery) plans: per `beta: true` action — PASS / PARTIAL(reason) / NOT_TESTED with the planned palette scenario id.

- [ ] **Step 1:** Run the full gate with the stand; triage any last FAIL (RED→GREEN).
- [ ] **Step 2:** Commit the coverage report + checkpoint document; the checkpoint ends with the explicit «следующий шаг» line pointing at the W3 palette plan.
- [ ] **Step 3:** Report to the product owner: current beta-action readiness table (compact) and the W3 plan draft trigger.

---

## Not in this plan

- W3 palette live scenarios (own plan — needs the working stand from Task 6 to design scenario provisioning realistically; Karakal excluded).
- W4 delivery (installer composition per beta scope, test update source over HTTP/file share, manual smoke protocol, handoff).
- `uv.lock` regeneration and uv-cache warming (D21 — the kit at `G:\kraken-transfer` supplies the build deps offline, but the lock rebuild still needs one network touch to download.pytorch.org (index pages for cu132 are not cached) plus the `">=3.14,<3.15"` fence; external prerequisite, not a task here).
- Removing excluded plugins from the installer (only their build status is observed in Task 7).
