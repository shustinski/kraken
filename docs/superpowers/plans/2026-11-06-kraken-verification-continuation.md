# Kraken Continuous Verification — Continuation 2026-11-06 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing Kraken quality system (markdown registries + `reproduce-stabilization.ps1`) into an enforced, machine-readable, executable verification system, and complete the remaining live acceptance of the real 18×18 Heggs project through the genuine Server→ServerAgentWorker→plugin→Hub path.

**Architecture:** Machine-readable JSON registries (`docs/quality/registry/`) become the single source of truth; the existing markdown documents are rendered from them by a generator, and a pytest enforcement suite fails the build when a resolved defect lacks a live regression test or a mandatory action is untracked. Stateful testing uses Hypothesis `RuleBasedStateMachine` driven against the real service layers (no mocks of the services under test). Live acceptance is pytest scenarios run against the running server from `config/local/server.toml`, with receipts/JUnit/evidence per run, exactly like the existing `live-worker` blocks. Quality gates are local PowerShell runners (the repo has no hosted CI — remote is a local bundle).

**Tech Stack:** Python 3.14.4 (uv, `.venv`), pytest + pytest-qt (PyQt6, offscreen), hypothesis (new dev extra), pytest-cov (already installed), PowerShell 5.1, real processes: `kraken-server`, `kraken-blob-gateway.exe`, `kraken-agent`/ServerAgentWorker, plugin subprocesses, PostgreSQL 16 (already running locally).

**Spec:** The user's mission brief (14 sections) delivered 2026-11-06; prior state in `docs/quality/checkpoint-2026-11-01.md`, `docs/quality/defects.md`, `docs/quality/functional-inventory.md`, `docs/quality/acceptance-matrix.md`, `docs/quality/test-gaps.md`. Environment facts from the user: server starts via `kraken-server --config config/local/server.toml`; test project folders `D:\Heggs\Source` (layer subfolders, 18×18 frames each, **read-only source data**).

## Global Constraints

- Never commit secrets or authentication material. `config/local/` (server.toml + `*.secret`) must stay git-ignored.
- Never modify existing user projects (server project `test` / `7c622c09-73fa-552f-9a52-e80b64aeb3ef`) or user data under `D:\Heggs\Source`. Destructive operations only in isolated quality projects/databases.
- Do not weaken or remove existing tests to get green. Skipped tests are NOT PASS.
- Status vocabulary everywhere: PASS / FAIL / PARTIAL / BLOCKED / NOT TESTED / NOT APPLICABLE.
- Every executed suite produces receipt + raw log + JUnit under `docs/quality/evidence/<YYYY-MM-DD>/`; no verification claim without execution evidence.
- KateGB is excluded from mandatory acceptance scope.
- Live worker acceptance must use the real ServerAgentWorker with scoped machine tokens (created per capability, revoked after the block); never substitute LocalAgentRuntime or precomputed outputs.
- Machine tokens: secrets stay DPAPI-encrypted outside git; receipts contain identity only.
- Loopback networking may require sandbox escalation (`require_escalated`); environment refusals are not Kraken defects.
- Working tree at plan start has uncommitted WIP (representation import kind filtering + Windows reserved-name fixes) — WP0 handles it.
- Model-quality and real-device electrical acceptance are out of scope (explicit boundary in every receipt).

## Review Focus

1. Folder paths with Windows reserved DOS device names (`aux`, `con`, …) at any depth — scan/create/import/delete must not crash or mis-delete (WIP under test; pin with regression).
2. Import source folders mixing or lacking supported kinds (CIF vs images) — plan must surface explicit `ImportIssue`s, never silently import the wrong kind.
3. Concurrent duplicate commands (same idempotency key, two clients) — exactly one effect, no orphan trees (D1 regressions must stay green under the stateful machine).
4. Token revocation or TCP loss mid-job — no partial/duplicate artifact publication; lease reclaimable (WP4 faults).
5. Server restart while a lease is held — job survives, old owner fenced (WP3 agent machine + WP4).

---

### Task 0.1: Verify and commit the uncommitted WIP as a checkpoint

**Files:**
- Modify (already in working tree): `src/kraken_manager/infrastructure/workspace_files.py`, `src/kraken_manager/application/imports.py`, `src/kraken_hub/composition/parts/create_representation.py`, `src/kraken_server/persistent_services_parts/create_representation.py`, `src/kraken_hub/composition/parts/materialize_representation_inputs.py`, `src/kraken_hub/infrastructure/remote/parts/representation.py`, `src/kraken_hub/presentation/qt/parts/add_layer_images.py`, `src/kraken_hub/presentation/qt/parts/add_representation.py`, `src/kraken_manager/application/project_application.py`, `src/kraken_manager/application/project_operations.py`, `src/kraken_manager/workspace.py`, `src/kraken_server/persistent_services_parts/materialize_representation_inputs.py`
- Revert hunk: `.gitignore` (restore the `config/local/` line)
- Test (already in working tree): `tests/test_workspace_files.py`, `tests/test_dataset_vector_inputs.py`, `tests/test_hub_embedded_service_parts/directory_import_preflighted.py`, plus the other modified test files listed in `git diff --name-only`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a committed checkpoint tree all later WPs build on; `import_source_suffixes(kind) -> frozenset[str]` in `kraken_manager.application.imports` (VECTOR → `{".cif"}`, IMAGE → `FRAME_IMAGE_SUFFIXES`).

- [ ] **Step 1: Revert the `.gitignore` hunk** — restore `config/local/` to `.gitignore` (the WIP removes it, which would make `config/local/*.secret` committable — forbidden by Global Constraints). Verify: `git check-ignore config/local/server.toml` prints the path.
- [ ] **Step 2: Review the full WIP diff** (`git diff`) against Review Focus items 1–2; confirm the new `ImportIssue` cases (`no_import_sources`, `ignored_files`) and the reserved-name `_make_directory`/`_remove_tree` behavior.
- [ ] **Step 3: Run the touched test files** — `pytest tests/test_workspace_files.py tests/test_dataset_vector_inputs.py tests/test_catalog_operation_parity.py tests/test_hub_catalog_route.py tests/test_hub_embedded_service_parts tests/test_server_workspace_inputs_parts tests/test_work_orders.py` — Expected: all PASS.
- [ ] **Step 4: Commit** — `git add -u` (never `config/local/`); message: `Fix representation import kind preflight and Windows reserved device names`.
- [ ] **Step 5: Independent review** — dispatch reviewer subagent on the committed diff (focus: reserved-name edge cases, issue non-blocking semantics, no unrelated changes).

### Task 0.2: Environment probe and continuation checkpoint document

**Files:**
- Create: `docs/quality/checkpoint-2026-11-06.md`
- Test: none (operational)

**Interfaces:**
- Produces: the checkpoint doc format all later WPs append to (section per WP, evidence links, exact next step — same format as `checkpoint-2026-11-01.md`).

- [ ] **Step 1: Probe and record** — postgres process state; `kraken-server`/gateway not running; `C:\kraken_test\{blobs,source,derived}` present; `D:\Heggs\Source` layout (list layer subfolders, count frame files per folder, confirm 18×18 = 324 per layer where applicable); `kraken-admin doctor` failure reason (default config path missing — expected, we use `--config config/local/server.toml`).
- [ ] **Step 2: Write the checkpoint** — carry over open items from `checkpoint-2026-11-01.md` (0518.cif fixture missing; legacy confidence PARTIAL; Windows-service/Hello/updater BLOCKED) and state the plan link. Commit.

---

### Task 1.1: Machine-readable registries (defects + actions)

**Files:**
- Create: `docs/quality/registry/defects.json`
- Create: `docs/quality/registry/actions.json`
- Create: `tools/quality/__init__.py`, `tools/quality/registry_schema.py`
- Test: `tests/test_quality_registry.py`

**Interfaces:**
- Produces (consumed by 1.2, 2.x, 5.x, 6.x, 9.x):
  - `defects.json`: `{"defects": [ {"id": "D7", "component": "kraken_hub", "title": str, "status": "OPEN"|"RESOLVED"|"EXCEPTION", "severity": "critical"|"major"|"minor", "regression_tests": ["tests/test_hub_resume_remote_failure.py::test_..."], "evidence": ["docs/quality/evidence/..."], "fixed_in": "f749893"|"", "repro": str, "exception_reason": str?} ]}`
  - `actions.json`: `{"actions": [ {"id": "HUB-OPEN-PROJECT", "surface": "HUB"|"MGR"|"SRV"|"AGT"|"PLG"|"GW", "entry_point": str, "preconditions": [str], "expected_behavior": str, "side_effects": [str], "error_conditions": [str], "tier": "smoke"|"unit"|"integration"|"gui"|"stateful"|"e2e"|"extended"|"regression", "status": "PASS"|"FAIL"|"PARTIAL"|"BLOCKED"|"NOT_TESTED"|"NOT_APPLICABLE", "tests": [test node ids], "evidence": [paths], "mandatory": bool, "exception_reason": str?} ]}`
  - `validate_registry(path) -> list[str]` in `tools/quality/registry_schema.py` (empty list = valid; no new dependencies — hand-rolled checks).
- Consumes: existing `docs/quality/defects.md` (D1–D8) and `docs/quality/functional-inventory.md` (all `HUB-*`, `MGR-*`, `SRV-*`, `AGT-*`, `PLG-*`, `GW-*` IDs) as the seed data — every existing ID must appear in the JSON, statuses copied from the 2026-11-01 matrix section.

- [ ] **Step 1: Write the failing enforcement tests** in `tests/test_quality_registry.py`:
  - `test_defects_json_valid`: parses, schema-valid, IDs unique.
  - `test_resolved_defects_have_live_regression_tests`: for each RESOLVED defect, every id in `regression_tests` matches a real collected test node id (collect via `pytest --collect-only -q` cached list, or parse test files with `ast`), AND at least one exists.
  - `test_defect_markers_are_registered`: every test node id appearing in any `regression_tests` belongs to a registered defect; scan test files for `kraken_defect = "D<n>"` module constants and require they map to registered defects (marker convention: regression tests carry `kraken_defect = "D7"` at module level).
  - `test_mandatory_actions_tracked`: every action with `mandatory: true` has status NOT in {NOT_TESTED} unless `exception_reason` set; every `tests` entry is a real node id.
- [ ] **Step 2: Run — expect FAIL** (files missing).
- [ ] **Step 3: Write `registry_schema.py`** (validators only) and **seed both JSON files** from the markdown sources. D1–D8 all RESOLVED with their existing regression test node ids (read from `defects.md` text, e.g. `tests/test_project_create_idempotency.py`). Actions: transcribe every ID from `functional-inventory.md` with the 2026-11-01 statuses from `acceptance-matrix.md`; KateGB actions get `mandatory: false`.
- [ ] **Step 4: Add `kraken_defect` markers** to the existing regression test files for D1–D8 (one line each, e.g. `kraken_defect = "D1"` in `tests/test_project_create_idempotency.py`).
- [ ] **Step 5: Run — expect PASS.** Commit: `Add machine-readable defect and action registries with enforcement tests`.

### Task 1.2: Render markdown from registries (drift prevention)

**Files:**
- Create: `tools/quality/generate_quality_docs.py`
- Modify: `docs/quality/defects.md` (status/evidence lines), `docs/quality/functional-inventory.md` (status table), `docs/quality/acceptance-matrix.md` (current-section table)
- Test: `tests/test_quality_registry.py` (extend)

**Interfaces:**
- Consumes: 1.1 JSON + `validate_registry`.
- Produces: `render_defects_md(registry) -> str`, `render_actions_md(registry) -> str`, `render_matrix_md(registry) -> str`; CLI `python tools/quality/generate_quality_docs.py --check` (exit 1 if markdown out of sync) and `--write`.

- [ ] **Step 1: Failing test** `test_markdown_in_sync_with_json`: run generator in `--check` mode; assert exit 0 after `--write`.
- [ ] **Step 2: Implement generator** — each rendered file keeps a `<!-- generated-from registry/*.json — do not edit between markers -->` fenced section; prose outside markers is preserved.
- [ ] **Step 3: Run `--write`, review the diff is limited to generated sections, commit** `Generate quality markdown from registries`.

---

### Task 2.1: Coverage report with explicit denominators

**Files:**
- Create: `tools/quality/coverage_report.py`
- Test: `tests/test_quality_coverage_report.py`
- Create (evidence): `docs/quality/evidence/<date>/coverage-report.json` + `.md`

**Interfaces:**
- Consumes: `docs/quality/registry/*.json`, JUnit XML files passed as args, optional `coverage.xml`/`.json` from pytest-cov.
- Produces: `build_report(registry_dir, junit_paths, coverage_data=None) -> dict` with keys `ui_actions`, `api_operations`, `error_paths`, `state_transitions`, `tiers`, `defect_regressions`, `source` — each metric `{covered: int, total: int, status_breakdown: {PASS: n, FAIL: n, PARTIAL: n, BLOCKED: n, NOT_TESTED: n, NOT_APPLICABLE: n}, uncovered: [ids]}`; CLI writes JSON+MD to `docs/quality/evidence/<date>/`.
- Denominators (explicit, in the report): UI actions = count of `mandatory: true` non-SRV actions; API operations = `surface: SRV` actions (each maps to route group + method from `functional-inventory.md`); error paths = tests carrying `pytest.mark.error_path` (fault/negative tests, incl. WP4 fault suites) — denominator is the set of error conditions listed per action in `actions.json` field `error_conditions: [str]`; state transitions = (state, action) pairs recorded by the WP3 machines in their evidence files; tiers = the 8 tier names; defect regressions = count of RESOLVED defects; source = lines+branches in changed files (`git diff --name-only <base>`) measured by pytest-cov.
- Rules: a test marked skipped in JUnit maps to NOT_TESTED for its action, never PASS. An action is PASS only if all its listed tests passed. Metrics without a denominator source yet (e.g. state transitions before WP3 runs) report `total: 0` and are labeled NOT TESTED, never omitted.

- [ ] **Step 1: Failing tests** in `tests/test_quality_coverage_report.py` using a fixture registry (3 actions: one PASS with 2 passing tests, one with 1 skipped test, one NOT_TESTED) + fixture JUnit: assert denominators, assert skipped→NOT_TESTED, assert uncovered list is non-empty and listed, assert NOT_APPLICABLE actions excluded from denominator.
- [ ] **Step 2: Implement CLI + report rendering.**
- [ ] **Step 3: Run the report against the real registries + latest evidence JUnit files; commit report + code** `Add coverage report with explicit denominators`.

### Task 2.2: Wire coverage + registry checks into a local quality-gate runner

**Files:**
- Create: `scripts/quality-gates.ps1` (parameters: `-Tier smoke,unit,integration,gui,stateful,e2e,extended,regression`, `-Block` names)
- Test: none (operational; itself exercised by WP4 first live run)

**Interfaces:**
- Consumes: `docs/quality/reproduce-stabilization.ps1` blocks; `tests/test_quality_registry.py`; `tools/quality/coverage_report.py`; pytest.
- Produces: one command per tier, each writing receipt+JUnit under `docs/quality/evidence/<date>/gate-<tier>/`; a gate FAILS (non-zero exit) if: registry enforcement test fails, a mandatory suite was skipped, or the coverage report shows a regression in a previously PASS mandatory action.
- Tier→suite map (initial, refine as WPs land): smoke = `test_desktop_runtime_smoke.py`; unit = fast non-network tests (default pytest run excluding markers `live`, `pg`); integration = `pg`-marked + storage contract suite; gui = pytest-qt offscreen; stateful = `tests/stateful/`; e2e = `test_end_to_end_functional.py` + live blocks; regression = all tests carrying `kraken_defect` markers; extended = explorer campaigns (WP6).

- [ ] **Step 1: Implement the runner** reusing `reproduce-stabilization.ps1` conventions (uuid run dir, receipt with command+exit code, JUnit).
- [ ] **Step 2: Run `-Tier smoke,unit,regression` on the committed tree; fix runner bugs (not test bugs); record receipt.** Commit `Add local quality gate runner`.

---

### Task 3.1: Add hypothesis; project/layer/representation stateful machine

**Files:**
- Modify: `pyproject.toml` (dev extra: `hypothesis>=6`)
- Create: `tests/stateful/__init__.py`, `tests/stateful/project_machine.py`
- Test: the machine file itself (pytest collects `Test*Machine` classes)

**Interfaces:**
- Consumes: real `kraken_manager` `EmbeddedProjectService` (local temp workspace, same construction as `tests/test_hub_embedded_service.py`).
- Produces: `ProjectMachine(RuleBasedStateMachine)` with rules: `create_project` (reuses idempotency keys from a small pool to force repeats), `create_layer`, `create_representation_image`, `create_representation_vector`, `import_frame` (real PNG bytes into a temp source dir), `rename_project/layer/representation`, `archive/restore` (project, layer, representation), `delete_layer`, `delete_project`, `note/quality_defect` (QualityDefect entity: create, assign, fix, verify, accept, reject transitions), `issue_work_order` + `start_workflow`/`pause_workflow`/`cancel_workflow` (Workflow/WorkItem entities, local work orders), `reopen_service` (new service instance over same workspace — persistence rule). Invariants checked after every step: (a) list state == journal-derived state; (b) archived items absent from active lists; (c) re-opened service equals in-memory state; (d) deleting unknown entity raises the domain error, state unchanged; (e) no orphan directories in workspace root; (f) artifact version history is append-only — versions never rewritten, lineage pointers valid.
- Settings: `@settings(max_examples=30, deadline=None, database=ExampleDatabase(path=".test_runtime/hypothesis/project_machine"))`; failing sequences saved and minimized by hypothesis.

- [ ] **Step 1: `uv add --dev hypothesis` (or add to dev extra), sync.**
- [ ] **Step 2: Write the machine** (rules + invariants above; invariants implemented as `@invariant` methods).
- [ ] **Step 3: Run `pytest tests/stateful/project_machine.py -v`.** If an invariant fails: decide product-bug vs machine-bug. Product bug → full defect workflow (register D9+, failing regression test, fix RED→GREEN, mark). Machine bug → fix the rule's precondition.
- [ ] **Step 4: Record transition coverage** (states×actions exercised) to the evidence dir; commit `Add stateful project machine (hypothesis)`.

### Task 3.2: Agent job/lease stateful machine

**Files:**
- Create: `tests/stateful/agent_job_machine.py`

**Interfaces:**
- Consumes: `LocalAgentRuntime` (in-memory store) with a real echo-style plugin subprocess (existing test plugin fixtures in `tests/test_agent_*.py` — reuse, do not mock the runtime).
- Produces: `AgentJobMachine(RuleBasedStateMachine)`: rules `submit_job`, `lease_job`, `heartbeat`, `complete_job`, `cancel_job`, `retry_failed`, `expire_leases` (advance controlled clock), `duplicate_complete`, `second_worker_lease`. Invariants: job state machine transitions legal; lease fencing — old owner after expiry gets 403-class failure; no duplicate artifact versions on repeated complete; cancel idempotent; queued→leased→terminal only.
- [ ] **Steps:** mirror 3.1 (write machine → run → defect workflow for real bugs → record transitions → commit `Add stateful agent job machine (hypothesis)`).

### Task 3.3: Server ACL/session stateful machine (PostgreSQL)

**Files:**
- Create: `tests/stateful/server_acl_machine.py`

**Interfaces:**
- Consumes: isolated PostgreSQL cluster fixture (same pattern as `tests/test_server_administration.py` / reproduce script PG blocks — own cluster + UUID db, no shared secrets).
- Produces: `ServerAclMachine(RuleBasedStateMachine)`: rules `register_account`, `grant_role`, `revoke_role`, `create_project`, `read_project`, `write_project` (rename), `wrong_if_match`, `new_session`, `disable_account`, `restart_server` (ServerConnection entity: kill + relaunch the server process mid-sequence; in-flight session tokens stay valid from DB, queued state survives). Invariants: viewer read-only (write → 403); ACL changes CAS by revision (stale → 409); principals isolated across projects; disabled account → 401; after restart: all previously created projects/ACLs readable, no job/queue loss.
- [ ] **Steps:** mirror 3.1 → commit `Add stateful server ACL machine (hypothesis)`.

---

### Task 4.1: Bring up the live server + gateway; preflight receipt

**Files:**
- Create: `docs/quality/probes/bringup_live_server.ps1`
- Test: none (operational)

**Interfaces:**
- Consumes: `config/local/server.toml` (secrets DPAPI-encrypted; blob roots `C:\kraken_test\*`); `kraken-blob-gateway.exe` at `src/kraken_admin/blob_gateway/target/release/`.
- Produces: running server `127.0.0.1:8080` + gateway `8081` (server.toml currently has `blob_gateway.enabled = false` — the probe starts the gateway via the standard lifecycle separately, as in the 2026-11-01 checkpoint; do NOT edit the user's server.toml), receipt `docs/quality/evidence/<date>/live-bringup.json` (health statuses, versions, protected project `test` hash baseline: revision, history length, both jobs' payload hashes — read-only).
- [ ] **Step 1: Write the probe** (start gateway, start `kraken-server --config config/local/server.toml`, poll `/health` both, authenticate via saved session as in `docs/quality/probes/*.py`, snapshot protected project state, write receipt; teardown optional `-KeepRunning`).
- [ ] **Step 2: Run with `-KeepRunning`; verify receipt; commit probe + receipt** `Add live server bring-up probe with protected-state baseline`.

### Task 4.2: Scoped machine tokens + isolated quality project from Heggs source

**Files:**
- Create: `tests/test_live_heggs_import.py` (env-gated: runs only with `KRAKEN_LIVE_HEGGS_CONFIG` pointing at a JSON with server_url, session_secret path, machine token path, layer name — secrets never in the JSON values)
- Modify: `docs/quality/reproduce-stabilization.ps1` (new blocks `live-heggs-import`, `live-heggs-worker`, `live-heggs-faults`, `live-heggs-gui`)
- Test: the new test file

**Interfaces:**
- Consumes: bring-up probe state; `D:\Heggs\Source\<layer>` (read-only); existing live-test driver patterns (search `KRAKEN_LIVE_WORKER_CONFIG` in `tests/` and follow them exactly).
- Produces: scoped token(s) (first: `frames.vectorize.v1`), created via admin API, identity-only in receipt; isolated quality project `heggs-<uuid>` (18×18) on the live server; image representation imported from ONE Heggs layer folder via the genuine upload path (server preflight → gateway tickets → upload); oracle: read back every artifact via HTTP/gateway, SHA-256 + size match source bytes, PNGs decode, frame (x,y) mapping correct for a sampled ≥9 cells incl. corners and center.

- [ ] **Step 1: Write the import scenario** (TDD not possible against live env — write scenario with full assertions first, run RED against no-project baseline, then run for real).
- [ ] **Step 2: Execute; save receipt/JUnit; verify protected project baseline unchanged (hashes from 4.1).**
- [ ] **Step 3: Commit test + receipt** `Verify Heggs 18x18 layer import through live server and gateway`.

### Task 4.3: Contour + NeuralImage + Karakal + stitching via real ServerAgentWorker

**Files:**
- Create: `tests/test_live_heggs_worker.py`
- Test: the new test file

**Interfaces:**
- Consumes: 4.2 project; scoped tokens per capability (`frames.vectorize.v1`, `frames.binary-segment.v2`, karakal analysis, `layer.stitch.v1`); NeuralImage model candidate `.test_runtime/pytest-e2e/test_single_project_twenty_sta0/export/479fcf1a-b71d-5725-9755-6acaaad6c5ae-S 660k_shift_128_epoch1.pth` — **verify its hash/receipt and safe loading before use** (per checkpoint); vectorize a bounded sub-rectangle (e.g. 2×2 cells) to keep runtime sane — full 324-cell vectorization only if runtime < 30 min.
- Produces: per plugin: job submitted via server queue → targeted lease → real subprocess → artifact publication → version history; oracles: CIFs parse with correct dimensions and input lineage to source versions; masks dimensions match frames; stitching artifact present; then two independent HTTP clients + two real Qt Hub windows open the project and re-read results/history. Receipts per step.
- [ ] **Step 1: Write scenarios with full assertions; run RED where a capability is missing.**
- [ ] **Step 2: Execute sequentially (contour → neuralimage → karakal → stitch); any product defect → full defect workflow (register, regression, RED→GREEN, re-run).**
- [ ] **Step 3: Commit** `Verify Heggs plugins through live ServerAgentWorker`.

### Task 4.4: Fault tests + user GUI run + token revocation + matrix update

**Files:**
- Create: `tests/test_live_heggs_faults.py`, `tests/test_live_heggs_gui.py`
- Test: both

**Interfaces:**
- Consumes: 4.3 setup; fault patterns from the existing `live-worker` fault block (worker unavailable, wrong token 401, stale cancel 409, real Hub cancel, plugin failure with correct blob hash, lease expiry/reclaim/fencing, duplicate complete 403, TCP disconnect after committed complete).
- Produces: same fault matrix extended to the new capabilities; user-GUI run: real `kraken-hub` process, login via saved account, open quality project, MyWork shows worker results, accept flow; screenshots + state JSON to evidence; then revoke ALL machine tokens (verify 401), final protected-baseline comparison, update `actions.json`/matrix: `REAL-PROJECT-SERVER-ACCEPTANCE` → PASS with boundary note (model quality / device acceptance excluded), checkpoint section.
- [ ] **Steps: write → run → defect workflow as needed → revoke tokens → update registries (generator `--write`) → run registry+coverage gates → commit** `Complete live Heggs acceptance with faults, GUI run and token revocation`.

---

### Task 5.1: Reusable Qt GUI harness

**Files:**
- Create: `tests/helpers/qt_harness.py`
- Test: `tests/test_qt_harness.py` (self-test: launches real Hub offscreen, drives one known action, asserts observable state)

**Interfaces:**
- Produces (consumed by 5.2, 6.x): fixtures/helpers — `launch_hub(tmp_workspace)` (real `kraken-hub` composition offscreen, saved local account, temp roots), `wait_until(predicate, timeout)` (observable completion: status bar text / journal event / network idle — no arbitrary sleeps), `capture_failure(window)` (screenshot PNG + widget-tree JSON + log tail, written to a provided evidence dir), `click_menu(path)`, `dialog_driver` (find modal, assert buttons, click, verify).
- [ ] **Steps: implement → self-test passes with a real click + state assertion (not just no-exception) → commit `Add reusable Qt GUI harness`.**

### Task 5.2: Inventory-driven GUI campaign (local-surface actions)

**Files:**
- Create: `tests/test_gui_action_inventory.py`
- Test: itself

**Interfaces:**
- Consumes: `actions.json` (surface HUB/MGR, local-capable actions), `qt_harness`.
- Produces: parametrized tests, one per action: precondition setup → entry point via real UI → assert observable outcome + persistence after service reopen. Actions whose preconditions need the live server are marked `pytest.mark.live_gui` (executed in WP4/4.4, not here). BLOCKED/EXCEPTION actions: `pytest.skip(reason=exception_reason)` — the coverage report maps these to NOT_TESTED, keeping them visible. Update `actions.json` statuses from real results.
- [ ] **Step 1: Parametrization loader** (read JSON at collection; unknown IDs fail the build — ties GUI suite to the registry).
- [ ] **Step 2: Implement scenarios for all local-capable HUB actions; run; defects → workflow.**
- [ ] **Step 3: Update registries + coverage report; commit** `Add inventory-driven GUI acceptance campaign`.

---

### Task 6.1: State-aware explorer with checkpoints

**Files:**
- Create: `tools/explorer/explorer.py`, `tools/explorer/invariants.py`
- Test: `tests/test_explorer.py` (unit: strategy determinism per seed, checkpoint resume, watchdog triggers)

**Interfaces:**
- Consumes: `actions.json` (preconditions define legal actions per state), `qt_harness` (offscreen), invariant functions (journal consistency, no crash, state-model expectations from WP3).
- Produces: `run_campaign(seed, budget_steps, evidence_dir)`: (1) discover legal actions from current state; (2) pick by reproducible seeded strategy biased toward untested (state, action) pairs; (3) execute via real UI; (4) observe + validate invariants; (5) record transition to `checkpoints/campaign-<seed>.json` (state, visited transitions, queue) after every step so interrupted campaigns resume; (6) on failure: shrink to minimal action sequence (replay from checkpoint), write reproduction + evidence, register candidate defect. Watchdog: per-action timeout, total budget, app crash → relaunch + continue from checkpoint. Destructive actions (delete project/layer) only against temp workspaces the explorer created.
- [ ] **Steps: unit tests first (determinism, resume, watchdog) → implement → run 2 campaigns (seeds 1 and 2, ≥150 steps each) → triage findings (defect workflow for confirmed) → commit** `Add state-aware explorer with checkpoint resume`.

---

### Task 7.1: Mutation testing on critical business logic (selective)

**Files:**
- Modify: `pyproject.toml` (dev extra: `mutmut`)
- Create: `scripts/mutation-check.ps1` (scoped `mutmut run` on `kraken_manager/application/imports.py`, lease/CAS logic in `kraken_server`, ACL policy; `mutmut results`; report to evidence)
- Test: none directly — surviving meaningful mutants become new regression tests (defect workflow, severity=minor, tagged `mutation`).

**Interfaces:**
- Consumes: existing test suites as the kill set.
- Produces: mutation report `docs/quality/evidence/<date>/mutation-<module>.md` (killed/survived/suspicious per module) + added tests for surviving mutants that represent real behavior (skip stylistic mutants — record the skip reason).
- [ ] **Steps: add dep → run scoped → triage (cap: fix only mutants changing observable behavior) → add tests → re-run until no behavior-level survivors in scope → commit** `Add scoped mutation testing for import, lease and ACL logic`.

### Task 7.2: Gateway fault/contract extension (from test-gaps)

**Files:**
- Create: `tests/test_blob_gateway_contract_extended.py` (or extend `tests/test_blob_gateway.py`)

**Interfaces:**
- Consumes: real `kraken-blob-gateway.exe` process fixture (existing pattern in `tests/test_blob_gateway.py`).
- Produces: coverage of the named gaps: Range requests; corrupted-object 409; deletion barrier; backup pause; server-issued ticket lifecycle (ticket garbage → 401, expiry). Independent oracle: byte-exact reads incl. ranges reassembled == original SHA-256.
- [ ] **Steps: TDD each gap → commit** `Extend Blob Gateway contract tests (range, 409, barrier, pause, tickets)`.

### Task 7.3: Weak-assertion audit of the core suite (selective)

**Files:**
- Create: `tools/quality/assertion_audit.py`
- Create: `docs/quality/evidence/<date>/assertion-audit.md`
- Test: `tests/test_assertion_audit.py`

**Interfaces:**
- Consumes: the fast (non-live) test suite source via `ast` parsing.
- Produces: a report listing tests with weak-or-missing assertions — (a) no `assert`/`pytest.raises` at all, (b) only return-code/exit-code checks, (c) only file-existence checks, (d) body under N lines with no state re-read (persistence) — scoped to `tests/test_hub_embedded_service*.py`, `tests/test_server_*.py`, `tests/test_agent_*.py`, `tests/stateful/`. Each finding gets a disposition: `strengthened` (test fixed to assert observable state), `justified` (reason recorded, e.g. property-based machine where invariants assert), or `accepted-risk` (owner note). Findings that reveal a test passing without exercising its intended action are defects of the test suite → fix under the defect workflow (registered, with the strengthened test as the regression).
- [ ] **Steps: implement auditor with the 4 detectors (unit-tested on synthetic fixtures) → run on scope → triage with cap of the 20 most impactful findings → strengthen/justify → re-run auditor to show delta → commit** `Add weak-assertion audit and strengthen flagged core tests`.

---

### Task 8.1: Final acceptance report and independent review

**Files:**
- Create: `docs/quality/acceptance-report-2026-11-06.md`
- Modify: `docs/quality/checkpoint-2026-11-06.md` (final state)

**Interfaces:**
- Consumes: all registries, coverage reports, evidence receipts, defect workflow records.
- Produces: report against the 15 mission acceptance criteria, each: met/partial/blocked + evidence links; explicit remaining limitations (0518.cif fixture, legacy confidence metrics, Windows service/Hello/updater BLOCKED, no hosted CI — gates are local, model-quality/device acceptance excluded by boundary); defect register final state; coverage summary with denominators.
- [ ] **Step 1: Run the full local gate (`scripts/quality-gates.ps1 -Tier` all runnable tiers); record.**
- [ ] **Step 2: Write the report (no claim without a receipt link).**
- [ ] **Step 3: Independent review** — reviewer subagent over (a) all app-code changes since plan start, (b) the framework code (registries, generator, report, harness, explorer). Findings → fix → re-verify.
- [ ] **Step 4: Commit** `Final acceptance report 2026-11-06`.

---

## Sequencing and dependencies

WP0 (0.1→0.2) → WP1 (1.1→1.2) → WP2 (2.1→2.2) → WP3 (3.1→3.2→3.3, parallelizable after 3.1) ∥ WP4 (4.1→4.2→4.3→4.4, needs the user's machine/server) → WP5 (5.1→5.2) → WP6 (6.1) → WP7 (7.1, 7.2, 7.3 parallel) → WP8.
WP4 can start right after WP0 if live-environment time is the bottleneck — its scenarios must still land as repo tests with receipts.
Every WP ends with: registries updated (generator `--write`), evidence committed, checkpoint section appended, gate run.

## Explicit non-goals (this plan)

- No production architecture changes to accommodate tests (test harness lives in `tests/`, `tools/quality`, `scripts/`).
- No hosted-CI provider setup (repo remote is a local bundle); `quality-gates.ps1` is the gate. GHA/GitLab files only if the user later connects a host.
- No KateGB acceptance. No model-quality or real-device electrical acceptance (boundary noted in every receipt).
- No rewrite of existing markdown history — generated sections only.
