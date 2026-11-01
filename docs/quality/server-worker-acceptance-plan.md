# Server worker acceptance plan

> **For agentic workers:** Use superpowers:executing-plans. Continue from `f1c43f8`; preserve prior acceptance evidence.

**Goal:** Prove the complete Server → ServerAgentWorker → plugin → version → Hub path, including failures and reread.

**Architecture:** Use the configured PostgreSQL Server and Gateway through authenticated HTTP. Write only to uniquely named quality projects. Use a dedicated revocable machine credential, real subprocess plugins and real Qt widgets. Run destructive queue fault injection on the existing private PostgreSQL test cluster.

**Tech Stack:** pytest, PyQt6/QTest, PostgreSQL 16, Kraken ServerAgentWorker and source plugins.

**Spec:** User attachment 6b5ae193-851d-4c91-a1e1-b905997a0006, “Kraken — продолжение стабилизации и сквозная приёмка серверного окружения”.

## Constraints

- Never mutate project `test`, its artifacts, or either existing job, especially waiting_for_user.
- KateGB excluded. No service installation, Hello, updater or external TLS changes.
- Never log/commit secrets. Machine credential stored with DPAPI; receipt contains identity/capabilities only.
- Machine capabilities are global by operation; they are not project restrictions. Do not poll arbitrary user work.
- Reuse previous evidence for unchanged expensive computations; new failures require regression RED → fix → GREEN and full scenario retry.

## Packages

- [x] Read checkpoint, matrix, defects, plan, gaps, review and reproducer; inspect HEAD/config and штатный machine credential lifecycle.
- [ ] Dedicated credential: prepare create/revoke, obtain administrative approval, verify authentication and denied capability. Revoke when block finishes.
- [ ] Minimal Contour path: separate server project, layer, small PNG grid, job, actual leased worker, CIF publication/version/hash/type/input lineage, Hub open/history/reopen.
- [ ] NeuralImage/Karakal/stitch: execute supported production paths using sufficient inputs, reuse expensive trained fixtures where valid.
- [ ] Faults: unavailable worker, bad input/plugin failure, interrupted connection, lease expiry/reissue, duplicate completion, cancel, denied access and CAS. Use test project/isolated cluster only.
- [ ] Required regression suites, persisted command/exit/JUnit/code/config/requirement receipts, matrix/checkpoint and independent final review.
- [ ] Remaining relevant gaps: exact Contour reference fixture if present; keep legacy confidence counter distinct from real metrics.

## Review focus

- A machine credential must never lease either pre-existing user job.
- A successful transport receipt must correspond to an actually readable output version.
- Duplicate completions/reclaims must not publish duplicate versions.
- HTTP failure in asynchronous Qt callbacks must remain diagnostic without aborting the process.
- A new client session must read the same inputs, job state, output bytes and history.

## Execution ledger

- BASE f1c43f8, clean checkout. Administrative token script prepared at `.test_runtime/quality_machine_credential.py`; no token issued. Approval pending for frames.vectorize.v1 only. Previous real-project read checks and schema0011 evidence retained.

- Block393bca4: D8 persistent upload preflight RED→GREEN, neighboring19PASS; actual Contour machine-worker path through Gateway/version/two Hub sessions PASS. Dedicated frames.vectorize token issued with explicit approval; revoke after fault block. Next: fault paths, other plugins and GUI job launch/history; stage remains PARTIAL.

- be1ca58: review Important(global lease race) RED2→GREEN14; targeted route and exact per-coordinate lineage accepted by independent reviewer. 95ec8f2: final live Contour/fault block6PASS, token revoked and HTTP401 verified. Next: NeuralImage genuine inference using existing trained E2E model; Karakal real Hub launch; stitch; final independent stage review. Stage PARTIAL.
