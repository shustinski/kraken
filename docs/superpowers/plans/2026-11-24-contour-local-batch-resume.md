# Contour Local Batch Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pause a managed Contour batch, reopen it locally with prior settings and completed CIFs, and continue only unfinished frames.

**Architecture:** Keep recovery data beside the existing desktop job attempts. The Agent copies verified output files into a fresh lease attempt; Contour persists a per-job settings and provenance record and uses the existing batch processor on the unfinished subset. Use the current result manifest to return complete or partial work.

**Tech Stack:** Python 3.14, PyQt6, ProcessPoolExecutor, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-24-contour-local-batch-resume-design.md`

## Global Constraints

- Local computer only; preserve original staging attempts.
- Existing 4,846 CIFs have unknown settings. Do not invent them.
- Changes to settings while paused require a warning and frame-level provenance.
- Reopened jobs remain paused until the user presses Continue.
- No new dependency and no behavior change for other plugins.

## Task 1: Durable local recovery

- [ ] Add tests for reclaiming a Contour job with matching manifest and existing CIFs, rejecting mismatched or unsafe files, and leaving other plugins' attempts isolated.
- [ ] Add a small Contour-specific recovery function called by `ServerAgentWorker._run_leased_job` after staging the new manifest. Copy verified CIFs from a matching previous attempt into the new attempt atomically; retain original files.
- [ ] Pass the prior settings/provenance record into the new Contour attempt, including the explicit unknown-settings case for legacy output.
- [ ] Run focused Agent recovery tests.

## Task 2: Contour pause and continuation

- [ ] Add tests for restored settings, visible existing CIFs, paused state, only-missing-frame continuation, and warning on changed settings.
- [ ] Add Pause and Continue UI actions in the existing run controls. Persist a settings snapshot before workers start, and load it when reopening a managed job.
- [ ] Use the current batch controller with a filtered image list, preserving progress relative to all job frames. Leave standalone processing unchanged.
- [ ] Run focused Contour UI and batch tests.

## Task 3: Bounded stop and output integrity

- [ ] Add tests for a worker that ignores cancellation and an interrupted CIF write.
- [ ] Stop submitting chunks on Pause, allow ten seconds for current workers, then terminate remaining workers without blocking the UI. Save CIFs through a temporary file and atomic replace.
- [ ] Validate resumed CIFs and recompute malformed outputs.
- [ ] Run batch worker and stop tests.

## Task 4: Result provenance and final checks

- [ ] Add tests that returned results record parameter snapshots and frame groups when settings changed, including unknown provenance for legacy files.
- [ ] Preserve the current single-settings result payload when nothing changed; extend only the `applied_parameters` mapping for mixed runs.
- [ ] Run relevant tests, Ruff checks and formatting; review the diff and run `graphify update .`.

## Review Focus

- New lease after a crash while old output is still being copied: the old attempt stays intact and no partial copy is counted.
- A source frame list or digest changes: old CIFs are not reused.
- User presses Pause twice or closes while pause is pending: worker shutdown remains bounded.
- An output has a matching name but invalid CIF contents: it is recomputed, not imported.
- Mixed settings metadata is truthful for every returned frame.
