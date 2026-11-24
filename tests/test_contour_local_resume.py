"""Recovery of Contour outputs across desktop lease attempts."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from kraken_agent.jobs import StagingWorkspace
from kraken_agent.contour_recovery import recover_contour_attempt
from tests.test_agent_execution_safety import manifest


def _attempt(root, claim, job):
    workspace = StagingWorkspace(root / claim, job.job_id)
    workspace.create()
    workspace.write_manifest(job)
    return workspace


def test_contour_attempt_recovers_matching_completed_cif_without_changing_old_file(tmp_path):
    job = manifest()
    root = tmp_path / "staging"
    old = _attempt(root, "old", job)
    cif = b"DS 1 1 1;\nL NM;\n( R 1.png );\n( S 2 2 );\nDF;\nE\n"
    (old.path / "outputs/1.cif").write_bytes(cif)
    (old.path / "contour-resume.json").write_text('{"version": 1, "settings": {}}', encoding="utf-8")
    new = _attempt(root, "new", job)

    assert recover_contour_attempt(root, new, job) == 1
    assert (new.path / "outputs/1.cif").read_bytes() == cif
    assert (new.path / "contour-resume.json").is_file()
    assert (old.path / "outputs/1.cif").read_bytes() == cif


def test_contour_attempt_ignores_changed_manifest_and_incomplete_output(tmp_path):
    job = manifest()
    root = tmp_path / "staging"
    old = _attempt(root, "old", job)
    (old.path / "outputs/1.cif").write_bytes(b"unfinished")
    new = _attempt(root, "new", job)
    assert recover_contour_attempt(root, new, job) == 0
    assert not (new.path / "outputs/1.cif").exists()

    changed = replace(job, inputs=(replace(job.inputs[0], sha256=hashlib.sha256(b"other").hexdigest()),))
    newer = _attempt(root, "newer", changed)
    assert recover_contour_attempt(root, newer, changed) == 0


def test_other_plugin_does_not_reuse_contour_outputs(tmp_path):
    job = replace(manifest(), operation="other.v1")
    root = tmp_path / "staging"
    old = _attempt(root, "old", job)
    (old.path / "outputs/1.cif").write_text("DS 1 1 1;\n( S 2 2 );\nDF;\nE\n")
    new = _attempt(root, "new", job)
    assert recover_contour_attempt(root, new, job) == 0
    assert not (new.path / "outputs/1.cif").exists()


def test_legacy_outputs_without_state_get_explicit_unknown_provenance(tmp_path):
    import json

    job = manifest()
    root = tmp_path / "staging"
    old = _attempt(root, "old", job)
    (old.path / "outputs/1.cif").write_bytes(b"DS 1 1 1;\n( S 2 2 );\nDF;\nE\n")
    new = _attempt(root, "new", job)

    assert recover_contour_attempt(root, new, job) == 1
    state = json.loads((new.path / "contour-resume.json").read_text(encoding="utf-8"))
    assert state["version"] == 1
    assert state["snapshots"] == []
    assert state["unknown_outputs"] == ["1"]
    assert not (old.path / "contour-resume.json").exists()


def test_existing_output_and_unsafe_entries_are_not_replaced(tmp_path):
    job = manifest()
    root = tmp_path / "staging"
    old = _attempt(root, "old", job)
    (old.path / "outputs/1.cif").write_bytes(b"DS 1 1 1;\n( S 2 2 );\nDF;\nE\n")
    new = _attempt(root, "new", job)
    (new.path / "outputs/1.cif").write_bytes(b"current")
    assert recover_contour_attempt(root, new, job) == 0
    assert (new.path / "outputs/1.cif").read_bytes() == b"current"

    other = _attempt(tmp_path / "staging2", "old", job)
    (other.path / "outputs/1.cif").mkdir()
    fresh = _attempt(tmp_path / "staging2", "new", job)
    assert recover_contour_attempt(tmp_path / "staging2", fresh, job) == 0
    assert not (fresh.path / "outputs/1.cif").exists()


def test_windows_crlf_cif_is_recovered(tmp_path):
    job = manifest()
    root = tmp_path / "staging"
    old = _attempt(root, "old", job)
    (old.path / "outputs/1.cif").write_bytes(b"DS 1 1 1;\r\nL NM;\r\n( S 2 2 );\r\nDF;\r\nE\r\n")
    new = _attempt(root, "new", job)
    assert recover_contour_attempt(root, new, job) == 1


def test_recovery_errors_do_not_fail_the_job_and_headless_jobs_skip_recovery(tmp_path, monkeypatch):
    from kraken_agent import contour_recovery

    job = manifest()
    root = tmp_path / "staging"
    old = _attempt(root, "old", job)
    (old.path / "outputs/1.cif").write_bytes(b"DS 1 1 1;\n( S 2 2 );\nDF;\nE\n")
    new = _attempt(root, "new", job)

    def failing(*_args):
        raise PermissionError("locked by antivirus")

    monkeypatch.setattr(contour_recovery, "_copy_regular", failing)
    assert contour_recovery.recover_contour_attempt_safely(root, new, job) == 0
    assert (old.path / "outputs/1.cif").is_file()

    monkeypatch.undo()
    headless = replace(job, parameters={"execution_mode": "headless"})
    newer = _attempt(root, "newer", headless)
    assert contour_recovery.recover_contour_attempt_safely(root, newer, headless) == 0
    assert not (newer.path / "outputs/1.cif").exists()
