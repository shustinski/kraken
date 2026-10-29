"""Real storage behavior behind the nonempty deployment self-test."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

from kraken_hub.presentation.probes.deployment_matrix_probe import prepare_project, probe_records, verify_project
from kraken_manager.domain.project import GridOrientation
from tests.test_work_orders import _service


def test_diagnostic_import_head_and_new_service_readback(tmp_path):
    records = probe_records(tmp_path)
    service = _service(tmp_path)
    session = service.create_initial_account("deployment", "Deployment", "")
    project = service.create_project(principal=session.principal, name="Deployment", width=2, height=2,
        orientation=GridOrientation.Y_DOWN, layer_template=True, idempotency_key="project")
    receipt = prepare_project(service, session, project, records)
    assert len(receipt["expected"]) == 2
    assert receipt["previous_vector_version_ids"][0] != receipt["expected"][0]["vector_version_id"]
    verify_project(_service(tmp_path), receipt)
    view = service.matrix_viewport(project.id, layer_id=receipt["layer_id"],
        representation_ids=(receipt["image_representation_id"], receipt["vector_representation_id"]),
        x1=1, y1=1, x2=2, y2=2)
    assert len(view["cells"]) == 4
    assert sum(bool(c.get("missing")) for c in view["cells"]) == 2


def test_reference_probe_reads_only_bounded_records_and_checks_sha(tmp_path):
    original = probe_records(tmp_path)[0]
    manifest = tmp_path / "pairs.jsonl"
    record = {k: v for k, v in original.items() if k not in {"x", "y", "diagnostic_color"}}
    manifest.write_text((json.dumps(record)+"\n")*16 + "not a record beyond diagnostic limit\n", encoding="utf-8")
    records = probe_records(tmp_path, manifest)
    assert len(records) == 16 and (records[-1]["x"], records[-1]["y"]) == (4, 4)
    assert Path(records[0]["image"]).read_bytes() == Path(original["image"]).read_bytes()
    record["image_sha256"] = "0"*64
    manifest.write_text(json.dumps(record)+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        probe_records(tmp_path, manifest)


@pytest.mark.parametrize("option", ["--self-test-reference-manifest", "--self-test-reopen-report"])
def test_reference_cli_requires_an_isolated_self_test_directory(option):
    result = subprocess.run([sys.executable, "-m", "kraken_hub", option, "unused.jsonl"],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "requires --self-test-dir" in result.stderr


def test_previous_fixture_rejects_another_profile_before_opening_accounts(tmp_path):
    from uuid import uuid4
    from kraken_hub.presentation.probes.deployment_upgrade_probe import previous_fixture

    root = tmp_path / ("desktop-self-test-" + uuid4().hex)
    (root / "profile").mkdir(parents=True)
    report = root / "report.json"
    report.write_text(json.dumps(dict(status="passed", directory=str(tmp_path / "another-profile"),
        settings_isolated=True, source_files_unchanged=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="isolated successful fixture"):
        previous_fixture(report)
    assert not (root / "profile" / "accounts.sqlite3").exists()
