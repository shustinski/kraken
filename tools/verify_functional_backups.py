"""Restore existing bounded-model and real-pair quality projects without training."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

def main():
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root)]
    from kraken_hub.composition import EmbeddedProjectService
    from kraken_manager.infrastructure.filesystem import FilesystemEventStore
    from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
    from tests.test_hub_embedded_service_parts.memory_secret_store import MemorySecretStoreMixin

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-project", type=Path, required=True, help="Completed bounded-model gate directory with receipt.json and data/")
    parser.add_argument("--quality-project", type=Path, required=True, help="Completed quality gate directory with receipt.json and data/")
    parser.add_argument("--output", type=Path, required=True, help="Fresh isolated output directory")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    evidence = {"status": "running", "training_started": False, "projects": []}

    for name, original_root in (
        ("model", args.model_project.resolve(strict=True)),
        ("quality", args.quality_project.resolve(strict=True)),
    ):
        original_receipt = json.loads((original_root / "receipt.json").read_text(encoding="utf-8"))
        project_id = original_receipt["project_id"]
        secret = MemorySecretStoreMixin._memory_secret_store()
        source = EmbeddedProjectService(original_root / "data", secret_store=secret)
        project = source.get_project(project_id)
        source_events = FilesystemEventStore(source.catalog_root, project_id)
        before = source_events.verify()
        target = output / name
        target.mkdir()
        manifest = source.export_backup(project_id, target / "bundle")
        restored = EmbeddedProjectService(target / "restored", secret_store=secret)
        assert restored.import_backup(target / "bundle") == project
        assert restored.scan_integrity().valid
        restored._projection(project_id).destroy_cache()
        reopened = EmbeddedProjectService(target / "restored", secret_store=secret)
        assert reopened.get_project(project_id) == project
        assert reopened.list_layers(project_id) == source.list_layers(project_id)
        assert reopened.list_work_plans(project_id) == source.list_work_plans(project_id)
        assert reopened.list_quality_issues(project_id) == source.list_quality_issues(project_id)
        restored_events = FilesystemEventStore(reopened.catalog_root, project_id)
        assert restored_events.verify() == before
        versions = []
        blobs = FilesystemBlobStore.for_project(reopened.catalog_root, project_id)
        for series in source.list_artifact_series(project_id):
            expected = source.artifact_versions(project_id, series.id)
            assert reopened.artifact_versions(project_id, series.id) == expected
            for version in expected:
                if version.blob is not None:
                    blobs.verify(version.blob)
                    assert hashlib.sha256(blobs.read(version.blob)).hexdigest() == version.blob.sha256
                    versions.append(str(version.id))
        if name == "model":
            for key in ("dataset_version_id", "image_version_id", "model_version_id", "mask_version_id"):
                assert original_receipt[key] in versions
            for layer in source.list_layers(project_id):
                assert reopened.get_layer_pipeline(project_id, layer.id) == source.get_layer_pipeline(project_id, layer.id)
            for key in ("model", "mask"):
                assert reopened.artifact_version(project_id, original_receipt[key + "_version_id"]).blob.sha256 == original_receipt[key + "_sha256"]
        else:
            issue = reopened.get_quality_issue(project_id, original_receipt["issue_id"])
            assert issue.state.value == "accepted"
            assert reopened.quality_issue_history(project_id, issue.id) == source.quality_issue_history(project_id, issue.id)
        assert source_events.verify() == before
        evidence["projects"].append({"kind": name, "project_id": project_id, "events": manifest.event_count,
            "managed_versions": len(versions), "checks": ["canonical backup and restore", "reopen after projection destruction",
                "version metadata and SHA", "domain views and lineage", "source event log unchanged"]})

    evidence.update(status="passed", seconds=round(time.monotonic() - started, 3),
        limitations=["Local project backup only; no PostgreSQL disaster recovery or account-secret transfer claim",
            "No model training or vectorization-quality evaluation"])
    (output / "receipt.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
