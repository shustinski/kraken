"""Restore an already-computed mosaic project; never train or recompute images."""
from pathlib import Path
import argparse
import json
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root)]
    from kraken_hub.composition import EmbeddedProjectService
    from tests.test_hub_embedded_service_parts.memory_secret_store import MemorySecretStoreMixin
    from kraken_manager.domain.multilayer import MultilayerModel
    from kraken_manager.application.errors import NotFoundError
    from kraken_manager.infrastructure.filesystem import FilesystemEventStore
    from kraken_manager.infrastructure.mosaic import MosaicReader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    secret = MemorySecretStoreMixin._memory_secret_store()
    source = EmbeddedProjectService(args.data_dir.resolve(strict=True), secret_store=secret)
    snapshot = source.get_multilayer_model(args.project_id)
    model = MultilayerModel.from_dict(snapshot)
    imported = [r for r in model.stitch_renders if r.state == "imported"]
    if not imported:
        raise ValueError("Source project has no accepted mosaic publication")
    workflows = {}
    for layer in source.list_layers(args.project_id):
        try:
            workflows[str(layer.id)] = source.get_layer_pipeline(args.project_id, layer.id)
        except NotFoundError:
            pass
    events = FilesystemEventStore(source.catalog_root, args.project_id)
    before = events.verify()
    bundle = source.export_backup(args.project_id, output / "bundle")
    restored = EmbeddedProjectService(output / "restored", secret_store=secret)
    project = restored.import_backup(output / "bundle")
    assert str(project.id) == args.project_id
    assert restored.scan_integrity().valid
    restored._projection(project.id).destroy_cache()
    reopened = EmbeddedProjectService(output / "restored", secret_store=secret)
    assert reopened.get_multilayer_model(project.id) == snapshot
    for layer_id, workflow in workflows.items():
        assert reopened.get_layer_pipeline(project.id, layer_id) == workflow
    checked = []
    for request in imported:
        expected = source.get_artifact_version(project.id, request.output.version_id)
        actual = reopened.get_artifact_version(project.id, request.output.version_id)
        assert actual == expected
        plan = next(s for s in model.stitches if (s.id, s.revision) == (request.stitch_id, request.stitch_revision))
        with MosaicReader(reopened.artifact_source_path(project.id, actual.id)) as reader:
            metadata = reader.validate(expected_plan=plan, expected_job_id=request.job_id)
        checked.append({"job_id": request.job_id, "version_id": str(actual.id), "sha256": actual.sha256,
            "blocks": metadata["render"]["block_count"], "input_versions": list(actual.input_version_ids)})
    assert events.verify() == before
    receipt = {"status": "passed", "training_started": False, "computation_started": False,
        "project_id": args.project_id, "events": bundle.event_count, "mosaics": checked, "pipeline_workflows": len(workflows),
        "seconds": round(time.monotonic()-started, 3), "checks": ["canonical backup and restore", "blob SHA and lineage",
            "projection destruction and reopen", "complete block and coverage validation", "source events unchanged"]}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
