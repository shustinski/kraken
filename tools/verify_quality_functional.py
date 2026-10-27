"""Small real managed-corpus quality lifecycle; no training or geometry assessment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(3)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--managed-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "plugins/contour/src")]
    from kraken_hub.composition import EmbeddedProjectService
    from kraken_hub.secret_store import KeyringSecretStore
    from kraken_manager.application.errors import AuthorizationError
    from kraken_manager.application.imports import ImportPlanner, ImportSource, ImportMappingMode
    from kraken_manager.domain.identity import Performer, ProjectRole
    from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
    from kraken_manager.domain.artifacts import BlobRef
    from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
    from contour.serializers_parts.load_polygons_cif import load_polygons_cif
    from PIL import Image

    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    record = next(r for r in records if r["issues"] == ["reference_image_name_mismatch"])
    state = json.loads((args.managed_profile / "state.json").read_text(encoding="utf-8"))
    imported = next(json.loads(line) for line in (args.managed_profile / "imported.jsonl").read_text(encoding="utf-8").splitlines()
                    if json.loads(line)["key"] == record["key"])
    blobs = FilesystemBlobStore.for_project(args.managed_profile / "data/catalog", state["project_id"])
    inputs = {}
    for role in ("image", "vector"):
        ref = BlobRef(record[role + "_sha256"], record[role + "_bytes"])
        blobs.verify(ref)
        data = blobs.read(ref)
        assert sha(data) == ref.sha256 == sha(Path(record[role]).read_bytes())
        path = output / "input" / Path(record[role]).name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        inputs[role] = path
    secret_name = "Kraken.QualityFunctional." + str(uuid4())
    secrets = KeyringSecretStore(secret_name)
    service = EmbeddedProjectService(output / "data", secret_store=secrets)
    password = uuid4().hex
    owner = service.create_initial_account("owner", "Functional audit owner", password).principal
    actors = {}
    for name in ("corrector", "reviewer"):
        service.accounts.create_account(name, name.title(), password, actor_id=str(owner.id))
        actors[name] = service.login(name, password).principal
    project = service.create_project(principal=owner, name="Technical CIF header correction only", width=1, height=1,
                                     orientation=GridOrientation.Y_DOWN, idempotency_key="project")
    for name, principal in actors.items():
        service.assign_project_role(principal=owner, project=project, target_principal_id=principal.id,
            role=ProjectRole.MAINTAINER, expected_revision=0, idempotency_key="role-" + name)
    performer = service.performers.create(Performer.create(name="Assigned corrector", color="#336699",
                                                          principal_id=actors["corrector"].id))
    layer = service.create_layer(principal=owner, project=project, name="Supplied metal annotation",
                                 layer_type=LayerType.METAL, order=0, idempotency_key="layer")
    versions = {}
    image_rep = None
    for role, kind in (("image", RepresentationKind.IMAGE), ("vector", RepresentationKind.VECTOR)):
        kwargs = {} if image_rep is None else {"source_image_representation_id": image_rep.id}
        rep = service.create_representation(principal=owner, project=service.get_project(project.id), layer=layer,
            name=role, kind=kind, active=True, idempotency_key="rep-" + role, **kwargs)
        if role == "image":
            image_rep = rep
        path = inputs[role]
        source = ImportSource(str(path), path.name, path.stat().st_size, record[role + "_sha256"])
        plan = ImportPlanner().plan(width=1, height=1, sources=(source,), mode=ImportMappingMode.EXPLICIT,
                                    explicit={str(path): (1, 1)})
        service.commit_managed_import(principal=owner, project=service.get_project(project.id), layer=layer,
                                      representation=rep, plan=plan, idempotency_key="import-" + role)
        series = next(s for s in service.list_artifact_series(project.id) if s.representation_id == rep.id)
        versions[role] = service.artifact_versions(project.id, series.id)[0]
    original = inputs["vector"].read_bytes()
    lines = original.splitlines(keepends=True)
    positions = [i for i, line in enumerate(lines) if line.strip().startswith(b"( R ")]
    assert len(positions) == 1
    position = positions[0]
    ending = b"\r\n" if lines[position].endswith(b"\r\n") else b"\n" if lines[position].endswith(b"\n") else b""
    fixed_lines = list(lines)
    fixed_lines[position] = b"( R " + inputs["image"].name.encode("ascii") + b" );" + ending
    corrected_path = output / "corrected.cif"
    corrected_path.write_bytes(b"".join(fixed_lines))
    issue = service.create_quality_issue(project_id=project.id, principal=owner, arguments={
        "title": "CIF image identity mismatch", "description": "Technical header identity only; geometry is unassessed",
        "severity": "minor", "source_version_ids": [str(versions["vector"].id)],
        "scope": {"layer_ids": [str(layer.id)], "frame_coordinates": [[1, 1]]}}, idempotency_key="issue")

    def command(operation, arguments, actor):
        return service.mutate_quality_issue(project_id=project.id, issue_id=issue.id, principal=actor,
            expected_revision=issue.revision, operation=operation, arguments=arguments, idempotency_key=operation)

    issue = command("assign", {"assignee_id": str(performer.id), "note": "Correct only the image header reference"}, owner)
    corrected = service.add_managed_artifact_version(principal=actors["corrector"], project_id=project.id,
        series_id=versions["vector"].series_id, source=corrected_path,
        parent_version_id=versions["vector"].id, idempotency_key="corrected-version")
    issue = command("correct", {"evidence_version_ids": [str(corrected.id)], "note": "Only R header replaced; original version retained"}, actors["corrector"])
    try:
        command("recheck", {"passed": True, "evidence_version_ids": [str(corrected.id)], "note": "Self review forbidden"}, actors["corrector"])
    except AuthorizationError as error:
        assert "own correction" in str(error)
    else:
        raise AssertionError("Self review was allowed")
    exported = output / "export" / inputs["vector"].name
    service.export_artifact_version(project.id, corrected, exported)
    name, size, _ = load_polygons_cif(exported)
    with Image.open(inputs["image"]) as image:
        assert name == inputs["image"].name and size == image.size
    observed_lines = exported.read_bytes().splitlines(keepends=True)
    assert observed_lines[:position] + observed_lines[position + 1:] == lines[:position] + lines[position + 1:]
    issue = command("recheck", {"passed": True, "evidence_version_ids": [str(corrected.id)],
        "note": "Independent parser and JPEG header check passed; all non-R bytes preserved; geometry/electrical correctness unassessed"}, actors["reviewer"])
    issue = command("accept", {"note": "Technical image identity correction accepted only"}, actors["reviewer"])
    try:
        service.export_artifact_version(project.id, corrected, exported)
    except FileExistsError:
        assert sha(exported.read_bytes()) == corrected.blob.sha256
    else:
        raise AssertionError("Existing destination was overwritten")
    reopened = EmbeddedProjectService(output / "data", secret_store=secrets)
    assert reopened.login("reviewer", password).principal.id == actors["reviewer"].id
    assert reopened.get_quality_issue(project.id, issue.id) == issue
    reopened._projection(project.id).destroy_cache()
    assert reopened.get_quality_issue(project.id, issue.id) == issue
    history = reopened.quality_issue_history(project.id, issue.id)
    assert [e["operation"] for e in history] == ["quality." + operation for operation in ("create", "assign", "correct", "recheck", "accept")]
    assert history[2]["actor_id"] != history[3]["actor_id"] == history[4]["actor_id"]
    for role in inputs:
        assert sha(Path(record[role]).read_bytes()) == record[role + "_sha256"]
        assert sha(inputs[role].read_bytes()) == record[role + "_sha256"]
    store = FilesystemBlobStore.for_project(reopened.catalog_root, str(project.id))
    for version in (*versions.values(), corrected):
        store.verify(version.blob)
    receipt = {"status": "passed", "seconds": round(time.monotonic() - started, 3), "key": record["key"],
        "source_managed_project_id": state["project_id"], "source_managed_versions": imported,
        "project_id": str(project.id), "issue_id": str(issue.id), "state": issue.state.value,
        "actors": {"owner": str(owner.id), **{k: str(v.id) for k, v in actors.items()}},
        "inputs": {role: {"path": record[role], "sha256": record[role + "_sha256"]} for role in inputs},
        "correction_sha256": corrected.blob.sha256, "export_sha256": sha(exported.read_bytes()),
        "history": history, "checks": ["managed and disk input SHA256", "real account authentication and ACL",
        "linked assignee", "immutable parent-linked correction", "self-review rejected", "independent CIF parser/JPEG header validation",
        "all non-R bytes preserved", "acceptance", "export collision preserves destination", "service reopen and projection rebuild",
        "original source unchanged", "original and corrected managed blobs verified"],
        "limitations": ["One technical image-header mismatch only", "No vector geometry assessment", "No training",
        "No electrical ground truth or electrical acceptance", "Local service; no new UI or PostgreSQL acceptance claimed",
        "Receipt is an audit tool output, not a dedicated quality-report API"]}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"status": "passed", "seconds": receipt["seconds"], "receipt": str(output / "receipt.json")}))


if __name__ == "__main__":
    main()
