"""Resume a hash-checked, managed reference project without changing its sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--width", default=100, type=int)
    parser.add_argument("--limit", type=int, help="Diagnostic stepping; a later run resumes the same full project")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root)]
    from kraken_hub.composition import EmbeddedProjectService
    from kraken_hub.secret_store import KeyringSecretStore
    from kraken_manager.application.imports import ImportPlanner, ImportSource, ImportMappingMode
    from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
    from kraken_manager.infrastructure.filesystem.locking import ProjectFileLock
    from kraken_manager.infrastructure.filesystem._atomic import atomic_write_json

    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    records = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines()]
    if not records or args.width < 1:
        raise ValueError("A nonempty manifest and positive catalog width are required")
    profile = args.profile.resolve()
    for record in records:
        for field in ("image", "vector"):
            if profile.is_relative_to(Path(record[field]).resolve().parent):
                raise ValueError("Audit profile must be separate from original image/vector directories")
    profile.mkdir(parents=True, exist_ok=True)
    state_path = profile / "state.json"

    def save(state):
        atomic_write_json(state_path, state)

    with ProjectFileLock(profile / "import.lock"):
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else dict(
                manifest_sha256=manifest_sha,
                width=args.width,
                height=math.ceil(len(records) / args.width),
                secret_service="Kraken.ReferenceAudit." + manifest_sha[:24],
                completed_keys=[],
                coordinate_semantics="catalog positions only; physical adjacency is unknown",
            )
        )
        if state["manifest_sha256"] != manifest_sha or state["width"] != args.width:
            raise ValueError("Resume manifest or catalog dimensions changed; use a new audit profile")
        save(state)
        service = EmbeddedProjectService(profile / "data", secret_store=KeyringSecretStore(state["secret_service"]))
        account = service.accounts.get_by_username("reference-audit")
        if account is None:
            actor = service.create_initial_account("reference-audit", "Reference audit importer", "").principal
        else:
            actor = service.identities.get(account.account_id)
        if actor is None or not actor.active:
            raise ValueError("Audit importer identity is missing or inactive")
        projects = service.list_projects()
        if "project_id" not in state:
            if len(projects) > 1:
                raise ValueError("Audit profile has unexpected projects")
            project = (
                projects[0]
                if projects
                else service.create_project(
                    principal=actor,
                    name="NN reference audit",
                    width=state["width"],
                    height=state["height"],
                    orientation=GridOrientation.Y_DOWN,
                    idempotency_key="reference-project",
                )
            )
            state["project_id"] = str(project.id)
            save(state)
        project = service.get_project(state["project_id"])
        if (project.width, project.height) != (state["width"], state["height"]):
            raise ValueError("Audit project dimensions do not match its source manifest")
        layers = service.list_layers(project.id)
        layer = (
            layers[0]
            if layers
            else service.create_layer(
                principal=actor,
                project=project,
                name="Metal reference corpus",
                layer_type=LayerType.METAL,
                order=0,
                idempotency_key="reference-layer",
            )
        )
        representations = {r.name: r for r in service.list_representations(project.id, layer.id)}
        image = representations.get("Microscopy as supplied")
        if image is None:
            image = service.create_representation(
                principal=actor,
                project=service.get_project(project.id),
                layer=service.list_layers(project.id)[0],
                name="Microscopy as supplied",
                kind=RepresentationKind.IMAGE,
                active=True,
                idempotency_key="reference-images",
            )
        vector = representations.get("CIF annotation as supplied")
        if vector is None:
            vector = service.create_representation(
                principal=actor,
                project=service.get_project(project.id),
                layer=service.list_layers(project.id)[0],
                name="CIF annotation as supplied",
                kind=RepresentationKind.VECTOR,
                source_image_representation_id=image.id,
                active=True,
                idempotency_key="reference-vectors",
            )
        state.update(
            owner_id=str(actor.id),
            layer_id=str(layer.id),
            image_representation_id=str(image.id),
            vector_representation_id=str(vector.id),
        )
        save(state)
        completed = set(state["completed_keys"])
        count = 0
        started = time.monotonic()
        receipts = {}
        receipt_path = profile / "imported.jsonl"
        if receipt_path.exists():
            for line in receipt_path.read_text(encoding="utf-8").splitlines():
                receipt = json.loads(line)
                if receipt["key"] in receipts and receipts[receipt["key"]] != receipt:
                    raise ValueError("Import receipts disagree; preserve evidence and investigate")
                receipts[receipt["key"]] = receipt
        with (profile / "imported.jsonl").open("a", encoding="utf-8") as output:
            for index, record in enumerate(records):
                if (profile / "stop.requested").exists():
                    print("Cooperative stop requested; committed progress retained.", flush=True)
                    break
                if record["key"] in completed:
                    continue
                if args.limit is not None and count >= args.limit:
                    break
                x, y = index % args.width + 1, index // args.width + 1
                imported = {}
                for role, representation in (("image", image), ("vector", vector)):
                    source = Path(record[role])
                    expected = record[role + "_sha256"]
                    item = ImportSource(str(source), source.name, record[role + "_bytes"], expected)
                    plan = ImportPlanner().plan(
                        width=project.width,
                        height=project.height,
                        sources=(item,),
                        mode=ImportMappingMode.EXPLICIT,
                        explicit={str(source): (x, y)},
                    )
                    result = service.commit_managed_import(
                        principal=actor,
                        project=service.get_project(project.id),
                        layer=layer,
                        representation=representation,
                        plan=plan,
                        idempotency_key=f"reference:{record['key']}:{role}:{expected}",
                    )
                    version = result.versions[0]
                    if version.sha256 != expected:
                        raise ValueError("Committed reference bytes do not match their audited identity")
                    imported[role + "_version_id"] = str(version.id)
                if record["issues"]:
                    severe = any(issue.startswith("decode_or_parse_error") for issue in record["issues"])
                    issue = service.create_quality_issue(
                        project_id=project.id,
                        principal=actor,
                        arguments=dict(
                            title=f"Reference annotation review {record['key']}",
                            description="Read-only corpus audit findings: " + "; ".join(record["issues"]),
                            severity="major" if severe else "minor",
                            scope=dict(layer_ids=[str(layer.id)], frame_coordinates=[[x, y]]),
                            source_version_ids=[imported["vector_version_id"]],
                        ),
                        idempotency_key=f"reference-finding:{record['key']}:{record['vector_sha256']}",
                    )
                    imported["quality_issue_id"] = issue.id
                receipt = dict(key=record["key"], x=x, y=y, **imported)
                prior = receipts.get(record["key"])
                if prior is not None and prior != receipt:
                    raise ValueError("Resumed import changed its committed version/quality receipt")
                if prior is None:
                    output.write(json.dumps(receipt, sort_keys=True) + "\n")
                    output.flush()
                    receipts[record["key"]] = receipt
                completed.add(record["key"])
                state["completed_keys"] = sorted(completed)
                save(state)
                count += 1
                if count % 25 == 0:
                    print(
                        f"{len(completed)}/{len(records)} pairs managed; current pass {time.monotonic() - started:.2f}s",
                        flush=True,
                    )
        state.update(complete=len(completed) == len(records), last_pass_seconds=round(time.monotonic() - started, 3))
        save(state)
        print(
            json.dumps({key: value for key, value in state.items() if key != "completed_keys"}, ensure_ascii=False),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
