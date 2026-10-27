"""Publish disjoint real train/validation inputs; heldout is excluded from the job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

from kraken_core.safe_files import open_regular_read
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.secret_store import KeyringSecretStore
from kraken_manager.application.dto import AddArtifactVersionCommand, CommandContext
from kraken_manager.application.use_cases import AddArtifactVersionHandler
from kraken_manager.domain.artifacts import ArtifactScope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = json.loads((args.profile / "state.json").read_text())
    manifest_bytes = (args.inventory / "pairs.jsonl").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == state["manifest_sha256"]
    records = {row["key"]: row for row in map(json.loads, manifest_bytes.decode("utf-8").splitlines())}
    splits = json.loads((args.inventory / "splits.json").read_text())
    used = {key for values in splits.values() for key in values}
    assert len(used) == sum(map(len, splits.values()))
    assert all(not records[key]["issues"] for key in used)
    image_hashes = [
        {records[key]["image_sha256"] for key in splits[name]} for name in ("train", "validation", "heldout")
    ]
    assert all(not image_hashes[i] & image_hashes[j] for i in range(3) for j in range(i + 1, 3))
    receipts = {row["key"]: row for row in map(json.loads, (args.profile / "imported.jsonl").read_text().splitlines())}
    provenance = []
    inputs = []
    temporary = output / "training-dataset.zip.partial"
    archive_path = output / "training-dataset.zip"
    count = 0
    with temporary.open("xb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for split in ("train", "validation"):
                prefix = "" if split == "train" else "validation/"
                for key in splits[split]:
                    record = records[key]
                    item = dict(key=key, partition=split)
                    for role, folder in (("image", "images"), ("vector", "cif")):
                        source = Path(record[role])
                        name = prefix + folder + "/" + source.name
                        digest = hashlib.sha256()
                        size = 0
                        with (
                            open_regular_read(source, root=source.parent) as reader,
                            archive.open(name, "w", force_zip64=True) as writer,
                        ):
                            while chunk := reader.read(1024 * 1024):
                                size += len(chunk)
                                digest.update(chunk)
                                writer.write(chunk)
                        if digest.hexdigest() != record[role + "_sha256"] or size != record[role + "_bytes"]:
                            raise ValueError("Source changed after the inventory; dataset is not published")
                        version_id = receipts[key][role + "_version_id"]
                        inputs.append(version_id)
                        item[role] = dict(
                            path=name, artifact_version_id=version_id, sha256=digest.hexdigest(), size_bytes=size
                        )
                    provenance.append(item)
                    count += 1
                    if count % 250 == 0:
                        print(f"{count} train/validation pairs packed; heldout remains excluded", flush=True)
            archive.writestr(
                "reference-partitions.json",
                json.dumps(
                    dict(
                        schema="kraken.reference-training.v1",
                        source_manifest_sha256=state["manifest_sha256"],
                        project_id=state["project_id"],
                        train_count=len(splits["train"]),
                        validation_count=len(splits["validation"]),
                        heldout_count_excluded=len(splits["heldout"]),
                        truth_status="supplied_annotations_not_independently_verified",
                        entries=provenance,
                    ),
                    sort_keys=True,
                ),
            )
        raw.flush()
        os.fsync(raw.fileno())
    os.rename(temporary, archive_path)
    digest = hashlib.sha256()
    with archive_path.open("rb") as reader:
        while chunk := reader.read(4 * 1024 * 1024):
            digest.update(chunk)
    archive_sha = digest.hexdigest()
    service = EmbeddedProjectService(args.profile / "data", secret_store=KeyringSecretStore(state["secret_service"]))
    owner = service.identities.get(state["owner_id"])
    assert owner is not None and owner.active
    key = "reference-training-dataset:" + state["manifest_sha256"] + ":" + archive_sha
    series = service.create_artifact_series(
        principal=owner,
        project_id=state["project_id"],
        scope=ArtifactScope.LAYER_ATTACHMENT,
        layer_id=state["layer_id"],
        name="NN supplied annotation train/validation dataset",
        idempotency_key=key + ":series",
    )
    with archive_path.open("rb") as reader:
        version = AddArtifactVersionHandler(service._uow(state["project_id"]), service.profiles, service.clock)(
            AddArtifactVersionCommand(
                context=CommandContext(actor=owner, idempotency_key=key + ":version"),
                project_id=state["project_id"],
                series_id=series.id,
                filename=archive_path.name,
                media_type="application/zip",
                expected_series_revision=service.artifact_stream_revision(state["project_id"], series.id),
                expected_sha256=archive_sha,
                input_version_ids=tuple(inputs),
                tool_name="Kraken reference dataset publisher",
                tool_version="1",
                parameters=dict(
                    source_manifest_sha256=state["manifest_sha256"],
                    train_count=len(splits["train"]),
                    validation_count=len(splits["validation"]),
                    heldout_count_excluded=len(splits["heldout"]),
                    truth_status="supplied_annotations_not_independently_verified",
                    artifact_role="dataset",
                    dataset_format="kraken.reference-training.v1",
                ),
            ),
            iter(lambda: reader.read(4 * 1024 * 1024), b""),
        )
    result = dict(
        status="passed",
        project_id=state["project_id"],
        layer_id=state["layer_id"],
        dataset_series_id=str(series.id),
        dataset_version_id=str(version.id),
        dataset_sha256=archive_sha,
        size_bytes=version.size_bytes,
        path=str(archive_path),
        input_version_count=len(inputs),
        train_count=len(splits["train"]),
        validation_count=len(splits["validation"]),
        heldout_excluded=len(splits["heldout"]),
        source_manifest_sha256=state["manifest_sha256"],
        seconds=round(time.monotonic() - started, 3),
    )
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
