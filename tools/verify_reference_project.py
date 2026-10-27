"""Read-only full source/blob/receipt/event verification of a managed real corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from kraken_manager.application.quality import QualityService
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
from kraken_manager.infrastructure.filesystem import FilesystemEventStore, SQLiteProjectionStore


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    evidence = {"status": "running"}
    try:
        state = json.loads((args.profile / "state.json").read_text(encoding="utf-8"))
        manifest_bytes = args.manifest.read_bytes()
        assert state["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
        records = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines()]
        assert state["complete"] and set(state["completed_keys"]) == {r["key"] for r in records}
        receipts = [
            json.loads(line) for line in (args.profile / "imported.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        by_key = {row["key"]: row for row in receipts}
        assert len(by_key) == len(receipts) == len(records), "Receipts contain missing or duplicate frames"
        catalog = args.profile / "data/catalog"
        events = FilesystemEventStore(catalog, state["project_id"])
        projections = SQLiteProjectionStore(catalog, state["project_id"])
        blobs = FilesystemBlobStore.for_project(catalog, state["project_id"])
        with events.lock.hold():
            count, streams = events.verify()
            assert projections.checkpoint == events.last_global_position()
            project = projections.get_project(state["project_id"])
            series = projections.list_artifact_series(project.id)
            assert len(series) == 2 * len(records)
            active = {}
            for start in range(0, len(series), 512):
                active.update(
                    projections.list_active_artifact_versions([item.id for item in series[start : start + 512]])
                )
            versions = {str(version.id): version for version in active.values()}
            series_by_id = {str(item.id): item for item in series}
            assert len(versions) == len(series)
            issues = {issue.id: issue for issue in QualityService.list_issues(events, project.id)}
            assert len(issues) == sum(bool(record["issues"]) for record in records)
            event_counts = Counter(event.data.get("event_type") for event in events.iter_project())
            expected_hashes = set()
            for index, record in enumerate(records):
                receipt = by_key[record["key"]]
                x, y = index % state["width"] + 1, index // state["width"] + 1
                assert (receipt["x"], receipt["y"]) == (x, y)
                for role in ("image", "vector"):
                    version = versions[receipt[role + "_version_id"]]
                    representation_id = state[role + "_representation_id"]
                    expected_series = deterministic_frame_series_id(representation_id, project.frame_id_at(x, y))
                    assert str(version.series_id) == str(expected_series)
                    item = series_by_id[str(version.series_id)]
                    assert str(item.project_id) == str(project.id) and str(item.layer_id) == state["layer_id"]
                    assert version.sha256 == record[role + "_sha256"] and version.size_bytes == record[role + "_bytes"]
                    expected_hashes.add(version.sha256)
                if record["issues"]:
                    issue = issues[receipt["quality_issue_id"]]
                    assert str(issue.state) == "open"
                    assert tuple(issue.source_version_ids) == (receipt["vector_version_id"],)
                    assert str(project.frame_id_at(x, y)) in issue.scope.frame_ids

            def check(record):
                total = 0
                for role in ("image", "vector"):
                    source = Path(record[role])
                    assert (
                        source.stat().st_size == record[role + "_bytes"]
                        and file_sha(source) == record[role + "_sha256"]
                    )
                    version = versions[by_key[record["key"]][role + "_version_id"]]
                    verified = blobs.verify(version.blob)
                    assert verified.sha256 == record[role + "_sha256"]
                    total += version.size_bytes
                return total

            total_bytes = 0
            with ThreadPoolExecutor(max_workers=4) as pool:
                for index, size in enumerate(pool.map(check, records), 1):
                    total_bytes += size
                    if index % 250 == 0:
                        print(
                            f"{index}/{len(records)} real image/CIF pairs: original SHA and managed blob verified",
                            flush=True,
                        )
            stored_hashes = {reference.sha256 for reference in blobs.iter_refs()}
            assert expected_hashes <= stored_hashes
        evidence.update(
            status="passed",
            pairs=len(records),
            artifact_series=len(series),
            active_versions=len(versions),
            source_and_blob_bytes_each=total_bytes,
            source_unchanged=True,
            quality_issues=len(issues),
            quality_severities=dict(Counter(str(issue.severity) for issue in issues.values())),
            events=count,
            streams=streams,
            event_types=dict(event_counts),
            unique_referenced_blobs=len(expected_hashes),
            unreferenced_blobs=len(stored_hashes - expected_hashes),
            receipt_duplicates=0,
            project_id=str(project.id),
            coordinate_semantics=state["coordinate_semantics"],
            manifest_sha256=state["manifest_sha256"],
        )
    except BaseException as exc:
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        evidence["seconds"] = round(time.monotonic() - started, 3)
        (report / "report.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
