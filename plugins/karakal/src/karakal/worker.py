"""Manifest-driven command line Worker launched by Kraken Agent."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from kraken_core.analysis_protocol import AnalysisOutcome
from kraken_core.analysis_run_protocol import (
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    AnalysisRecordBundle,
    canonical_json,
)

from .core.headless import run_analysis


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bundle(path: Path, frames: tuple) -> AnalysisRecordBundle:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    uncompressed_size = 0
    with temporary.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as archive:
            for frame in frames:
                encoded = (canonical_json(frame.to_payload()) + "\n").encode("utf-8")
                uncompressed_size += len(encoded)
                archive.write(encoded)
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return AnalysisRecordBundle(
        relative_path=path.relative_to(path.parents[1]).as_posix(),
        sha256=digest,
        compressed_size=path.stat().st_size,
        uncompressed_size=uncompressed_size,
        frame_count=len(frames),
    )


def execute(
    job_path: Path,
    result_path: Path,
    progress_path: Path,
    workspace: Path,
    cancel_path: Path | None = None,
) -> AnalysisPartitionResultManifest:
    job = AnalysisPartitionJobManifest.read(job_path)
    output_dir = workspace / "outputs"

    def progress(completed: int, total: int, frame_id: str) -> None:
        _atomic_json(
            progress_path,
            {"job_id": job.job_id, "completed_frames": completed, "total_frames": total, "frame_id": frame_id},
        )

    execution = run_analysis(
        job,
        workspace,
        output_dir,
        progress,
        None if cancel_path is None else cancel_path.exists,
    )
    bundle = None
    if execution.frames:
        bundle = _write_bundle(output_dir / "frames.jsonl.gz", execution.frames)
    result = AnalysisPartitionResultManifest(
        job_id=job.job_id,
        run_id=job.run_id,
        partition_id=job.partition_id,
        project_id=job.project_id,
        outcome=execution.outcome,
        bundle=bundle,
        message=execution.message,
    )
    _atomic_json(result_path, result.to_payload())
    return result


def _path_argument(value: str | None, environment_key: str) -> Path:
    resolved = str(value or os.environ.get(environment_key, "")).strip()
    if not resolved:
        raise ValueError(f"Missing --{environment_key.lower().replace('_', '-')} / {environment_key}")
    return Path(resolved).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Karakal headless analysis worker")
    parser.add_argument("--job")
    parser.add_argument("--result")
    parser.add_argument("--progress")
    parser.add_argument("--workspace")
    parser.add_argument("--cancel")
    args = parser.parse_args(argv)
    try:
        result = execute(
            _path_argument(args.job, "KRAKEN_JOB_MANIFEST"),
            _path_argument(args.result, "KRAKEN_RESULT_MANIFEST"),
            _path_argument(args.progress, "KRAKEN_PROGRESS_PATH"),
            _path_argument(args.workspace, "KRAKEN_STAGING_ROOT"),
            None if not (args.cancel or os.environ.get("KRAKEN_CANCEL_PATH")) else _path_argument(args.cancel, "KRAKEN_CANCEL_PATH"),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 2
    if result.outcome == AnalysisOutcome.CANCELLED:
        return 130
    if result.outcome == AnalysisOutcome.FAILED:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
