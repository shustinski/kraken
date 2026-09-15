"""Streaming validation for bounded analysis-result record bundles."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from typing import BinaryIO

from .analysis_protocol import AnalysisFrameResult
from .analysis_run_protocol import AnalysisRecordBundle


MAX_COMPRESSED_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_ANALYSIS_RECORD_BYTES = 1024 * 1024
MAX_METRICS_PER_FRAME = 256


def stream_bundle_records(
    stream: BinaryIO,
    bundle: AnalysisRecordBundle,
    *,
    expected_frame_ids: Iterable[str],
) -> Iterator[AnalysisFrameResult]:
    expected = frozenset(str(frame_id) for frame_id in expected_frame_ids)
    if bundle.compressed_size > MAX_COMPRESSED_BUNDLE_BYTES:
        raise ValueError("Compressed analysis record bundle exceeds the 64 MiB limit")
    if bundle.uncompressed_size > MAX_UNCOMPRESSED_BUNDLE_BYTES:
        raise ValueError("Analysis record bundle exceeds the 256 MiB expansion limit")
    seen: set[str] = set()
    uncompressed_size = 0
    with gzip.GzipFile(fileobj=stream, mode="rb") as archive:
        while True:
            raw_line = archive.readline(MAX_ANALYSIS_RECORD_BYTES + 1)
            if not raw_line:
                break
            if len(raw_line) > MAX_ANALYSIS_RECORD_BYTES:
                raise ValueError("Analysis frame record exceeds the 1 MiB limit")
            uncompressed_size += len(raw_line)
            if uncompressed_size > MAX_UNCOMPRESSED_BUNDLE_BYTES:
                raise ValueError("Analysis record bundle exceeds the 256 MiB expansion limit")
            try:
                payload = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Analysis record bundle contains invalid JSONL") from exc
            if not isinstance(payload, dict):
                raise ValueError("Analysis frame record must be a JSON object")
            frame = AnalysisFrameResult.from_payload(payload)
            if frame.frame_id not in expected:
                raise ValueError(f"Analysis bundle contains an unknown frame: {frame.frame_id}")
            if frame.frame_id in seen:
                raise ValueError(f"Analysis bundle contains a duplicate frame: {frame.frame_id}")
            if len(frame.metrics) > MAX_METRICS_PER_FRAME:
                raise ValueError(f"Analysis frame {frame.frame_id} exceeds the metric limit")
            seen.add(frame.frame_id)
            yield frame
    if uncompressed_size != bundle.uncompressed_size:
        raise ValueError("Analysis bundle uncompressed size does not match its manifest")
    if len(seen) != bundle.frame_count:
        raise ValueError("Analysis bundle frame count does not match its manifest")
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"Analysis bundle does not contain every partition frame: {', '.join(missing[:5])}")


def sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


__all__ = [
    "MAX_ANALYSIS_RECORD_BYTES",
    "MAX_COMPRESSED_BUNDLE_BYTES",
    "MAX_METRICS_PER_FRAME",
    "MAX_UNCOMPRESSED_BUNDLE_BYTES",
    "sha256_stream",
    "stream_bundle_records",
]
