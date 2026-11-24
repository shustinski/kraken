"""Carry local Contour results into a new lease attempt for the same job."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from uuid import uuid4

from kraken_core.plugin_protocol import PluginJobManifest
from kraken_core.safe_files import is_link_or_reparse, open_exclusive_write, open_regular_read

from .jobs import StagingWorkspace


def _complete_cif(path: Path) -> bool:
    try:
        with open_regular_read(path) as stream:
            data = stream.read()
    except OSError, ValueError:
        return False
    # Contour writes CIFs in text mode, so Windows output uses CRLF.
    data = data.replace(b"\r\n", b"\n")
    return data.startswith(b"DS 1 1 1;\n") and b"( S " in data[:512] and data.rstrip().endswith(b"DF;\nE")


def _copy_regular(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    try:
        with open_regular_read(source) as reader, open_exclusive_write(temporary) as writer:
            while block := reader.read(1024 * 1024):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def recover_contour_attempt(staging_root: Path, workspace: StagingWorkspace, manifest: PluginJobManifest) -> int:
    """Restore completed CIFs from an identical older local attempt, if any."""
    if manifest.operation != "frames.vectorize.v1":
        return 0
    expected_manifest = (workspace.path / "job.json").read_bytes()
    expected_names = {f"{Path(item.relative_path).stem}.cif" for item in manifest.inputs}
    candidates: list[tuple[int, float, Path, list[Path]]] = []
    for attempt in staging_root.iterdir():
        if attempt == workspace.root or not attempt.is_dir() or is_link_or_reparse(attempt):
            continue
        old = attempt / manifest.job_id
        outputs = old / "outputs"
        old_manifest = old / "job.json"
        if any(is_link_or_reparse(path) for path in (old, outputs, old_manifest)):
            continue
        if not outputs.is_dir() or not old_manifest.is_file():
            continue
        try:
            with open_regular_read(old_manifest) as stream:
                if stream.read(len(expected_manifest) + 1) != expected_manifest:
                    continue
        except OSError:
            continue
        files = [path for path in outputs.iterdir() if path.name in expected_names and _complete_cif(path)]
        if files:
            candidates.append((len(files), attempt.stat().st_mtime, old, files))
    if not candidates:
        return 0
    _count, _modified, old, files = max(candidates, key=lambda candidate: candidate[:2])
    destination_dir = workspace.path / "outputs"
    copied = 0
    for source in files:
        destination = destination_dir / source.name
        if destination.exists() or destination.is_symlink():
            continue
        _copy_regular(source, destination)
        copied += 1
    state_source = old / "contour-resume.json"
    state_destination = workspace.path / "contour-resume.json"
    if not os.path.lexists(state_destination):
        if state_source.is_file() and not is_link_or_reparse(state_source):
            _copy_regular(state_source, state_destination)
        else:
            # Output produced before settings snapshots existed: say so
            # explicitly instead of inventing the lost settings.
            legacy = {
                "version": 1,
                "snapshots": [],
                "current": None,
                "frames": {},
                "unknown_outputs": sorted(path.stem for path in files),
                "complete": False,
            }
            _write_json(state_destination, legacy)
    return copied


def _write_json(destination: Path, payload: dict) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    try:
        with open_exclusive_write(temporary) as writer:
            writer.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def recover_contour_attempt_safely(staging_root: Path, workspace: StagingWorkspace, manifest: PluginJobManifest) -> int:
    """Best-effort reuse: an unreadable old attempt never fails the new lease.

    Copies are atomic, so CIFs already copied before an error stay valid and
    the rest are simply recomputed. Headless runs rewrite every output, so
    reusing old ones would only cost time.
    """
    if manifest.parameters.get("execution_mode") == "headless":
        return 0
    try:
        return recover_contour_attempt(staging_root, workspace, manifest)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not reuse Contour output from an earlier attempt of job %s", manifest.job_id, exc_info=True
        )
        return 0
