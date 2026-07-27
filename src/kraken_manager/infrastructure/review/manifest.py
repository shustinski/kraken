"""Canonical JSON codec for the domain-owned ReviewPackageManifestV1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from kraken_manager.domain.common import PerformerId, PrincipalId, ReviewBatchId
from kraken_manager.domain.workflows import ReviewPackageFileV1, ReviewPackageManifestV1


REVIEW_PACKAGE_SCHEMA = "kraken.review-package.v1"


def manifest_to_dict(manifest: ReviewPackageManifestV1) -> dict[str, Any]:
    return {
        "schema": REVIEW_PACKAGE_SCHEMA,
        "schema_version": manifest.SCHEMA_VERSION,
        "package_id": str(manifest.package_id),
        "batch_id": None if manifest.batch_id is None else str(manifest.batch_id),
        "project_id": str(manifest.project_id),
        "layer_id": str(manifest.layer_id),
        "performer_id": None if manifest.performer_id is None else str(manifest.performer_id),
        "issued_by": None if manifest.issued_by is None else str(manifest.issued_by),
        "issued_at": manifest.issued_at.isoformat(),
        "due_at": None if manifest.due_at is None else manifest.due_at.isoformat(),
        "instructions": manifest.instructions,
        "signature_algorithm": manifest.signature_algorithm,
        "files": [
            {
                "frame_id": str(item.frame_id),
                "artifact_version_id": str(item.artifact_version_id),
                "sha256": item.sha256,
                "relative_path": item.relative_path,
                "x": item.x,
                "y": item.y,
                "role": item.role,
            }
            for item in manifest.files
        ],
    }


def canonical_manifest_json(manifest: ReviewPackageManifestV1) -> str:
    return json.dumps(manifest_to_dict(manifest), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_digest(manifest: ReviewPackageManifestV1) -> str:
    return hashlib.sha256(canonical_manifest_json(manifest).encode("utf-8")).hexdigest()


def manifest_from_dict(payload: Mapping[str, Any]) -> ReviewPackageManifestV1:
    if payload.get("schema") != REVIEW_PACKAGE_SCHEMA or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported review package schema")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("Review package files must be an array")
    files: list[ReviewPackageFileV1] = []
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid review package file")
        files.append(
            ReviewPackageFileV1(
                frame_id=str(item.get("frame_id", "")),
                artifact_version_id=str(item.get("artifact_version_id", "")),
                sha256=str(item.get("sha256", "")),
                relative_path=str(item.get("relative_path", "")),
                x=None if item.get("x") is None else int(item["x"]),
                y=None if item.get("y") is None else int(item["y"]),
                role=str(item.get("role", "vector")),
            )
        )
    due_at = payload.get("due_at")
    return ReviewPackageManifestV1(
        package_id=ReviewBatchId(str(payload.get("package_id", ""))),
        batch_id=None if payload.get("batch_id") is None else ReviewBatchId(str(payload["batch_id"])),
        project_id=str(payload.get("project_id", "")),
        layer_id=str(payload.get("layer_id", "")),
        performer_id=None
        if payload.get("performer_id") is None
        else PerformerId(str(payload["performer_id"])),
        issued_by=None if payload.get("issued_by") is None else PrincipalId(str(payload["issued_by"])),
        issued_at=datetime.fromisoformat(str(payload.get("issued_at", "")).replace("Z", "+00:00")),
        due_at=None if due_at is None else datetime.fromisoformat(str(due_at).replace("Z", "+00:00")),
        instructions=str(payload.get("instructions", "")),
        files=tuple(files),
        signature_algorithm=str(payload.get("signature_algorithm", "")),
    )


def manifest_from_json(raw: str) -> ReviewPackageManifestV1:
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("Review package manifest must be an object")
    return manifest_from_dict(payload)


__all__ = [
    "REVIEW_PACKAGE_SCHEMA",
    "ReviewPackageFileV1",
    "ReviewPackageManifestV1",
    "canonical_manifest_json",
    "manifest_digest",
    "manifest_from_dict",
    "manifest_from_json",
    "manifest_to_dict",
]
