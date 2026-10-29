"""remote_client: representation."""
from __future__ import annotations
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
    StructureState,
)

from kraken_hub.infrastructure.remote.parts.extract_review_archive import (
    RemoteServerError,
    _layer_file_binding,
    _workspace_layer_result,
)


class RepresentationMixin:
    @staticmethod
    def _representation(
        project_id: object,
        layer_id: object,
        item: Mapping[str, object],
    ) -> Representation:
        created = item.get("created_at")
        kind = RepresentationKind(str(item.get("kind", "image")))
        return Representation(
            id=str(item.get("representation_id") or item.get("id")),
            project_id=ProjectId(str(project_id)),
            layer_id=str(layer_id),
            name=str(item["name"]),
            kind=kind,
            purpose=RepresentationPurpose(
                str(
                    item.get(
                        "purpose",
                        "vector" if kind is RepresentationKind.VECTOR else "source",
                    )
                )
            ),
            note=str(item.get("note", "")),
            source=None if item.get("source") is None else str(item["source"]),
            source_image_representation_id=(
                None
                if item.get("source_image_representation_id") is None
                else str(item["source_image_representation_id"])
            ),
            active=bool(item.get("active", False)),
            state=StructureState(str(item.get("state", "active"))),
            revision=int(item.get("revision", 0)),
            created_at=(
                datetime.fromisoformat(str(created))
                if created
                else datetime.now().astimezone()
            ),
        )

    def create_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        idempotency_key: str,
        layer_id: object | None = None,
    ) -> Layer:
        del principal, layer_id
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers",
            payload={"name": name, "type": layer_type.value, "order": order},
            idempotency_key=idempotency_key,
            if_match=project.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid layer")
        return self._layer(project.id, payload)

    def layer_file_binding(self, project_id: object, layer_id: object):
        try:
            payload = self._call(
                "GET",
                f"/api/v1/projects/{project_id}/layers/{layer_id}/files",
            )
        except RemoteServerError as exc:
            if exc.status == 404:
                return None
            raise
        if not isinstance(payload, Mapping):
            return None
        return _layer_file_binding(payload)

    def scan_layer_source(self, directory: Path | str, *, maximum_frames: int, project_id: object):
        from kraken_manager.workspace import LayerSourceScan

        del maximum_frames  # The server applies the project's authoritative frame limit.
        payload = self._call(
            "POST", f"/api/v1/projects/{project_id}/layers/scan", payload={"directory": str(directory)}
        )
        return LayerSourceScan(
            selected_root=payload["selected_root"], working_directory=payload["working_directory"],
            jpg_files=tuple(payload["jpg_files"]), bmp_files=tuple(payload["bmp_files"]),
            ssc_files=tuple(payload["ssc_files"]), prv_files=tuple(payload["prv_files"]),
            frame_positions=payload["frame_positions"], total_files=payload["total_files"],
            total_bytes=payload["total_bytes"], issues=tuple(payload["issues"]),
            file_fingerprints={key: tuple(value) for key, value in payload["file_fingerprints"].items()},
        )

    def create_external_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        image_directory: Path | str,
        ssc_directory: Path | str | None,
        prv_directory: Path | str | None,
        idempotency_key: str,
    ):
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/external",
            payload={
                "name": name,
                "type": layer_type.value,
                "order": order,
                "image_directory": str(image_directory),
                "ssc_directory": None if not ssc_directory else str(ssc_directory),
                "prv_directory": None if not prv_directory else str(prv_directory),
            },
            idempotency_key=idempotency_key,
        )
        return _workspace_layer_result(self, project.id, payload)

    def create_layer_from_disk(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        scan: object,
        conversion: object,
        idempotency_key: str,
        progress=None,
        cancelled=None,
    ):
        del principal
        if cancelled is not None and cancelled.is_set():
            raise RemoteServerError("Импорт слоя отменён")
        from dataclasses import asdict

        total = int(getattr(scan, "total_files", 0) or 0)
        if progress is not None:
            progress(0, max(1, total), "Копирование на сервере…")
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/import",
            payload={
                "name": name,
                "type": layer_type.value,
                "order": order,
                "scan": asdict(scan),  # type: ignore[arg-type]
                "conversion": asdict(conversion),  # type: ignore[arg-type]
            },
            idempotency_key=idempotency_key,
            timeout=600,
        )
        if progress is not None:
            progress(max(1, total), max(1, total), "Готово")
        return _workspace_layer_result(self, project.id, payload)

    def rename_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        idempotency_key: str,
    ) -> Layer:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/{layer.id}/rename",
            payload={"name": name},
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid layer")
        return self._layer(project.id, payload)
