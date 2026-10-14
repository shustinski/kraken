"""remote_client: create representation."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
    StructureState,
)


class CreateRepresentationMixin:
    def reorder_layers(
        self,
        *,
        principal: Principal,
        project: Project,
        layers: tuple[Layer, ...] | list[Layer],
        layer_ids: tuple[str, ...],
        idempotency_key: str,
    ) -> tuple[Layer, ...]:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/reorder",
            payload={
                "layer_ids": list(layer_ids),
                "expected_revisions": {
                    str(layer.id): layer.revision for layer in layers
                },
            },
            idempotency_key=idempotency_key,
        )
        values = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            self._layer(project.id, item)
            for item in values
            if isinstance(item, Mapping)
        )

    def archive_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        idempotency_key: str,
    ) -> Layer:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/{layer.id}/archive",
            payload={},
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid layer")
        return self._layer(project.id, payload)

    def list_representations(
        self,
        project_id: object,
        layer_id: object,
        *,
        include_archived: bool = False,
    ) -> tuple[Representation, ...]:
        payload = self._call(
            "GET",
            f"/api/v1/projects/{project_id}/layers/{layer_id}/representations"
            + ("?include_archived=true" if include_archived else ""),
        )
        values = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        representations = tuple(
            self._representation(project_id, layer_id, item)
            for item in values
            if isinstance(item, Mapping)
        )
        if include_archived:
            return representations
        return tuple(
            item
            for item in representations
            if item.state is not StructureState.ARCHIVED
        )

    def create_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        kind: RepresentationKind,
        idempotency_key: str,
        note: str = "",
        source: str | None = None,
        source_image_representation_id: object | None = None,
        active: bool = False,
        purpose: RepresentationPurpose = RepresentationPurpose.SOURCE,
    ) -> Representation:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project.id}/layers/{layer.id}/representations",
            payload={
                "name": name,
                "kind": kind.value,
                "purpose": (
                    RepresentationPurpose.VECTOR.value
                    if kind is RepresentationKind.VECTOR
                    else purpose.value
                ),
                "note": note,
                "source": source,
                "source_image_representation_id": (
                    None
                    if source_image_representation_id is None
                    else str(source_image_representation_id)
                ),
                "active": active,
            },
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid representation")
        return self._representation(project.id, layer.id, payload)

    def _update_representation(
        self,
        *,
        project: Project,
        layer: Layer,
        representation: Representation,
        operation: Mapping[str, object],
        idempotency_key: str,
    ) -> Representation:
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{project.id}/layers/{layer.id}/representations/{representation.id}",
            payload={
                **operation,
                "expected_representation_revision": representation.revision,
            },
            idempotency_key=idempotency_key,
            if_match=layer.revision,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Server returned an invalid representation")
        return self._representation(project.id, layer.id, payload)

    def rename_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        name: str,
        idempotency_key: str,
    ) -> Representation:
        del principal
        return self._update_representation(
            project=project,
            layer=layer,
            representation=representation,
            operation={"name": name},
            idempotency_key=idempotency_key,
        )

    def update_representation_note(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        note: str,
        idempotency_key: str,
    ) -> Representation:
        del principal
        return self._update_representation(
            project=project,
            layer=layer,
            representation=representation,
            operation={"note": note},
            idempotency_key=idempotency_key,
        )
