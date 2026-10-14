"""persistent_services: update representation."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from kraken_manager.application.dto import (
    ActivateRepresentationCommand,
    ArchiveRepresentationCommand,
    DeactivateRepresentationCommand,
    RenameRepresentationCommand,
    UpdateRepresentationNoteCommand,
)
from kraken_manager.domain.common import (
    RepresentationId,
    validate_uuid,
)
from kraken_server.services import CommandContext, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_id,
    _project_id,
    _representation_dict,
)


class UpdateRepresentationMixin:
    def update_representation(
        self,
        project_id: str,
        layer_id: str,
        representation_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        operations = [key for key in ("name", "note", "active", "archive") if key in payload]
        if len(operations) != 1:
            raise ValidationError("Exactly one representation operation is required")
        try:
            common = {
                "context": self._application_context(context),
                "project_id": _project_id(project_id),
                "layer_id": _layer_id(layer_id),
                "representation_id": RepresentationId(
                    validate_uuid(representation_id, field="representation_id")
                ),
                "expected_layer_revision": self._require_revision(context),
                "expected_representation_revision": int(
                    payload.get("expected_representation_revision", -1)
                ),
            }
            operation = operations[0]
            if operation == "name":
                value = self._rename_representation(
                    RenameRepresentationCommand(name=str(payload["name"]), **common)
                )
            elif operation == "note":
                value = self._update_representation_note(
                    UpdateRepresentationNoteCommand(note=str(payload["note"]), **common)
                )
            elif operation == "active":
                value = (
                    self._activate_representation(
                        ActivateRepresentationCommand(**common)
                    )
                    if bool(payload["active"])
                    else self._deactivate_representation(
                        DeactivateRepresentationCommand(**common)
                    )
                )
            else:
                if not bool(payload["archive"]):
                    raise ValueError("archive must be true")
                value = self._archive_representation(ArchiveRepresentationCommand(**common))
            return _representation_dict(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")
