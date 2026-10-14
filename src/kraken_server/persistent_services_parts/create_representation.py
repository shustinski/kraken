"""persistent_services: create representation."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from kraken_manager.application.dto import (
    CreateRepresentationCommand,
)
from kraken_manager.application.dto import (
    CommandContext as ApplicationCommandContext,
)
from kraken_manager.application.errors import (
    AuthorizationError as ApplicationAuthorizationError,
)
from kraken_manager.application.errors import (
    ConcurrencyError as ApplicationConcurrencyError,
)
from kraken_manager.application.errors import (
    ConflictError as ApplicationConflictError,
)
from kraken_manager.application.errors import (
    NotFoundError as ApplicationNotFoundError,
)
from kraken_manager.application.errors import (
    StorageCapabilityError,
)
from kraken_manager.domain.common import (
    RepresentationId,
)
from kraken_manager.domain.project import (
    RepresentationKind,
    RepresentationPurpose,
)
from kraken_server.services import CommandContext, ConflictError, ForbiddenError, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_id,
    _project_id,
    _representation_dict,
)


class CreateRepresentationMixin:
    def list_representations(
        self, project_id: str, layer_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        try:
            layer = self.projections.get_layer(_layer_id(layer_id))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid layer id") from exc
        if layer is None or str(layer.project_id) != project_id:
            raise NotFoundError(layer_id)
        return [
            _representation_dict(representation)
            for representation in self.projections.list_representations(
                layer.id, include_archived=include_archived
            )
        ]

    def create_representation(
        self,
        project_id: str,
        layer_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        actor = self._actor(context.actor_id)
        try:
            representation = self._create_representation(
                CreateRepresentationCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=context.idempotency_key,
                        gitlab_identity_verified=True,
                    ),
                    project_id=_project_id(project_id),
                    layer_id=_layer_id(layer_id),
                    name=str(payload.get("name", "")),
                    kind=RepresentationKind(str(payload.get("kind", ""))),
                    purpose=RepresentationPurpose(
                        str(
                            payload.get(
                                "purpose",
                                "vector" if str(payload.get("kind", "")) == "vector" else "source",
                            )
                        )
                    ),
                    expected_layer_revision=self._require_revision(context),
                    note=str(payload.get("note", "")),
                    source=None if payload.get("source") is None else str(payload["source"]),
                    source_image_representation_id=(
                        None
                        if payload.get("source_image_representation_id") is None
                        else RepresentationId(str(payload["source_image_representation_id"]))
                    ),
                    active=bool(payload.get("active", False)),
                )
            )
            return _representation_dict(representation)
        except ApplicationNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except (ApplicationConcurrencyError, ApplicationConflictError) as exc:
            raise ConflictError(str(exc)) from exc
        except ApplicationAuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc
        except (StorageCapabilityError, ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc
